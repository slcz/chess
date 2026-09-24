/*
 * test_nn.c — C unit tests for the Phase 2 input encoder (chess_nn.h).
 *
 * Deliberately a separate binary from tests/test_units.c (the Phase 1 rules
 * gate): NN-side tests must be able to evolve without touching the rules
 * test surface. Same hand-rolled CHECK harness style.
 *
 * What is verified here (the deeper cross-checks — a full Python reference
 * implementation compared tensor-for-tensor over random games — live in
 * tests/test_nn_features.py):
 *   1. purity: encode leaves the State bit-for-bit unchanged
 *   2. startpos tensor: placement decode, empty history, globals
 *   3. mover perspective: board flip + color reorder after 1.e4
 *   4. history frames land in the right slots (h = 1 … 7), zero before
 *      the game start
 *   5. repetition flags on a known shuffle repetition
 *   6. ep file, castling-bit reorder, halfmove scaling
 *   7. ring push/pop (record/drop) semantics
 */
#include <stdio.h>
#include <string.h>

#include "chess.h"
#include "chess_nn.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (cond) {                                                         \
            g_pass++;                                                       \
        } else {                                                            \
            g_fail++;                                                       \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);          \
        }                                                                   \
    } while (0)

/* ---- helpers ---------------------------------------------------------- */

/* Find `uci` among the legal moves and make it, recording the arrival
 * position into the history ring — exactly the sequence the FFI handle
 * performs on push. Returns false (and does nothing) if not legal. */
static bool play(State *s, NnHistory *h, const char *uci) {
    MoveList ml;
    gen_legal(s, &ml);
    for (int i = 0; i < ml.count; i++) {
        char u[8];
        move_to_uci(ml.moves[i], u);
        if (strcmp(u, uci) == 0) {
            make_move(s, ml.moves[i]);
            nn_history_record(h, s);
            return true;
        }
    }
    return false;
}

/* Frame-h piece-slot of `pc` seen by `mover`, at the token of square `sq`. */
static float at(const float *t, int mover, int sq, int frame, int pc) {
    int token = (mover == BLACK) ? (sq ^ 56) : sq;
    int dim = frame * 12
            + (pc_color(pc) == mover ? pc_type(pc) : 6 + pc_type(pc));
    return t[token * NN_TOKEN_DIM + dim];
}

static int frame_hot_count(const float *t, int frame) {
    int hot = 0;
    for (int tok = 0; tok < NN_TOKENS; tok++)
        for (int d = 0; d < 12; d++)
            if (t[tok * NN_TOKEN_DIM + frame * 12 + d] != 0.0f) hot++;
    return hot;
}

/* ---- tests ------------------------------------------------------------ */

static void test_purity_and_startpos(void) {
    State s;
    NnHistory h;
    static float t[NN_TOTAL];

    state_init_startpos(&s);
    nn_history_reset(&h);
    nn_history_record(&h, &s);

    State before = s;
    nn_encode(&s, &h, t);
    CHECK(memcmp(&s, &before, sizeof s) == 0); /* encoder is read-only */

    /* Frame 0: every occupied square decodes to the right piece slot, and
     * nothing else is set. */
    for (int sq = 0; sq < 64; sq++) {
        int pc = s.board[sq];
        for (int slot = 0; slot < 12; slot++) {
            float v = t[sq * NN_TOKEN_DIM + slot]; /* mover is WHITE: token==sq */
            CHECK((pc != EMPTY && slot == pc) ? v == 1.0f : v == 0.0f);
        }
    }
    CHECK(frame_hot_count(t, 0) == 32);
    for (int hi = 1; hi < NN_HISTORY; hi++) /* no history yet */
        CHECK(frame_hot_count(t, hi) == 0);

    /* Globals at startpos: full castling, no ep, no repetition, clock 0. */
    for (int tok = 0; tok < NN_TOKENS; tok++) {
        const float *row = t + tok * NN_TOKEN_DIM;
        CHECK(row[112] == 1.0f && row[113] == 1.0f);
        CHECK(row[114] == 1.0f && row[115] == 1.0f);
        CHECK(row[116] == 0.0f);
        for (int d = 96; d < 112; d++) CHECK(row[d] == 0.0f);
    }
}

static void test_perspective_flip_after_e4(void) {
    State s;
    NnHistory h;
    static float t[NN_TOTAL];

    state_init_startpos(&s);
    nn_history_reset(&h);
    nn_history_record(&h, &s);
    CHECK(play(&s, &h, "e2e4"));

    nn_encode(&s, &h, t);
    CHECK(s.side == BLACK);

    /* Current frame, BLACK mover: black pawn e7 is "ours" (slot PAWN=0) on
     * the flipped token; white pawn e4 is "enemy" (slot 6+PAWN). The white
     * pawn's origin e2 must be empty in frame 0 (token = E2 ^ 56). */
    CHECK(at(t, BLACK, E7, 0, pc_of(BLACK, PAWN)) == 1.0f);
    CHECK(at(t, BLACK, E4, 0, pc_of(WHITE, PAWN)) == 1.0f);
    CHECK(t[(E2 ^ 56) * NN_TOKEN_DIM + 6 + PAWN] == 0.0f);

    /* History frame 1 = startpos from BLACK's view: the white pawn is back
     * on e2 as an enemy piece. */
    CHECK(at(t, BLACK, E2, 1, pc_of(WHITE, PAWN)) == 1.0f);
    CHECK(at(t, BLACK, E7, 1, pc_of(BLACK, PAWN)) == 1.0f);
    CHECK(frame_hot_count(t, 1) == 32);

    /* ep after e2e4: target e3, file 4 — one-hot in every token. */
    for (int tok = 0; tok < NN_TOKENS; tok++) {
        const float *row = t + tok * NN_TOKEN_DIM;
        for (int f = 0; f < 8; f++) CHECK(row[104 + f] == (f == 4 ? 1.0f : 0.0f));
    }
}

static void test_history_frames(void) {
    State s;
    NnHistory h;
    static float t[NN_TOTAL];

    state_init_startpos(&s);
    nn_history_reset(&h);
    nn_history_record(&h, &s);
    CHECK(play(&s, &h, "e2e4"));
    CHECK(play(&s, &h, "e7e5"));
    CHECK(play(&s, &h, "g1f3"));
    nn_encode(&s, &h, t);

    /* ply 3, WHITE moved last => mover BLACK, flip on. */
    CHECK(s.side == BLACK);
    /* frame 1: after e4 e5 — black pawn on e5 (ours), none on e7. */
    CHECK(at(t, BLACK, E5, 1, pc_of(BLACK, PAWN)) == 1.0f);
    CHECK(at(t, BLACK, E7, 1, pc_of(BLACK, PAWN)) == 0.0f);
    /* frame 2: after e4 only — black pawn still e7, e5 empty. */
    CHECK(at(t, BLACK, E7, 2, pc_of(BLACK, PAWN)) == 1.0f);
    CHECK(at(t, BLACK, E5, 2, pc_of(BLACK, PAWN)) == 0.0f);
    /* frame 3: startpos; frames 4..7: before the game => all zero. */
    CHECK(at(t, BLACK, E7, 3, pc_of(BLACK, PAWN)) == 1.0f);
    for (int hi = 4; hi < NN_HISTORY; hi++) CHECK(frame_hot_count(t, hi) == 0);
    for (int hi = 0; hi < 4; hi++) CHECK(frame_hot_count(t, hi) == 32);

    /* halfmove clock 3 / 100 in every token's last dim. */
    for (int tok = 0; tok < NN_TOKENS; tok++) {
        float want = (float)s.halfmove / 100.0f;
        CHECK(t[tok * NN_TOKEN_DIM + 116] == want);
    }
}

static void test_repetition_flags(void) {
    State s;
    NnHistory h;
    static float t[NN_TOTAL];

    state_init_startpos(&s);
    nn_history_reset(&h);
    nn_history_record(&h, &s);
    /* Knights out and back: position after ply 4 equals position 0. */
    CHECK(play(&s, &h, "g1f3"));
    CHECK(play(&s, &h, "g8f6"));
    CHECK(play(&s, &h, "f3g1"));
    CHECK(play(&s, &h, "f6g8"));
    nn_encode(&s, &h, t);

    /* keys[4] == keys[0]: the CURRENT position (frame h=0) has occurred
     * before -> flag 1. Every other window position is appearing for the
     * first time; position 0 itself (h=4) has nothing earlier to repeat. */
    for (int tok = 0; tok < NN_TOKENS; tok++) {
        const float *row = t + tok * NN_TOKEN_DIM;
        CHECK(row[96 + 0] == 1.0f);
        for (int hi = 1; hi < NN_HISTORY; hi++) CHECK(row[96 + hi] == 0.0f);
    }
}

static void test_castling_reorder(void) {
    State s;
    NnHistory h;
    static float t[NN_TOTAL];

    /* Rights "Kq" (white kingside + black queenside), BLACK to move:
     * mover-first slots must read [BK=0, BQ=1, WK=1, WQ=0]. */
    CHECK(fen_parse(&s, "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1") == 0);
    nn_history_reset(&h);
    nn_history_record(&h, &s);
    nn_encode(&s, &h, t);
    for (int tok = 0; tok < NN_TOKENS; tok++) {
        const float *row = t + tok * NN_TOKEN_DIM;
        CHECK(row[112] == 0.0f && row[113] == 1.0f);
        CHECK(row[114] == 1.0f && row[115] == 0.0f);
    }

    /* Same rights, WHITE to move: [WK=1, WQ=0, BK=0, BQ=1]. */
    CHECK(fen_parse(&s, "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1") == 0);
    nn_encode(&s, &h, t);
    for (int tok = 0; tok < NN_TOKENS; tok++) {
        const float *row = t + tok * NN_TOKEN_DIM;
        CHECK(row[112] == 1.0f && row[113] == 0.0f);
        CHECK(row[114] == 0.0f && row[115] == 1.0f);
    }
}

static void test_ring_pop_semantics(void) {
    State s;
    NnHistory h;
    static float t[NN_TOTAL];
    Move m1, m2;
    MoveList ml;

    state_init_startpos(&s);
    nn_history_reset(&h);
    nn_history_record(&h, &s);

    /* Locate d2d4 / d7d5 by UCI, then drive state+ring like the FFI. */
    gen_legal(&s, &ml);
    m1 = m2 = 0;
    for (int i = 0; i < ml.count; i++) {
        char u[8];
        move_to_uci(ml.moves[i], u);
        if (strcmp(u, "d2d4") == 0) m1 = ml.moves[i];
    }
    make_move(&s, m1);
    nn_history_record(&h, &s);
    gen_legal(&s, &ml);
    for (int i = 0; i < ml.count; i++) {
        char u[8];
        move_to_uci(ml.moves[i], u);
        if (strcmp(u, "d7d5") == 0) m2 = ml.moves[i];
    }
    make_move(&s, m2);
    nn_history_record(&h, &s);

    /* Pop back to ply 1 (exactly what chess_pop does), then encode:
     * frame 1 must be the START position — ring = [pos0, pos1], and the
     * start position is the one with the white pawn still on d2 (the
     * position after 1.d4 does NOT have it, so this pins the ring index). */
    nn_history_drop(&h);
    unmake_move(&s, m2);
    nn_encode(&s, &h, t);
    CHECK(s.side == BLACK);
    CHECK(at(t, BLACK, D2, 1, pc_of(WHITE, PAWN)) == 1.0f);
    CHECK(at(t, BLACK, D4, 1, pc_of(WHITE, PAWN)) == 0.0f);
    CHECK(frame_hot_count(t, 2) == 0); /* no frame 2 exists at ply 1 */

    /* Pop to ply 0: ring = [pos0], no history frames at all. */
    nn_history_drop(&h);
    unmake_move(&s, m1);
    nn_encode(&s, &h, t);
    CHECK(s.side == WHITE);
    for (int hi = 1; hi < NN_HISTORY; hi++) CHECK(frame_hot_count(t, hi) == 0);
}

static void test_all_values_are_featurish(void) {
    /* Over a short random game: dims [0,112) of every token must be exactly
     * 0.0 or 1.0; dim 116 must be a plausible clock/100. (Cheap invariant
     * sweep that catches stray writes from the ring indexing.) */
    State s;
    NnHistory h;
    static float t[NN_TOTAL];

    state_init_startpos(&s);
    nn_history_reset(&h);
    nn_history_record(&h, &s);

    uint64_t rng = 0xC0FFEE123456789ULL;
    for (int ply = 0; ply < 40 && state_termination(&s) == TERM_PLAYING; ply++) {
        MoveList ml;
        gen_legal(&s, &ml);
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        make_move(&s, ml.moves[(rng >> 33) % (uint64_t)ml.count]);
        nn_history_record(&h, &s);

        nn_encode(&s, &h, t);
        for (int tok = 0; tok < NN_TOKENS; tok++) {
            const float *row = t + tok * NN_TOKEN_DIM;
            for (int d = 0; d < 112; d++)
                CHECK(row[d] == 0.0f || row[d] == 1.0f);
            CHECK(row[116] >= 0.0f && row[116] <= 1.0f);
        }
    }
}

int main(void) {
    chess_init();
    test_purity_and_startpos();
    test_perspective_flip_after_e4();
    test_history_frames();
    test_repetition_flags();
    test_castling_reorder();
    test_ring_pop_semantics();
    test_all_values_are_featurish();

    printf("test_nn: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
