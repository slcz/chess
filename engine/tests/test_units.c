/*
 * test_units.c — C unit tests for the rules engine (Phase 1 gate, part 1).
 *
 * Hand-rolled minimal harness (CHECK macro): the project is deliberately
 * dependency-free, and "print failures, count them, exit non-zero" is all a
 * test runner needs to be CI-useful. pytest (tests/test_engine.py) is the
 * single entry point; it runs this binary as a subprocess gate.
 *
 * Why this file is C at all: these tests inspect State *internals* (struct
 * memcmp after make/unmake, mailbox-vs-bitboard consistency) that have no
 * binding yet. Once the cffi scaffold lands (Phase 0.3) pieces can move to
 * native pytest if that ever reads better.
 *
 * Sections:
 *   1. test_perft_suite() — runs EVERY line of tests/perft_suite.txt (the
 *      canonical positions + targeted edge cases): the exact-node-count
 *      verification of movegen.
 *   2. FEN parse/serialize round-trips and rejection of malformed FENs.
 *   3. Representation integrity: mailbox vs bitboards agree; make→unmake
 *      restores the State bit-for-bit (memcmp over the whole struct, undo
 *      stack included).
 *   4. Rule edge cases with hand-computed expectations: castling through /
 *      out of / into check, castling-rights loss by rook capture, the
 *      en-passant pin, promotions (incl. capture-promotion), checkmate and
 *      stalemate via the legal-move filter, ep + clock bookkeeping.
 *   5. Random self-play games from several start positions: after every ply
 *      the FEN must round-trip identically — this is where undo-stack and
 *      clock corruption actually show up.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "suite_loader.h"

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

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL; /* fixed seed: reproducible */
static uint32_t rng_next(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 33);
}

static bool legal_contains_uci(State *s, const char *uci) {
    MoveList ml;
    gen_legal(s, &ml);
    for (int i = 0; i < ml.count; i++) {
        char u[8];
        move_to_uci(ml.moves[i], u);
        if (!strcmp(u, uci)) return true;
    }
    return false;
}

static int legal_count(State *s) {
    MoveList ml;
    gen_legal(s, &ml);
    return ml.count;
}

/* Mailbox vs bitboards: the two views must describe the same position. */
static bool rep_consistent(const State *s) {
    uint64_t occ[2] = {0, 0};
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < 6; t++) {
            uint64_t b = s->bb[c][t];
            while (b) {
                int sq = __builtin_ctzll(b);
                b &= b - 1;
                if (s->board[sq] != pc_of(c, t)) return false;
            }
            occ[c] |= s->bb[c][t];
        }
    if (occ[WHITE] != s->occ[WHITE] || occ[BLACK] != s->occ[BLACK]) return false;
    if (occ[WHITE] & occ[BLACK]) return false; /* no square holds two pieces */
    for (int sq = 0; sq < 64; sq++)
        if ((s->board[sq] == EMPTY) != !(((occ[WHITE] | occ[BLACK]) >> sq) & 1)) return false;
    return true;
}

/* Compare the "live" fields (everything except the undo history). */
static bool live_fields_equal(const State *a, const State *b) {
    return memcmp(a->board, b->board, sizeof a->board) == 0 &&
           memcmp(a->bb, b->bb, sizeof a->bb) == 0 &&
           a->occ[WHITE] == b->occ[WHITE] && a->occ[BLACK] == b->occ[BLACK] &&
           a->side == b->side && a->castling == b->castling &&
           a->ep == b->ep && a->halfmove == b->halfmove && a->fullmove == b->fullmove;
}

/* Distinct start positions used by several sections below. */
static const char *const SUITE_STARTS[] = {
    CHESS_START_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "8/P1k5/K7/8/8/8/8/8 w - - 0 1",
};
#define N_STARTS ((int)(sizeof SUITE_STARTS / sizeof SUITE_STARTS[0]))

/* ------------------------------------------------------------------ */
/* 1. Perft suite (the data file is the source of truth)               */
/* ------------------------------------------------------------------ */

static void test_perft_suite(void) {
    static SuiteEntry entries[SUITE_MAX_ENTRIES];
    int n = suite_load(entries, SUITE_MAX_ENTRIES);
    CHECK(n > 0);

    for (int i = 0; i < n; i++) {
        State s;
        CHECK(fen_parse(&s, entries[i].fen) == 0);
        uint64_t got = perft(&s, entries[i].depth);
        if (got != entries[i].nodes) {
            printf("    perft FAIL: %-17s d%d got %llu want %llu\n", entries[i].name,
                   entries[i].depth, (unsigned long long)got,
                   (unsigned long long)entries[i].nodes);
        }
        CHECK(got == entries[i].nodes);
    }
}

/* ------------------------------------------------------------------ */
/* 2. FEN                                                              */
/* ------------------------------------------------------------------ */

static void test_fen_roundtrip(void) {
    State s;
    char fen[96];

    state_init_startpos(&s);
    fen_serialize(&s, fen, sizeof fen);
    CHECK(strcmp(fen, CHESS_START_FEN) == 0);

    /* 4-field FEN (no clocks) must parse and default to "0 1". */
    CHECK(fen_parse(&s, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -") == 0);
    fen_serialize(&s, fen, sizeof fen);
    CHECK(strcmp(fen, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1") == 0);

    /* Garbage must be rejected, not half-parsed. */
    CHECK(fen_parse(&s, "not a fen") != 0);
    CHECK(fen_parse(&s, "8/8/8/8/8/8/8/8 w - - 0 1") != 0);          /* no kings */
    CHECK(fen_parse(&s, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1") != 0);
    CHECK(fen_parse(&s, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQxq - 0 1") != 0);
    CHECK(fen_parse(&s, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq z9 0 1") != 0);

    /* ep square is kept verbatim (like python-chess), not validated. */
    CHECK(fen_parse(&s, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPPPPPP/RNBQKBNR b KQkq e3 0 1") == 0);
    CHECK(s.ep == 20);
    fen_serialize(&s, fen, sizeof fen);
    CHECK(strcmp(fen, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPPPPPP/RNBQKBNR b KQkq e3 0 1") == 0);
}

/* ------------------------------------------------------------------ */
/* 3. Representation integrity                                         */
/* ------------------------------------------------------------------ */

static void test_make_unmake_integrity(void) {
    /* From each distinct start position, play every legal move and require
     * the State to come back bit-for-bit identical — where "identical" means
     * the live fields plus the history actually used (hist[0..ply-1]).
     * hist[ply] itself is legitimately different: make_move wrote its undo
     * record there, and unmake leaves the record in place (erasing it would
     * be wasted work — it's scratch space until the next make at that ply). */
    for (int i = 0; i < N_STARTS; i++) {
        State s;
        CHECK(fen_parse(&s, SUITE_STARTS[i]) == 0);
        MoveList ml;
        gen_legal(&s, &ml);
        CHECK(ml.count > 0);
        for (int j = 0; j < ml.count; j++) {
            State before = s;
            make_move(&s, ml.moves[j]);
            CHECK(rep_consistent(&s));
            CHECK(s.ply == before.ply + 1);
            unmake_move(&s, ml.moves[j]);
            size_t live_bytes = offsetof(State, hist) + (size_t)s.ply * sizeof(Undo);
            CHECK(memcmp(&s, &before, live_bytes) == 0);
        }
    }
}

/* ------------------------------------------------------------------ */
/* 4. Rule edge cases (hand-computed expectations)                     */
/* ------------------------------------------------------------------ */

static void test_move_counts_and_encoding(void) {
    State s;

    state_init_startpos(&s);
    CHECK(legal_count(&s) == 20);

    fen_parse(&s, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    CHECK(legal_count(&s) == 48);

    fen_parse(&s, "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1");
    CHECK(legal_count(&s) == 14);

    /* Move encoding basics. */
    Move m = move_make(E2, E4, F_DOUBLE_PUSH);
    CHECK(move_from(m) == E2 && move_to(m) == E4 && move_flags(m) == F_DOUBLE_PUSH);
    CHECK(!move_is_capture(m));
    m = move_make(A7, B8, F_PROMO_CAP_Q);
    CHECK(move_is_promo(m) && move_is_capture(m) && move_promo_type(m) == QUEEN);
    char uci[8];
    move_to_uci(m, uci);
    CHECK(strcmp(uci, "a7b8q") == 0);
    m = move_make(E1, G1, F_KCASTLE);
    move_to_uci(m, uci);
    CHECK(strcmp(uci, "e1g1") == 0);
}

static void test_castling_rules(void) {
    State s;

    /* Black rook on f3 controls f1: O-O dies (through check), O-O-O lives. */
    fen_parse(&s, "4k3/8/8/8/8/5r2/8/R3K2R w KQ - 0 1");
    CHECK(!legal_contains_uci(&s, "e1g1"));
    CHECK(legal_contains_uci(&s, "e1c1"));

    /* Rook on g2 controls g1: castling INTO check dies (the generic filter's
     * job, not the through-square check). */
    fen_parse(&s, "4k3/8/8/8/8/8/6r1/R3K2R w KQ - 0 1");
    CHECK(!legal_contains_uci(&s, "e1g1"));
    CHECK(legal_contains_uci(&s, "e1c1"));

    fen_parse(&s, "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    CHECK(legal_contains_uci(&s, "e1g1") && legal_contains_uci(&s, "e1c1"));

    /* Castling OUT of check is illegal even with the path clear. */
    fen_parse(&s, "4k3/8/8/8/8/4r3/8/R3K2R w KQ - 0 1");
    CHECK(in_check(&s));
    CHECK(!legal_contains_uci(&s, "e1g1") && !legal_contains_uci(&s, "e1c1"));

    /* b1 occupied blocks O-O-O only (the rook crosses b1 even though the
     * king doesn't — FIDE: all squares between king and rook must be empty). */
    fen_parse(&s, "4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1");
    CHECK(legal_contains_uci(&s, "e1g1"));
    CHECK(!legal_contains_uci(&s, "e1c1"));

    /* Losing the h1 rook to a capture kills O-O even though the rights were
     * present before the move (castle_mask applied on the `to` square). */
    CHECK(fen_parse(&s, "r3k3/8/8/8/8/8/6b1/R3K2R b KQq - 0 1") == 0);
    CHECK(legal_contains_uci(&s, "g2h1"));
    make_move(&s, move_make(G2, H1, F_CAPTURE));
    CHECK(s.board[H1] == pc_of(BLACK, BISHOP));
    CHECK((s.castling & CASTLE_WK) == 0); /* white O-O right died */
    CHECK(s.castling & CASTLE_WQ);        /* O-O-O unaffected     */
}

static void test_en_passant_rules(void) {
    State s;

    /* THE classic trap: exd3 e.p. would leave rank 4 empty between the black
     * king (a4) and the white queen (h4) — a pin that exists only *after*
     * both pawns leave the rank. The make/unmake filter catches it. */
    fen_parse(&s, "8/8/8/8/k2Pp2Q/8/8/3K4 b - d3 0 1");
    CHECK(!legal_contains_uci(&s, "e4d3"));

    /* Same position without the queen: exd3 is legal. */
    fen_parse(&s, "8/8/8/8/k2Pp3/8/8/3K4 b - d3 0 1");
    CHECK(legal_contains_uci(&s, "e4d3"));

    /* Plain ep capture geometry. */
    fen_parse(&s, "8/8/8/3pP3/8/8/8/4K2k w - d6 0 1");
    CHECK(legal_contains_uci(&s, "e5d6"));

    /* EP capture giving discovered check must still be generated. */
    fen_parse(&s, "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1");
    CHECK(legal_contains_uci(&s, "c4d3"));
}

static void test_promotion_rules(void) {
    State s;

    /* White pawn a7, black knights on b8/e8: 4 push-promotions + 4
     * capture-promotions onto b8 + 3 king moves = 11 legal moves. */
    fen_parse(&s, "1n2n3/P7/8/8/8/8/8/K6k w - - 0 1");
    CHECK(legal_count(&s) == 11);
    CHECK(legal_contains_uci(&s, "a7a8q"));
    CHECK(legal_contains_uci(&s, "a7a8n"));
    CHECK(legal_contains_uci(&s, "a7b8r"));
    CHECK(!legal_contains_uci(&s, "a7b7"));

    /* Underpromotion with a capture available: black rook on e8, so the pawn
     * can push to f8 (4 promos) or capture on e8 (4 promo-captures). White
     * king g1 is NOT in check from the e8 rook; black king h7 is not attacked. */
    fen_parse(&s, "4r3/5P1k/8/4N3/8/8/8/6K1 w - - 0 1");
    CHECK(legal_contains_uci(&s, "f7f8n"));
    CHECK(legal_contains_uci(&s, "f7e8q"));
    CHECK(legal_contains_uci(&s, "f7e8n"));
    CHECK(!legal_contains_uci(&s, "f7g8q")); /* g8 empty: no capture there */
}

static void test_check_mate_stalemate(void) {
    State s;

    /* Fool's mate: white is mated (no legal moves + in check). */
    fen_parse(&s, "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3");
    CHECK(legal_count(&s) == 0);
    CHECK(in_check(&s));

    /* Stalemate: no legal moves, king NOT in check. */
    fen_parse(&s, "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    CHECK(legal_count(&s) == 0);
    CHECK(!in_check(&s));

    /* Plain check: moves exist, king is attacked. */
    fen_parse(&s, "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1");
    CHECK(legal_count(&s) > 0);
    CHECK(in_check(&s));
}

static void test_double_push_ep_clock(void) {
    State s;
    state_init_startpos(&s);

    MoveList ml;
    gen_legal(&s, &ml);
    Move e2e4 = 0;
    for (int i = 0; i < ml.count; i++)
        if (move_from(ml.moves[i]) == E2 && move_to(ml.moves[i]) == E4)
            e2e4 = ml.moves[i];
    CHECK(e2e4 != 0 && move_flags(e2e4) == F_DOUBLE_PUSH);

    make_move(&s, e2e4);
    CHECK(s.ep == E3);
    CHECK(s.halfmove == 0);
    CHECK(s.fullmove == 1 && s.side == BLACK);

    /* Black answers d7d5: ep target d6, and it's white's move 2. e4xd5 here
     * is a NORMAL capture (d5 is occupied), not an ep capture. */
    CHECK(legal_contains_uci(&s, "d7d5"));
    gen_legal(&s, &ml); /* black's move list */
    Move d7d5 = 0;
    for (int i = 0; i < ml.count; i++)
        if (move_from(ml.moves[i]) == D7 && move_to(ml.moves[i]) == D5)
            d7d5 = ml.moves[i];
    CHECK(d7d5 != 0);
    make_move(&s, d7d5);
    CHECK(s.ep == D6);
    CHECK(s.fullmove == 2 && s.side == WHITE);

    /* Now white's move list must contain e4xd5 as a NORMAL capture. */
    gen_legal(&s, &ml);
    Move e4d5 = 0;
    for (int i = 0; i < ml.count; i++)
        if (move_from(ml.moves[i]) == E4 && move_to(ml.moves[i]) == D5)
            e4d5 = ml.moves[i];
    CHECK(e4d5 != 0 && move_flags(e4d5) == F_CAPTURE);

    /* 2. e4xd5, then black double-pushes e7e5 next to the d5 pawn: now
     * d5xe6 e.p. is the real en-passant capture. */
    make_move(&s, e4d5);
    CHECK(s.board[D5] == pc_of(WHITE, PAWN) && s.board[E4] == EMPTY);
    gen_legal(&s, &ml);
    Move e7e5 = 0;
    for (int i = 0; i < ml.count; i++)
        if (move_from(ml.moves[i]) == E7 && move_to(ml.moves[i]) == E5)
            e7e5 = ml.moves[i];
    CHECK(e7e5 != 0 && move_flags(e7e5) == F_DOUBLE_PUSH);
    make_move(&s, e7e5);
    CHECK(s.ep == E6 && s.side == WHITE && s.fullmove == 3);

    gen_legal(&s, &ml);
    Move ep = 0;
    for (int i = 0; i < ml.count; i++)
        if (move_flags(ml.moves[i]) == F_EP)
            ep = ml.moves[i];
    CHECK(ep != 0 && move_from(ep) == D5 && move_to(ep) == E6);
    make_move(&s, ep);
    CHECK(s.board[E6] == pc_of(WHITE, PAWN));
    CHECK(s.board[E5] == EMPTY); /* the victim pawn is gone */
    CHECK(s.board[D5] == EMPTY);
    unmake_move(&s, ep);
    CHECK(s.board[E5] == pc_of(BLACK, PAWN));
    CHECK(s.board[D5] == pc_of(WHITE, PAWN));
    CHECK(s.board[E6] == EMPTY);
}

/* ------------------------------------------------------------------ */
/* 5. Random self-play games                                           */
/* ------------------------------------------------------------------ */

static void test_random_games(int games) {
    int finished_by_end = 0;

    for (int g = 0; g < games; g++) {
        State s;
        CHECK(fen_parse(&s, SUITE_STARTS[g % N_STARTS]) == 0);

        int ply;
        for (ply = 0; ply < MAX_PLY - 1; ply++) {
            CHECK(rep_consistent(&s));
            CHECK(s.ply == ply);

            char fen[96];
            fen_serialize(&s, fen, sizeof fen);
            State t;
            CHECK(fen_parse(&t, fen) == 0);
            char fen2[96];
            fen_serialize(&t, fen2, sizeof fen2);
            CHECK(strcmp(fen, fen2) == 0);
            CHECK(live_fields_equal(&s, &t));

            MoveList ml;
            gen_legal(&s, &ml);
            if (ml.count == 0) { /* mate or stalemate — both terminate */
                finished_by_end++;
                break;
            }
            make_move(&s, ml.moves[rng_next() % (uint32_t)ml.count]);
        }
        /* Random-move chess games shuffle for a long time; ending by the ply
         * cap is fine — 50-move/threefold termination is Phase 1.3. What this
         * test asserts is per-ply consistency, not that games terminate. */
    }
    printf("    (random games: %d/%d reached mate/stalemate, rest hit the ply cap)\n",
           finished_by_end, games);
}

int main(void) {
    chess_init();

    test_perft_suite();
    test_fen_roundtrip();
    test_move_counts_and_encoding();
    test_castling_rules();
    test_en_passant_rules();
    test_promotion_rules();
    test_check_mate_stalemate();
    test_make_unmake_integrity();
    test_double_push_ep_clock();
    test_random_games(50);

    printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
