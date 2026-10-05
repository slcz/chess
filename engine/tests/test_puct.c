/* test_puct.c — C gate for the Phase 1 PUCT search (chess_puct.h).
 *
 * Covered here (the Python-parity and real-net items of the plan's gate
 * list wait for the real evaluator):
 *   1. chess_policy_slot known answers (startpos, castling, both colors'
 *      promotions incl. capture-promos, synthetic enemy-region slot)
 *   2. mate-in-1 pin: fool's-mate position, best move d8h4, Q = +1,
 *      terminals never hit the evaluator (call count < sims)
 *   3. bookkeeping: priors sum to 1; N(root) = sum N(children);
 *      W(root) = -sum W(children)  (W is in the parent-mover's view)
 *   4. value sign convention: an evaluator that always says "mover wins"
 *      must drive every root child's Q to -1
 *   5. prior delivery: a slot-patterned logp arrives as the exact softmax
 *   6. mask fuzz: puct_request_mask == gen_legal x chess_policy_slot over
 *      a random game; slots legal-range and pairwise distinct
 *   7. OOM give-up: tiny block, think aborts, counter set, game playable
 *   8. mate-in-2 ladder: mate proven (Q > 0.9), tree persists across
 *      puct_play, final move delivers checkmate
 * Run with "bench" to print stub-eval sims/s from the start position.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "chess.h"
#include "chess_nn.h"
#include "chess_puct.h"

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

#define FOOLS_MATE_FEN "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2"
#define LADDER_FEN "7k/8/8/8/8/8/R7/1R5K w - - 0 1"

/* ---- scripted evaluators ---------------------------------------------- */

typedef struct { int calls; } CountCtx;

static void eval_count(void *ctx, int n, const float *planes,
                       const uint8_t *masks, float *logp, float *wdl) {
    (void)planes; (void)masks;
    CountCtx *c = ctx;
    c->calls += n;
    for (int r = 0; r < n; r++) {
        memset(logp + r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + r * 3;
        w[0] = 0; w[1] = 1; w[2] = 0;
    }
}

static void eval_mover_wins(void *ctx, int n, const float *planes,
                            const uint8_t *masks, float *logp, float *wdl) {
    (void)ctx; (void)planes; (void)masks;
    for (int r = 0; r < n; r++) {
        memset(logp + r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + r * 3;
        w[0] = 1; w[1] = 0; w[2] = 0;
    }
}

static void eval_pattern(void *ctx, int n, const float *planes,
                         const uint8_t *masks, float *logp, float *wdl) {
    (void)ctx; (void)planes; (void)masks;
    for (int r = 0; r < n; r++) {
        float *lp = logp + r * PUCT_POLICY_SIZE;
        for (int s = 0; s < PUCT_POLICY_SIZE; s++)
            lp[s] = (s % 7 == 0) ? 1.5f : 0.0f;
        float *w = wdl + r * 3;
        w[0] = 0; w[1] = 1; w[2] = 0;
    }
}

/* ---- helpers ----------------------------------------------------------- */

static bool near_d(double a, double b, double eps) {
    return fabs(a - b) < eps;
}

/* ---- 1. policy slot known answers -------------------------------------- */

static void test_policy_slot(void) {
    State s;
    state_init_startpos(&s);
    CHECK(chess_policy_slot(&s, move_make(E2, E4, F_DOUBLE_PUSH)) == 12 * 64 + 28);
    CHECK(chess_policy_slot(&s, move_make(E1, G1, F_KCASTLE)) == 4 * 64 + 6);

    CHECK(fen_parse(&s, "8/8/8/8/8/8/6p1/K6k b - - 0 1") == 0);
    CHECK(chess_policy_slot(&s, move_make(G2, G1, F_PROMO_Q)) == 54 * 64 + 62);
    CHECK(chess_policy_slot(&s, move_make(G2, G1, F_PROMO_N)) == 4096 + 6 * 9 + 3);

    CHECK(fen_parse(&s, "k7/P7/8/8/8/8/8/K7 w - - 0 1") == 0);
    CHECK(chess_policy_slot(&s, move_make(A7, A8, F_PROMO_Q)) == 48 * 64 + 56);
    CHECK(chess_policy_slot(&s, move_make(A7, A8, F_PROMO_N)) == 4096 + 3);
    CHECK(chess_policy_slot(&s, move_make(A7, B8, F_PROMO_CAP_N)) == 4096 + 6);

    /* enemy-reply region: a "black-shaped" downward promo in white's frame */
    CHECK(fen_parse(&s, "8/8/8/8/8/8/p7/K6k w - - 0 1") == 0);
    CHECK(chess_policy_slot(&s, move_make(A2, A1, F_PROMO_N)) == 4168 + 3);
    CHECK(chess_policy_slot(&s, move_make(A2, B1, F_PROMO_N)) == 4168);

    CHECK(fen_parse(&s, "8/8/8/8/8/8/6p1/K6k b - - 0 1") == 0);
    CHECK(chess_policy_slot(&s, move_make(G2, H1, F_PROMO_CAP_B)) == 4096 + 54 + 7);
}

/* ---- 2 + 3. mate-in-1 pin and bookkeeping ------------------------------ */

static void test_mate_in_1(void) {
    char fen_before[96], fen_after[96];
    PuctGame *g = puct_game_new(FOOLS_MATE_FEN, 0);
    CHECK(g != NULL);
    fen_serialize(puct_state(g), fen_before, sizeof fen_before);

    CountCtx cc = {0};
    puct_set_eval(g, eval_count, &cc);
    puct_think(g, 300);

    fen_serialize(puct_state(g), fen_after, sizeof fen_after);
    CHECK(strcmp(fen_before, fen_after) == 0);   /* rewind left no trace */

    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    float prior[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, prior);
    CHECK(n == 30);   /* black's legal replies in this position */

    double prior_sum = 0, w_children = 0;
    uint32_t n_children_sum = 0, best_visits = 0, best_move = 0;
    for (int i = 0; i < n; i++) {
        prior_sum += prior[i];
        w_children += q[i] * visits[i];
        n_children_sum += visits[i];
        if (visits[i] > best_visits) {
            best_visits = visits[i];
            best_move = moves[i];
        }
        char u[8];
        move_to_uci(moves[i], u);
        if (strcmp(u, "d8h4") == 0)
            CHECK(near_d(q[i], 1.0, 1e-9));   /* proven mate */
    }
    char best_uci[8];
    move_to_uci((Move)best_move, best_uci);
    CHECK(strcmp(best_uci, "d8h4") == 0);
    CHECK(near_d(prior_sum, 1.0, 1e-6));

    /* N(root) = sum N(children); W(root) = -sum W(children):
     * the root-expansion backs nothing up, and every sim then touches the
     * root and exactly one child, with the sign flipping between them. */
    uint32_t root_visits;
    double root_w;
    puct_root_stats(g, &root_visits, &root_w);
    CHECK(root_visits == n_children_sum);
    CHECK(near_d(root_w, -w_children, 1e-9));

    CHECK(cc.calls < 300);   /* terminals were backed up without eval */
    puct_game_free(g);
}

/* ---- 4. value sign convention ------------------------------------------ */

static void test_value_sign(void) {
    PuctGame *g = puct_game_new(NULL, 0);
    puct_set_eval(g, eval_mover_wins, NULL);
    puct_think(g, 2);

    /* Root expansion backs nothing up; each of the 2 sims then expands one
     * new root child (always-first tie-break). The child's own eval says
     * "mover (the root's opponent) wins" = -1 in the root mover's view,
     * while the root collects +1 per sim. */
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, NULL);
    CHECK(n == 20);
    int visited = 0;
    for (int i = 0; i < n; i++) {
        if (visits[i] > 0) {
            visited++;
            CHECK(visits[i] == 1);
            CHECK(near_d(q[i], -1.0, 1e-9));
        }
    }
    CHECK(visited == 2);
    uint32_t root_visits;
    double root_w;
    puct_root_stats(g, &root_visits, &root_w);
    CHECK(root_visits == 2);
    CHECK(near_d(root_w, 2.0, 1e-9));
    puct_game_free(g);
}

/* ---- 5. prior delivery -------------------------------------------------- */

static void test_prior_pattern(void) {
    PuctGame *g = puct_game_new(NULL, 0);
    puct_set_eval(g, eval_pattern, NULL);
    puct_think(g, 1);

    State root;
    state_init_startpos(&root);
    MoveList ml;
    gen_legal(&root, &ml);
    CHECK(ml.count == 20);

    uint16_t moves[MAX_MOVES];
    float prior[MAX_MOVES];
    int n = puct_root_children(g, moves, NULL, NULL, prior);
    CHECK(n == ml.count);

    double z[MAX_MOVES], sum = 0;
    for (int i = 0; i < n; i++) {
        int slot = chess_policy_slot(&root, ml.moves[i]);
        z[i] = (slot % 7 == 0) ? 1.5 : 0.0;
        sum += exp(z[i]);
    }
    for (int i = 0; i < n; i++)
        CHECK(near_d(prior[i], exp(z[i]) / sum, 1e-6));
    puct_game_free(g);
}

/* ---- 6. mask fuzz ------------------------------------------------------- */

static void test_mask_fuzz(void) {
    srand(7);
    PuctGame *g = puct_game_new(NULL, 0);
    static uint8_t mask[PUCT_POLICY_SIZE];
    int positions = 0;
    for (int ply = 0; ply < 200; ply++) {
        const State *s = puct_state(g);
        if (state_termination((State *)s) != TERM_PLAYING)
            break;
        MoveList ml;
        gen_legal(s, &ml);
        CHECK(ml.count > 0);

        puct_request_mask(g, mask);
        bool seen[PUCT_POLICY_SIZE] = {false};
        int marked = 0;
        for (int i = 0; i < PUCT_POLICY_SIZE; i++)
            marked += mask[i];
        CHECK(marked == ml.count);
        for (int i = 0; i < ml.count; i++) {
            int slot = chess_policy_slot(s, ml.moves[i]);
            CHECK(slot >= 0 && slot < PUCT_POLICY_SIZE);
            CHECK(!seen[slot]);
            seen[slot] = true;
            CHECK(mask[slot] == 1);
        }
        positions++;
        puct_play(g, ml.moves[rand() % ml.count]);
    }
    CHECK(positions > 100);
    puct_game_free(g);
}

/* ---- 7. OOM give-up ------------------------------------------------------ */

static void test_oom(void) {
    PuctGame *g = puct_game_new(NULL, sizeof(PuctNode) * 40);
    CHECK(g != NULL);
    puct_think(g, 1000);
    CHECK(puct_oom_count(g) >= 1);
    Move m = puct_best_move(g);
    CHECK(m != 0);
    CHECK(puct_play(g, m) == 0);
    puct_think(g, 100);
    char fen[96];
    fen_serialize(puct_state(g), fen, sizeof fen);
    State chk;
    CHECK(fen_parse(&chk, fen) == 0);   /* state survived, still serializable */
    puct_game_free(g);
}

/* ---- 9. compaction: pack on fill, remap, keep searching ------------------ */

static void test_compaction(void) {
    PuctGame *g = puct_game_new(NULL, sizeof(PuctNode) * 6000);
    CHECK(g != NULL);
    for (int round = 0; round < 12; round++) {
        puct_think(g, 100);
        Move m = puct_best_move(g);
        CHECK(m != 0);
        CHECK(puct_play(g, m) == 0);
    }
    /* the block is far smaller than 12 rounds of search: only packing on
     * fill (dead subtrees of played moves dropped, indices remapped) keeps
     * this going without a single OOM */
    CHECK(puct_compact_count(g) > 0);
    CHECK(puct_oom_count(g) == 0);
    char fen[96];
    fen_serialize(puct_state(g), fen, sizeof fen);
    State chk;
    CHECK(fen_parse(&chk, fen) == 0);
    puct_game_free(g);
}

/* ---- 8. mate-in-2 ladder + subtree keeping ------------------------------- */
static void test_ladder(void) {
    PuctGame *g = puct_game_new(LADDER_FEN, 0);
    puct_think(g, 4000);

    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, NULL);
    CHECK(n > 0);
    uint32_t bv = 0;
    int bi = -1;
    for (int i = 0; i < n; i++)
        if (visits[i] > bv) { bv = visits[i]; bi = i; }
    CHECK(bi >= 0 && near_d(q[bi], 1.0, 0.1));   /* forced mate proven */

    CHECK(puct_play(g, moves[bi]) == 0);

    /* the played child arrives with its subtree already built */
    CHECK(puct_root_children(g, NULL, NULL, NULL, NULL) > 0);

    puct_think(g, 200);   /* black: single forced reply */
    Move black_m = puct_best_move(g);
    CHECK(black_m != 0);
    CHECK(puct_play(g, black_m) == 0);

    puct_think(g, 300);   /* white: mate in 1 is on the board */
    n = puct_root_children(g, moves, visits, q, NULL);
    bv = 0; bi = -1;
    for (int i = 0; i < n; i++)
        if (visits[i] > bv) { bv = visits[i]; bi = i; }
    CHECK(bi >= 0 && near_d(q[bi], 1.0, 0.1));
    CHECK(puct_play(g, moves[bi]) == 0);
    CHECK(state_termination((State *)puct_state(g)) == TERM_CHECKMATE);
    puct_game_free(g);
}

/* ---- bench ---------------------------------------------------------------- */

static void bench(void) {
    PuctGame *g = puct_game_new(NULL, 0);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    puct_think(g, 30000);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("bench: 30000 sims in %.3fs = %.0f sims/s (oom=%u)\n",
           dt, 30000.0 / dt, puct_oom_count(g));
    puct_game_free(g);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    chess_init();
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        bench();
        return 0;
    }
    const char *only = getenv("PUCT_ONLY");
    if (!only || strcmp(only, "slot") == 0) test_policy_slot();
    if (!only || strcmp(only, "mate1") == 0) test_mate_in_1();
    if (!only || strcmp(only, "sign") == 0) test_value_sign();
    if (!only || strcmp(only, "prior") == 0) test_prior_pattern();
    if (!only || strcmp(only, "fuzz") == 0) test_mask_fuzz();
    if (!only || strcmp(only, "compact") == 0) test_compaction();
    if (!only || strcmp(only, "oom") == 0) test_oom();
    if (!only || strcmp(only, "ladder") == 0) test_ladder();
    printf("test_puct: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
