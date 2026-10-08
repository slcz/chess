#define _POSIX_C_SOURCE 200809L
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
 *   9. mux parity: 3 games on a shared landing zone reach the exact root
 *      visit vector of a single-game think (deterministic single thread)
 *   10. zone pressure: batch cap below walkers×games splits rounds
 *      mid-game; bookkeeping invariants hold (parks == root visits)
 *   11. recycling + quotas: games play to a ply cap, are reset in place,
 *      and the run stops at the preset game count
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

static void eval_count(void *ctx, PuctEvalOp op, int bank, int n,
                       const float *planes,
                       const uint8_t *masks, float *logp, float *wdl) {
    (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;                     /* PULL/WAIT are no-ops for sync evals */
    CountCtx *c = ctx;
    c->calls += n;
    for (int r = 0; r < n; r++) {
        memset(logp + r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + r * 3;
        w[0] = 0; w[1] = 1; w[2] = 0;
    }
}

static void eval_mover_wins(void *ctx, PuctEvalOp op, int bank, int n,
                            const float *planes,
                            const uint8_t *masks, float *logp, float *wdl) {
    (void)ctx; (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;
    for (int r = 0; r < n; r++) {
        memset(logp + r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + r * 3;
        w[0] = 1; w[1] = 0; w[2] = 0;
    }
}

static void eval_pattern(void *ctx, PuctEvalOp op, int bank, int n,
                         const float *planes,
                         const uint8_t *masks, float *logp, float *wdl) {
    (void)ctx; (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;
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
    puct_set_eval(eval_count, &cc);
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
    puct_set_walkers(g, 1);
    puct_set_eval(eval_mover_wins, NULL);
    puct_think(g, 1);

    /* one sim: the walker first expands the root (its row fills priors
     * only, no sim), then descends into the first root child — uniform
     * placeholder priors, no visits yet — expands it, and its row backs
     * "mover at the leaf wins" up: Q = -1 in the root mover's view, and
     * the root collects the opposite sign. */
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, NULL);
    CHECK(n == 20);
    int hot = 0;
    for (int i = 0; i < n; i++) {
        if (visits[i] > 0) {
            hot++;
            CHECK(visits[i] == 1);
            CHECK(near_d(q[i], -1.0, 1e-9));
        }
    }
    CHECK(hot == 1);
    uint32_t root_visits;
    double root_w;
    puct_root_stats(g, &root_visits, &root_w);
    CHECK(root_visits == 1);
    CHECK(near_d(root_w, 1.0, 1e-9));
    puct_game_free(g);
}

/* ---- 4b. single walker degenerates to sequential search ----------------- */

static void test_single_walker(void) {
    PuctGame *g = puct_game_new(FOOLS_MATE_FEN, 0);
    puct_set_walkers(g, 1);
    puct_think(g, 200);
    char best_uci[8];
    move_to_uci(puct_best_move(g), best_uci);
    CHECK(strcmp(best_uci, "d8h4") == 0);
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    double q[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, q, NULL);
    for (int i = 0; i < n; i++) {
        char u[8];
        move_to_uci(moves[i], u);
        if (strcmp(u, "d8h4") == 0)
            CHECK(near_d(q[i], 1.0, 1e-9));
    }
    puct_game_free(g);
}

/* ---- 4c. determinism: same inputs, same visit vector -------------------- */

static void test_determinism(void) {
    PuctGame *g1 = puct_game_new(NULL, 0);
    PuctGame *g2 = puct_game_new(NULL, 0);
    puct_think(g1, 400);
    puct_think(g2, 400);
    uint16_t m1[MAX_MOVES], m2[MAX_MOVES];
    uint32_t v1[MAX_MOVES], v2[MAX_MOVES];
    int n1 = puct_root_children(g1, m1, v1, NULL, NULL);
    int n2 = puct_root_children(g2, m2, v2, NULL, NULL);
    CHECK(n1 == n2 && n1 > 0);
    for (int i = 0; i < n1 && i < n2; i++) {
        CHECK(m1[i] == m2[i]);
        CHECK(v1[i] == v2[i]);
    }
    puct_game_free(g1);
    puct_game_free(g2);
}

/* ---- 5. prior delivery -------------------------------------------------- */

static void test_prior_pattern(void) {
    PuctGame *g = puct_game_new(NULL, 0);
    puct_set_eval(eval_pattern, NULL);
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

/* ---- 10. mux parity: shared zone == single-game think -------------------- */

static void test_mux_parity(void) {
    PuctGame *ref = puct_game_new(NULL, 0);
    puct_think(ref, 60);
    uint16_t rm[MAX_MOVES];
    uint32_t rv[MAX_MOVES];
    int rn = puct_root_children(ref, rm, rv, NULL, NULL);
    CHECK(rn == 20);

    /* 3 games, batch cap 8×3 = 24: no mid-game zone splits, so each game
     * must reproduce the reference think bit-for-bit */
    PuctMux *m = puct_mux_new(3, 24, 0);
    int done_games = 0;
    for (int step = 0; step < 100 && done_games < 3; step++) {
        puct_mux_step(m, 60);
        done_games = 0;
        for (int i = 0; i < puct_mux_games(m); i++) {
            PuctGame *g = puct_mux_game(m, i);
            if (!puct_think_done(g))
                continue;
            done_games++;
            uint16_t mv[MAX_MOVES];
            uint32_t vi[MAX_MOVES];
            int n = puct_root_children(g, mv, vi, NULL, NULL);
            CHECK(n == rn);
            for (int k = 0; k < n; k++) {
                CHECK(mv[k] == rm[k]);
                CHECK(vi[k] == rv[k]);
            }
        }
    }
    CHECK(done_games == 3);
    puct_mux_free(m);
    puct_game_free(ref);
}

/* ---- 11. zone pressure: batch cap below walkers×games --------------------- */

static void test_mux_pressure(void) {
    PuctMux *m = puct_mux_new(3, 5, 0);   /* cap 5 < 8 walkers: rounds split */
    uint32_t inherited[3] = {0};
    bool expect[3] = {false, false, false};
    int checks = 0;
    for (int step = 0; step < 500 && checks < 20; step++) {
        puct_mux_step(m, 37);
        for (int i = 0; i < puct_mux_games(m); i++) {
            PuctGame *g = puct_mux_game(m, i);
            if (!puct_think_done(g))
                continue;
            uint32_t parks, terms, root_visits;
            double root_w;
            puct_park_stats(g, &parks, &terms, NULL);
            puct_root_stats(g, &root_visits, &root_w);
            if (puct_root_children(g, NULL, NULL, NULL, NULL) == 0) {
                expect[i] = false;           /* terminal: game over */
                puct_game_reset(g);
                continue;
            }
            /* the played child becomes the root with its visits intact;
             * every park of the next think then backs the root up exactly
             * once, so root visits == inherited + parks */
            if (expect[i]) {
                CHECK(root_visits == parks + inherited[i]);
                checks++;
            }
            uint16_t mv[MAX_MOVES];
            uint32_t vi[MAX_MOVES];
            int n = puct_root_children(g, mv, vi, NULL, NULL);
            uint32_t bv = 0;
            Move best = 0;
            for (int k = 0; k < n; k++)
                if (vi[k] > bv) { bv = vi[k]; best = mv[k]; }
            inherited[i] = bv;
            expect[i] = true;
            CHECK(best != 0);
            CHECK(puct_play(g, best) == 0);
        }
    }
    CHECK(checks >= 20);
    CHECK(puct_mux_rows(m) > 0);
    puct_mux_free(m);
}

/* ---- 12. recycling + run quota -------------------------------------------- */

static void test_mux_recycle(void) {
    PuctMux *m = puct_mux_new(2, 16, 0);
    int finished = 0;
    /* run until 4 games complete (each pool game recycled once): the
     * driver plays a move per completed think and adjudicates at 12 ply */
    for (int step = 0; step < 4000 && finished < 4; step++) {
        puct_mux_step(m, 24);
        for (int i = 0; i < puct_mux_games(m); i++) {
            PuctGame *g = puct_mux_game(m, i);
            if (!puct_think_done(g))
                continue;
            if (puct_state(g)->ply >= 12 ||
                state_termination((State *)puct_state(g)) != TERM_PLAYING) {
                finished++;
                CHECK(puct_state(g)->ply >= 1);
                puct_game_reset(g);
                CHECK(puct_state(g)->ply == 0);   /* fresh startpos game */
                continue;
            }
            Move mv = puct_best_move(g);
            CHECK(mv != 0);
            CHECK(puct_play(g, mv) == 0);
        }
    }
    CHECK(finished == 4);

    /* rows quota: drive until 400 more eval rows accumulate (the games
     * quota is out of the picture — recycling keeps the pool running) */
    uint64_t target = puct_mux_rows(m) + 400;
    for (int step = 0; puct_mux_rows(m) < target; step++) {
        CHECK(step < 20000);
        puct_mux_step(m, 24);
        for (int i = 0; i < puct_mux_games(m); i++) {
            PuctGame *g = puct_mux_game(m, i);
            if (!puct_think_done(g))
                continue;
            if (puct_state(g)->ply >= 12 ||
                state_termination((State *)puct_state(g)) != TERM_PLAYING) {
                puct_game_reset(g);
                continue;
            }
            Move mv = puct_best_move(g);
            CHECK(mv != 0);
            CHECK(puct_play(g, mv) == 0);
        }
    }
    CHECK(puct_mux_rows(m) >= target);
    puct_mux_free(m);
}

/* ---- bench ---------------------------------------------------------------- */

/* Phase 3b measurement: pure-CPU + dummy-evaluator throughput of the
 * double-buffered mux — the ceiling the real net must fit under. */
static void bench_mux(int games_max, int rows_max, int sims) {
    PuctMux *m = puct_mux_new(games_max, rows_max, 0);
    uint64_t row_quota = 200000;
    int finished = 0;
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    while (puct_mux_rows(m) < row_quota) {
        puct_mux_step(m, sims);
        for (int i = 0; i < puct_mux_games(m); i++) {
            PuctGame *g = puct_mux_game(m, i);
            if (!puct_think_done(g))
                continue;
            if (puct_state(g)->ply >= 60 ||
                state_termination((State *)puct_state(g)) != TERM_PLAYING) {
                finished++;
                puct_game_reset(g);
                continue;
            }
            Move mv = puct_best_move(g);
            if (mv)
                puct_play(g, mv);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("bench_mux: %llu rows in %.2fs = %.0f rows/s, %d games finished "
           "(games=%d rows/bank=%d sims=%d)\n",
           (unsigned long long)puct_mux_rows(m), dt,
           (double)puct_mux_rows(m) / dt, finished,
           games_max, rows_max, sims);
    puct_mux_free(m);
}

static void bench(int walkers) {
    PuctGame *g = puct_game_new(NULL, 0);
    puct_set_walkers(g, walkers);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    puct_think(g, 30000);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("bench: 30000 sims in %.3fs = %.0f sims/s (walkers=%d oom=%u)\n",
           dt, 30000.0 / dt, walkers, puct_oom_count(g));
    puct_game_free(g);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    chess_init();
    if (argc > 1 && strcmp(argv[1], "bench") == 0) {
        int ws[4] = {1, 2, 4, 8};
        for (int i = 0; i < 4; i++)
            bench(ws[i]);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "benchmux") == 0) {
        bench_mux(64, 64, 200);
        bench_mux(256, 256, 200);
        return 0;
    }
    const char *only = getenv("PUCT_ONLY");
    if (!only || strcmp(only, "slot") == 0) test_policy_slot();
    if (!only || strcmp(only, "mate1") == 0) test_mate_in_1();
    if (!only || strcmp(only, "sign") == 0) test_value_sign();
    if (!only || strcmp(only, "single") == 0) test_single_walker();
    if (!only || strcmp(only, "det") == 0) test_determinism();
    if (!only || strcmp(only, "prior") == 0) test_prior_pattern();
    if (!only || strcmp(only, "fuzz") == 0) test_mask_fuzz();
    if (!only || strcmp(only, "compact") == 0) test_compaction();
    if (!only || strcmp(only, "oom") == 0) test_oom();
    if (!only || strcmp(only, "ladder") == 0) test_ladder();
    if (!only || strcmp(only, "mux_parity") == 0) test_mux_parity();
    if (!only || strcmp(only, "mux_pressure") == 0) test_mux_pressure();
    if (!only || strcmp(only, "mux_recycle") == 0) test_mux_recycle();
    printf("test_puct: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
