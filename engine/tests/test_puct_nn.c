/* test_puct_nn.c — real-net smoke (plan gate 7): the 42k seed checkpoint
 * through the walker/IO pipeline. Not part of `make check` (needs torch +
 * a checkpoint); run manually:
 *
 *   PUCT_CKPT=runs/policy_value_net_42k.pt build/test_puct_nn
 *
 * Verifies: the evaluator attaches; a low-sim think picks a legal, sane
 * move from the start position; a demo game against a greedy-material
 * opponent completes legally (exercising puct_play's out-of-tree path).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "chess.h"
#include "chess_nn_eval.h"

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

static bool is_legal(const State *s, Move m) {
    MoveList ml;
    gen_legal(s, &ml);
    for (int i = 0; i < ml.count; i++)
        if (ml.moves[i] == m)
            return true;
    return false;
}

static int material_gain(const State *s, Move m) {
    static const int val[6] = {1, 3, 3, 5, 9, 0};
    int d = 0;
    if (move_is_capture(m))
        d += val[pc_type(s->board[move_to(m)])];
    if (move_is_promo(m))
        d += val[move_promo_type(m)] - 1;
    return d;
}

static void print_top(const PuctGame *g, int k) {
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, NULL, NULL);
    for (int i = 0; i < k && i < n; i++) {
        char u[8];
        move_to_uci(moves[i], u);
        printf("  %s visits=%u\n", u, visits[i]);
    }
}

int main(void) {
    const char *ckpt = getenv("PUCT_CKPT");
    if (!ckpt || !*ckpt) {
        printf("PUCT_CKPT not set: NN smoke skipped\n");
        return 0;
    }
    const char *device = getenv("PUCT_DEVICE");
    if (!device || !*device)
        device = "mps";

    chess_init();
    PuctGame *g = puct_game_new(NULL, 0);
    CHECK(puct_attach_nn(ckpt, device, PUCT_MAX_WALKERS) == 0);

    /* smoke 1: low-sim think from the start position picks a legal move */
    puct_think(g, 60);
    Move best = puct_best_move(g);
    CHECK(best != 0);
    CHECK(is_legal(puct_state(g), best));
    char best_uci[8];
    move_to_uci(best, best_uci);
    printf("startpos think(60): %s\n", best_uci);
    print_top(g, 3);
    puct_play(g, best);

    /* smoke 2: demo game vs greedy material, 30 plies or game end */
    for (int ply = 0; ply < 30; ply++) {
        if (state_termination((State *)puct_state(g)) != TERM_PLAYING)
            break;
        puct_think(g, 40);
        Move m = puct_best_move(g);
        CHECK(m != 0 && is_legal(puct_state(g), m));
        puct_play(g, m);

        if (state_termination((State *)puct_state(g)) != TERM_PLAYING)
            break;
        MoveList ml;                       /* greedy-material opponent */
        gen_legal(puct_state(g), &ml);
        Move opp = ml.moves[0];
        int bs = material_gain(puct_state(g), opp);
        for (int i = 1; i < ml.count; i++) {
            int gain = material_gain(puct_state(g), ml.moves[i]);
            if (gain > bs) {
                bs = gain;
                opp = ml.moves[i];
            }
        }
        CHECK(puct_play(g, opp) == 0);     /* out-of-tree move: fresh root */
    }
    char fen[96];
    fen_serialize(puct_state(g), fen, sizeof fen);
    State chk;
    CHECK(fen_parse(&chk, fen) == 0);
    printf("demo game after %d plies: %s\n",
           (int)puct_state(g)->ply, fen);

    /* throughput: real-net evals through the walker/IO pipeline */
    puct_think(g, 500);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    puct_think(g, 2000);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double dt = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
    printf("nn bench: 2000 sims in %.3fs = %.0f sims/s\n", dt, 2000.0 / dt);

    puct_game_free(g);
    printf("test_puct_nn: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
