/*
 * puct_probe.c — search-convergence probe: for each FEN on stdin, run a
 * FRESH search (new tree, no reuse) at several sim levels and report the
 * chosen move at each level.
 *
 *   puct_probe <ckpt> <levels> [device=mps]     # levels: "0,32,64,..."
 *   stdin:  one FEN per line (blank/# lines skipped)
 *   stdout: MOVE <pos> <n> <uci> <visits> <value>
 *           one row per (position, level); value = root value in the
 *           mover's view (-(W/N) of the root, whose W is parent-view).
 *
 * Level 0 is the raw-policy player: think(1) evaluates the root once and
 * the move is argmax of the policy priors (visit counts ignored). For
 * n >= 1 the move is argmax visits, ties broken by prior.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "chess_nn_eval.h"

static Move choose(const PuctGame *g, int level) {
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    float prior[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, NULL, prior);
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (level == 0) {
            if (prior[i] > prior[best])
                best = i;
        } else if (visits[i] > visits[best] ||
                   (visits[i] == visits[best] && prior[i] > prior[best])) {
            best = i;
        }
    }
    return n ? moves[best] : 0;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <ckpt> <levels> [device=mps]\n", argv[0]);
        return 2;
    }
    const char *ckpt = argv[1];
    const char *device = argc > 3 ? argv[3] : "mps";

    int levels[128], n_levels = 0;
    char *lvl = strdup(argv[2]);
    for (char *t = strtok(lvl, ","); t && n_levels < 128;
         t = strtok(NULL, ","))
        levels[n_levels++] = atoi(t);

    chess_init();
    PuctGame *g = puct_game_new(NULL, 0);
    if (puct_attach_nn(ckpt, device, PUCT_MAX_WALKERS) != 0) {
        fprintf(stderr, "puct_attach_nn failed\n");
        return 1;
    }
    puct_game_free(g);                 /* model is cached; game was a probe */

    char line[128];
    int pos = 0;
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = '\0';
        if (!line[0] || line[0] == '#')
            continue;
        for (int li = 0; li < n_levels; li++) {
            int n = levels[li];
            g = puct_game_new(line, 0);
            if (!g) {
                fprintf(stderr, "bad fen: %s\n", line);
                return 1;
            }
            puct_think(g, n > 0 ? n : 1);
            Move m = choose(g, n);
            char u[8] = "0000";
            if (m)
                move_to_uci(m, u);
            uint32_t visits;
            double wsum;
            puct_root_stats(g, &visits, &wsum);
            /* root W is the parent-mover's view; flip for the mover */
            printf("MOVE %d %d %s %u %.4f\n", pos, n, u, visits,
                   visits ? -wsum / (double)visits : 0.0);
            puct_game_free(g);
        }
        fflush(stdout);
        pos++;
    }
    free(lvl);
    return 0;
}
