/* match_util.c — see chess_match.h. */
#define _POSIX_C_SOURCE 200809L
#include <time.h>

#include "chess_match.h"
#include "chess_rng.h"

double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

Move match_pick_move(const uint16_t *moves, const uint32_t *visits,
                     const float *prior, int n, int ply, int sample_plys,
                     bool prior_side, double temp, uint32_t boost) {
    if (n <= 0)
        return 0;
    double w[MAX_MOVES];
    if (prior_side) {
        for (int i = 0; i < n; i++)
            w[i] = prior[i];
    } else {
        for (int i = 0; i < n; i++)
            w[i] = (double)visits[i] + boost;
    }
    if (ply < sample_plys)
        return moves[sample_index(w, n, temp)];
    int best = 0;
    for (int i = 1; i < n; i++)
        if (w[i] > w[best])
            best = i;
    return moves[best];
}
