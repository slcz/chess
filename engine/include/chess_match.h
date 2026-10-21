/*
 * chess_match.h — small utilities shared by the match drivers
 * (selfplay, sf_gauntlet): a monotonic clock for elapsed
 * reporting, and the common move-choice policy (weights = priors for a
 * prior-side player, else visit counts; sample early, argmax late).
 */
#ifndef CHESS_MATCH_H
#define CHESS_MATCH_H

#include "chess.h"

/* Seconds on the monotonic clock — elapsed-time reporting only. */
double now_s(void);

/* The match drivers' move choice: the first sample_plys plies sample
 * ∝ w^(1/temp) (sample_index), later plies argmax. The weights are the
 * policy priors for a prior-side player, else the visit counts plus
 * boost (sf_gauntlet uses 1 so unvisited moves stay reachable). 0 if
 * n <= 0. */
Move match_pick_move(const uint16_t *moves, const uint32_t *visits,
                     const float *prior, int n, int ply, int sample_plys,
                     bool prior_side, double temp, uint32_t boost);

#endif /* CHESS_MATCH_H */
