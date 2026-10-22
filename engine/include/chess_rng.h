/*
 * chess_rng.h — splitmix64 random stream + weighted index sampling,
 * shared by the match drivers (opening sampling, visit/prior-weighted
 * move choice). One global stream per process; identical seeds reproduce
 * identical streams (the regression goldens depend on it).
 */
#ifndef CHESS_RNG_H
#define CHESS_RNG_H

#include <stdint.h>

void     rng_seed(uint64_t seed);   /* seed 0 becomes 1 */
uint64_t rng_next(void);            /* raw splitmix64 output */
double   rng_uniform(void);         /* [0,1), 53-bit resolution */

/* Index i sampled ∝ w[i]^(1/temp): temp = 1 is the plain proportional
 * sample (bit-identical stream); temp <= 0 is the deterministic argmax
 * (first max on ties); temp > 1 flattens toward uniform. */
int      sample_index(const double *w, int n, double temp);

#endif /* CHESS_RNG_H */
