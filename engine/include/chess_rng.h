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

/* Standard normal (Box–Muller; the second output of each pair is
 * discarded to keep the stream single-valued). */
double   rng_normal(void);

/* Gamma(shape, 1): Marsaglia–Tsang for shape >= 1; below 1 via the
 * boost G(x) = G(x+1)·U^(1/x) (same stream either way). */
double   rng_gamma(double shape);

/* n i.i.d. Gamma(alpha, 1) draws normalized to sum 1 — the Dirichlet
 * distribution behind AGZ root-noise exploration. */
void     rng_dirichlet(double *out, int n, double alpha);
double   rng_uniform_s(uint64_t *st);
void     rng_dirichlet_s(uint64_t *st, double *out, int n, double alpha);

#endif /* CHESS_RNG_H */
