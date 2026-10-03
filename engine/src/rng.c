/*
 * rng.c — implementation of chess_rng.h: the exact C mirror of
 * chess_ai/selfplay/rng.py (see the header for why bit-exactness matters).
 *
 * Every arithmetic step below is written in the same order as its Python
 * counterpart so IEEE-754 doubles reproduce the Python stream bit-for-bit:
 * integer ops are exact by construction; the two floating points
 * (uniform's midpoint mapping, gumbel's inverse CDF) use the same op order
 * and the same libm exp/log the Python interpreter calls.
 */
#include "chess_rng.h"

#include <math.h>

#define RNG_GOLDEN 0x9E3779B97F4A7C15ULL /* splitmix64 additive constant */
#define RNG_M1     0xBF58476D1CE4E5B9ULL
#define RNG_M2     0x94D049BB133111EBULL

uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z;
    *state += RNG_GOLDEN;
    z = *state;
    z = (z ^ (z >> 30)) * RNG_M1;
    z = (z ^ (z >> 27)) * RNG_M2;
    return z ^ (z >> 31);
}

uint64_t rng_game_seed(uint64_t unit_seed, uint64_t game_index) {
    uint64_t a = splitmix64_next(&unit_seed);
    uint64_t b = a ^ game_index;
    return splitmix64_next(&b);
}

static uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

void rng_init(Rng256 *r, uint64_t seed) {
    int i, nonzero = 0;
    for (i = 0; i < 4; i++)
        r->s[i] = splitmix64_next(&seed);
    for (i = 0; i < 4; i++)
        nonzero |= (r->s[i] != 0);
    if (!nonzero)
        r->s[0] = 1;
}

uint64_t rng_next_u64(Rng256 *r) {
    uint64_t result, t;
    result = rotl64(r->s[0] + r->s[3], 23) + r->s[0];
    t = r->s[1] << 17;
    r->s[2] ^= r->s[0];
    r->s[3] ^= r->s[1];
    r->s[1] ^= r->s[2];
    r->s[0] ^= r->s[3];
    r->s[2] ^= t;
    r->s[3] = rotl64(r->s[3], 45);
    return result;
}

double rng_uniform(Rng256 *r) {
    /* ((x >> 11) + 0.5) * 2^-53 — op order mirrors rng.py: the integer is
     * widened exactly (it is < 2^53), +0.5, then one multiply by the exact
     * power of two 0x1.0p-53. */
    return ((double)(rng_next_u64(r) >> 11) + 0.5) * 0x1.0p-53;
}

double rng_gumbel(Rng256 *r) {
    double u = rng_uniform(r);
    return -log(-log(u));
}
