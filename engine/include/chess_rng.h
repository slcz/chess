/*
 * chess_rng.h — the search RNG (Phase 3.1): the C mirror of
 * chess_ai/selfplay/rng.py.
 *
 * WHY THIS MODULE EXISTS (PARALLEL_DESIGN.md R6)
 * ==============================================
 * The C engine and the Python reference (chess_ai/selfplay/gumbel.py) must
 * consume *identical random streams* from identical seeds — that is what
 * makes the L2 exact-diff gate (C sequential vs Python, same seed ⇒
 * identical games) possible at all. So the generator, the seeding and the
 * uniform mapping are pinned here line for line against rng.py, and the
 * tests assert stream parity over the process boundary
 * (tests/test_search_l2.py drives BOTH implementations in one process):
 *
 *   - splitmix64 for seed derivation (Vigna's reference constants),
 *   - xoshiro256++ as the per-game generator, seeded via four splitmix64
 *     draws (the all-zero state is guarded away like rng.py does),
 *   - uniform() maps to (0, 1) exclusive at 53-bit precision via a midpoint
 *     offset — never 0.0, never 1.0, so -log(-log(u)) (the Gumbel draw) is
 *     always finite, and an IEEE-double implementation reproduces it
 *     bit-exactly,
 *   - gumbel() = inverse-CDF Gumbel(0, 1).
 *
 * Consumption order (enforced by the search in search.c, mirrored in
 * gumbel.py): per move, one gumbel() per legal action in movegen
 * enumeration order; per sampled move one uniform() resolved by inverse CDF
 * over the candidates in enumeration order; nothing else draws from the
 * stream. Per-game instance — never shared across threads (PARALLEL_DESIGN:
 * a game lives on one thread for its whole life).
 */
#ifndef CHESS_RNG_H
#define CHESS_RNG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* xoshiro256++ state: four 64-bit words, seeded from splitmix64. Kept in
 * the header so foreign runtimes (Python tests, ABI cffi) can drive the
 * generator directly for stream-parity checks. */
typedef struct {
    uint64_t s[4];
} Rng256;

/* One splitmix64 step on `state` (advanced in place); returns the output.
 * Pure arithmetic — the exact reference constants from rng.py. */
uint64_t splitmix64_next(uint64_t *state);

/* Per-game seed derivation (rng.py game_seed): one splitmix64 draw from the
 * work-unit seed, XOR the game index, one more draw. */
uint64_t rng_game_seed(uint64_t unit_seed, uint64_t game_index);

/* Seed the generator: four splitmix64 draws from `seed`; the all-zero state
 * (the generator's single fixed point, unreachable from splitmix64 seeding)
 * is guarded to s[0] = 1 exactly like rng.py. */
void rng_init(Rng256 *r, uint64_t seed);

/* Raw next 64-bit output (xoshiro256++). */
uint64_t rng_next_u64(Rng256 *r);

/* Uniform in (0, 1), 53-bit precision, endpoints excluded:
 *   ((x >> 11) + 0.5) * 2^-53
 * The 53 high bits land on the *midpoints* of 2^53 equal bins. The
 * arithmetic order matches rng.py so an IEEE-754 double run is bit-exact. */
double rng_uniform(Rng256 *r);

/* Standard Gumbel(0, 1) via inverse CDF: -log(-log(u)). */
double rng_gumbel(Rng256 *r);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_RNG_H */
