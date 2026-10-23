/* rng.c — the one random stream behind the match drivers (chess_rng.h).
 * The global state is one atomic Weyl step (splitmix64 is state += K
 * then mix — fetch_add reproduces it exactly), so the mux's walk tasks
 * can each grab a per-game seed at think start with no lock; all hot
 * sampling runs lock-free on the caller-owned copy (the *_stream
 * pair). */
#include <math.h>
#include <stdatomic.h>

#include "chess_rng.h"

static _Atomic uint64_t state;

static uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4B5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint64_t rng_next_stream(uint64_t *st) {
    *st += 0x9E3779B97F4A7C15ull;
    return mix(*st);
}

void rng_seed(uint64_t seed) {
    atomic_store(&state, seed ? seed : 1);
}

uint64_t rng_next(void) {
    uint64_t z = atomic_fetch_add(&state, 0x9E3779B97F4A7C15ull);
    return mix(z + 0x9E3779B97F4A7C15ull);
}

double rng_uniform(void) {
    return (double)(rng_next() >> 11) / 9007199254740992.0;
}

int sample_index_stream(uint64_t *st, const double *w, int n, double temp) {
    if (temp <= 0.0) {                    /* -> 0: deterministic argmax */
        int best = 0;
        for (int i = 1; i < n; i++)
            if (w[i] > w[best])
                best = i;
        return best;
    }
    if (temp == 1.0) {                    /* the original stream, exactly */
        double total = 0.0;
        for (int i = 0; i < n; i++)
            total += w[i];
        if (total <= 0.0)
            return n - 1;
        double r = (double)(rng_next_stream(st) >> 11) / 9007199254740992.0
                   * total;
        for (int i = 0; i < n; i++) {
            r -= w[i];
            if (r < 0.0)
                return i;
        }
        return n - 1;
    }
    /* general T: sample ∝ w^(1/T), log-shift by the max for stability */
    double lw[n];
    double lm = -INFINITY;
    for (int i = 0; i < n; i++) {
        lw[i] = w[i] > 0.0 ? log(w[i]) / temp : -INFINITY;
        if (lw[i] > lm)
            lm = lw[i];
    }
    if (lm == -INFINITY)
        return n - 1;                     /* all-zero: the old answer */
    double total = 0.0;
    for (int i = 0; i < n; i++)
        total += lw[i] == -INFINITY ? 0.0 : exp(lw[i] - lm);
    double r = (double)(rng_next_stream(st) >> 11) / 9007199254740992.0
               * total;
    for (int i = 0; i < n; i++) {
        r -= lw[i] == -INFINITY ? 0.0 : exp(lw[i] - lm);
        if (r < 0.0)
            return i;
    }
    return n - 1;
}

int sample_index(const double *w, int n, double temp) {
    /* one atomic draw seeds a throwaway stream: driver-side sampling,
     * rare and never contended by the walk tasks */
    uint64_t st = temp > 0.0 ? rng_next() : 0;
    return sample_index_stream(&st, w, n, temp);
}
