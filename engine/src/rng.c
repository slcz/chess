/* rng.c — the one random stream behind the match drivers (chess_rng.h). */
#include <math.h>

#include "chess_rng.h"

static uint64_t state;

void rng_seed(uint64_t seed) {
    state = seed ? seed : 1;
}

uint64_t rng_next(void) {                 /* splitmix64 */
    state += 0x9E3779B97F4A7C15ull;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

double rng_uniform(void) {
    return (double)(rng_next() >> 11) / 9007199254740992.0;
}

int sample_index(const double *w, int n, double temp) {
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
        double r = rng_uniform() * total;
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
    double r = rng_uniform() * total;
    for (int i = 0; i < n; i++) {
        r -= lw[i] == -INFINITY ? 0.0 : exp(lw[i] - lm);
        if (r < 0.0)
            return i;
    }
    return n - 1;
}
