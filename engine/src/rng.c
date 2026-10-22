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

double rng_normal(void) {
    static const double two_pi = 6.283185307179586476925286766559;
    double u1 = rng_uniform();
    if (u1 < 1e-300)
        u1 = 1e-300;                      /* log(0) guard */
    return sqrt(-2.0 * log(u1)) * cos(two_pi * rng_uniform());
}

double rng_gamma(double shape) {
    if (shape < 1.0) {                    /* boost: G(x) = G(x+1)·U^(1/x) */
        return rng_gamma(shape + 1.0) * pow(rng_uniform(), 1.0 / shape);
    }
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {                            /* Marsaglia–Tsang squeeze */
        double x = rng_normal();
        double v = 1.0 + c * x;
        if (v <= 0.0)
            continue;
        v = v * v * v;
        double u = rng_uniform();
        if (u < 1.0 - 0.0331 * x * x * x * x)
            return d * v;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
            return d * v;
    }
}

/* Per-stream variants: the search keeps one stream per game so a transcript
 * does not depend on which thread walked it or in what order. */
double rng_uniform_s(uint64_t *st) {
    *st += 0x9E3779B97F4A7C15ull;
    uint64_t z = *st;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (double)(z >> 11) / 9007199254740992.0;
}

static double rng_normal_s(uint64_t *st) {
    double u1 = rng_uniform_s(st);
    if (u1 < 1e-300)
        u1 = 1e-300;
    return sqrt(-2.0 * log(u1)) *
           cos(6.283185307179586476925286766559 * rng_uniform_s(st));
}

static double rng_gamma_s(uint64_t *st, double shape) {
    if (shape < 1.0)
        return rng_gamma_s(st, shape + 1.0) * pow(rng_uniform_s(st), 1.0 / shape);
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x = rng_normal_s(st);
        double v = 1.0 + c * x;
        if (v <= 0.0)
            continue;
        v = v * v * v;
        double u = rng_uniform_s(st);
        if (u < 1.0 - 0.0331 * x * x * x * x)
            return d * v;
        if (log(u) < 0.5 * x * x + d * (1.0 - v + log(v)))
            return d * v;
    }
}

void rng_dirichlet_s(uint64_t *st, double *out, int n, double alpha) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = rng_gamma_s(st, alpha);
        sum += out[i];
    }
    if (sum <= 0.0) {
        for (int i = 0; i < n; i++)
            out[i] = 1.0 / n;
        return;
    }
    for (int i = 0; i < n; i++)
        out[i] /= sum;
}

void rng_dirichlet(double *out, int n, double alpha) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = rng_gamma(alpha);
        sum += out[i];
    }
    if (sum <= 0.0) {                     /* all-underflow: degenerate to
                                             uniform rather than NaN */
        for (int i = 0; i < n; i++)
            out[i] = 1.0 / n;
        return;
    }
    for (int i = 0; i < n; i++)
        out[i] /= sum;
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
