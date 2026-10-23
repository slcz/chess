/* sp_dump.c — SPSHARD1 writer (chess_sp_dump.h). */
#define _POSIX_C_SOURCE 200809L
#include "chess_sp_dump.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

void sp_dump_shard_path(const char *dir, int worker, char *buf, size_t cap) {
    snprintf(buf, cap, "%s/shard-%d.spbin", dir, worker);
}

/* All writes go through this: a short write on a shard is fatal for the
 * run (the header contract), so it is checked, not retried. */
static int write_all(int fd, const void *p, size_t n) {
    const char *c = p;
    while (n > 0) {
        ssize_t w = write(fd, c, n);
        if (w < 0) {
            if (errno == EINTR)
                continue;
            return -1;
        }
        c += w;
        n -= (size_t)w;
    }
    return 0;
}

int sp_dump_open_shard(const char *dir, int worker, uint64_t unit_seed) {
    mkdir(dir, 0777);   /* ok if it exists */
    char path[512];
    sp_dump_shard_path(dir, worker, path, sizeof path);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return -1;
    char hdr[8] = {'S', 'P', 'S', 'H', 'A', 'R', 'D', '1'};
    struct { uint32_t version, worker; uint64_t seed; } h = {SP_VERSION,
                                                             (uint32_t)worker,
                                                             unit_seed};
    if (write_all(fd, hdr, 8) ||
        write_all(fd, &h, sizeof h)) {
        close(fd);
        return -1;
    }
    return fd;
}

int sp_dump_game_head(int fd, const SpGameRec *g) {
    char fen[SP_FEN_MAX] = {0};
    strncpy(fen, g->start_fen, SP_FEN_MAX - 1);
    if (write_all(fd, &g->game_index, 4) ||
        write_all(fd, &g->n_plies, 4) ||
        write_all(fd, &g->end_kind, 1) ||
        write_all(fd, &g->resigner, 1) ||
        write_all(fd, &g->n_sims, 4) ||
        write_all(fd, &g->z_white, 8) ||
        write_all(fd, fen, SP_FEN_MAX))
        return -1;
    return 0;
}

/* f32 → f16 bits, round-to-nearest-even. Uses the mantissa-carry trick:
 * a rounded-up 0x400 mantissa ORs into the exponent field, which is
 * exactly the encoding the rounded value needs — including the carry
 * out of the top normal → Inf. Exact-general (NaN/Inf/subnormals), not
 * just the [0,1] probs this writer actually sees. */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof x);
    uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    int exp16 = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (((x >> 23) & 0xffu) == 0xffu) {         /* Inf / NaN */
        uint32_t m16 = m >> 13;
        return (uint16_t)(sign | 0x7c00u | (m ? (m16 ? m16 : 1u) : 0u));
    }
    if (exp16 >= 0x1f)                          /* |x| ≥ 2^16 → ±Inf */
        return (uint16_t)(sign | 0x7c00u);
    if (exp16 <= 0) {                           /* f16 subnormal / flush */
        if (exp16 < -10)                        /* |x| < 2^-25 rounds to 0 */
            return sign;
        m |= 0x800000u;                         /* the implicit leading 1 */
        uint32_t shift = (uint32_t)(14 - exp16);
        uint32_t half = 1u << (shift - 1);
        uint32_t kept = m >> shift;
        uint32_t rem = m & ((half << 1) - 1u);
        if (rem > half || (rem == half && (kept & 1u)))
            kept++;
        return (uint16_t)(sign | kept);         /* kept 0x400 → 2^-14 normal */
    }
    uint32_t kept = m >> 13;                    /* top 10 mantissa bits */
    uint32_t rem = m & 0x1fffu;                 /* 13 dropped bits */
    if (rem > 0x1000u || (rem == 0x1000u && (kept & 1u)))
        kept++;
    return (uint16_t)(sign | ((uint32_t)exp16 << 10) | kept);
}

int sp_dump_ply(int fd, const SpPlyRec *r) {
    char uci[SP_UCI_MAX] = {0};
    strncpy(uci, r->played_uci, SP_UCI_MAX - 1);
    if (write_all(fd, &r->mover, 1) ||
        write_all(fd, uci, SP_UCI_MAX) ||
        write_all(fd, &r->n_pi, 2) ||
        write_all(fd, &r->outlook, 8) ||
        write_all(fd, &r->sims, 2))
        return -1;
    uint16_t h[SP_PI_MAX];
    for (uint16_t i = 0; i < r->n_pi; i++)
        h[i] = f32_to_f16(r->pi_prob[i]);
    return write_all(fd, h, (size_t)r->n_pi * sizeof h[0]);
}

int sp_dump_game(int fd, const SpGameRec *g) {
    if (sp_dump_game_head(fd, g))
        return -1;
    for (uint32_t p = 0; p < g->n_plies; p++)
        if (sp_dump_ply(fd, &g->plies[p]))
            return -1;
    return 0;
}

int sp_dump_write_done(const char *dir, int n_workers, uint64_t unit_seed,
                       uint32_t games_done) {
    char path[512];
    snprintf(path, sizeof path, "%s/DONE.json", dir);
    FILE *f = fopen(path, "w");
    if (!f)
        return -1;
    int ok = fprintf(f, "{\"n_workers\": %d, \"unit_seed\": %llu, "
                        "\"games_done\": %u}\n",
                     n_workers, (unsigned long long)unit_seed,
                     games_done) > 0;
    fclose(f);
    return ok ? 0 : -1;
}
