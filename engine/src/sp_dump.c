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

int sp_dump_ply(int fd, const SpPlyRec *r) {
    char uci[SP_UCI_MAX] = {0};
    strncpy(uci, r->played_uci, SP_UCI_MAX - 1);
    if (write_all(fd, &r->mover, 1) ||
        write_all(fd, &r->played_slot, 2) ||
        write_all(fd, uci, SP_UCI_MAX) ||
        write_all(fd, &r->n_pi, 2) ||
        write_all(fd, &r->outlook, 8) ||
        write_all(fd, &r->sims, 2))
        return -1;
    for (uint16_t i = 0; i < r->n_pi; i++)
        if (write_all(fd, &r->pi_slot[i], 2) ||
            write_all(fd, &r->pi_prob[i], 4))
            return -1;
    return 0;
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
