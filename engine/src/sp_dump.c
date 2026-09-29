/*
 * sp_dump.c — the finished-game record dump (see chess_sp_dump.h for the
 * file-format contract). Reads the game exclusively through the public
 * sp_rec_* / sp_result_* accessors — the same boundary the Python
 * exporter used, so the file carries exactly what records ever carried.
 *
 * Writes are raw fd write() calls (no stdio buffering, no allocation);
 * a ply block is assembled in a 128-byte stack buffer, so the cost per
 * finished game is 1 + n_plies syscalls to the page cache — negligible
 * next to the game's search, and no fsync: durability is the caller's
 * concern (3.4's shard writer / HF backup).
 */
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "chess_sp_dump.h"

static int write_full(int fd, const void *buf, size_t len) {
    const uint8_t *p = buf;
    while (len > 0) {
        ssize_t n = write(fd, p, len);
        if (n < 0)
            return -1;          /* EINTR included: a partial dump fails
                                   the run loudly rather than retrying
                                   mid-record */
        p += n;
        len -= (size_t)n;
    }
    return 0;
}

void sp_dump_shard_path(const char *dir, int worker, char *buf, size_t cap) {
    snprintf(buf, cap, "%s/shard-w%d.spbin", dir, worker);
}

int sp_dump_open_shard(const char *dir, int worker, uint64_t unit_seed) {
    char path[SP_DUMP_PATH_MAX + 32];
    sp_dump_shard_path(dir, worker, path, sizeof path);
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return -1;
    char hdr[24];
    memcpy(hdr, "SPSHARD1", 8);
    uint32_t version = 1, w = (uint32_t)worker;
    memcpy(hdr + 8, &version, 4);
    memcpy(hdr + 12, &w, 4);
    memcpy(hdr + 16, &unit_seed, 8);
    if (write_full(fd, hdr, sizeof hdr) != 0) {
        close(fd);
        return -1;
    }
    return fd;
}

int sp_dump_game(int fd, const SpGame *g) {
    char fen[CHESS_FEN_MAX];
    sp_result_start_fen(g, fen);

    /* Game header block: 4+4+1+1+4+8+96 (the layout chess_sp_dump.h
     * documents). Field-by-field memcpy — no struct padding in the file. */
    uint8_t hdr[4 + 4 + 1 + 1 + 4 + 8 + CHESS_FEN_MAX];
    uint32_t gi = (uint32_t)sp_result_game_index(g);
    uint32_t n_plies = (uint32_t)sp_record_count(g);
    uint32_t n_sims = (uint32_t)sp_result_n_sims(g);
    int8_t resigner = (int8_t)sp_result_resigner(g);
    double z = sp_result_z_white(g);
    memcpy(hdr + 0, &gi, 4);
    memcpy(hdr + 4, &n_plies, 4);
    hdr[8] = (uint8_t)sp_result_kind(g);
    hdr[9] = (uint8_t)resigner;
    memcpy(hdr + 10, &n_sims, 4);
    memcpy(hdr + 14, &z, 8);
    size_t flen = 0;
    while (flen < CHESS_FEN_MAX && fen[flen])
        flen++;
    memset(hdr + 22, 0, CHESS_FEN_MAX);
    memcpy(hdr + 22, fen, flen);
    if (write_full(fd, hdr, sizeof hdr) != 0)
        return -1;

    for (uint32_t i = 0; i < n_plies; i++) {
        int n_pi = sp_rec_n_pi(g, (int)i);
        if (n_pi < 0 || n_pi > SP_REC_CANDIDATES)
            return -1;
        char uci[8];
        sp_rec_played_uci(g, (int)i, uci);
        /* One write per ply: the whole block (≤ 21 + 6·16 = 117 B) is
         * assembled first. */
        uint8_t ply[21 + 6 * SP_REC_CANDIDATES];
        int16_t played = (int16_t)sp_rec_played_slot(g, (int)i);
        double outlook = sp_rec_outlook(g, (int)i);
        ply[0] = (uint8_t)sp_rec_mover(g, (int)i);
        memcpy(ply + 1, &played, 2);
        memcpy(ply + 3, uci, 8);
        uint16_t k = (uint16_t)n_pi;
        memcpy(ply + 11, &k, 2);
        memcpy(ply + 13, &outlook, 8);
        size_t off = 21;
        for (int j = 0; j < n_pi; j++) {
            uint16_t slot = (uint16_t)sp_rec_pi_slot(g, (int)i, j);
            float prob = (float)sp_rec_pi_prob(g, (int)i, j);
            memcpy(ply + off, &slot, 2);
            memcpy(ply + off + 2, &prob, 4);
            off += 6;
        }
        if (write_full(fd, ply, off) != 0)
            return -1;
    }
    return 0;
}
