/*
 * chess_sp_dump.h — finished-game record dump for the self-play driver
 * (selfplay.c). Byte format identical to the SPSHARD1 shard format this
 * project's trainer tooling already parses — this header is deliberately
 * self-contained (stdint only, no search-engine types): the dump works on
 * plain fields the self-play driver reads straight off the PUCT readouts.
 *
 * THE FILE FORMAT (little-endian, packed)
 * =====================================================================
 * Shard = 24-byte header, then finished games concatenated back to back,
 * one file per worker (selfplay writes worker 0 only):
 *
 *   shard header, 24 B:
 *     char[8]  magic      "SPSHARD1"
 *     u32      version    2 (v1 = pre-PCR: no per-ply sims field)
 *     u32      worker     writer index
 *     u64      unit_seed  the run's seed (ingest dedups (seed, game_index))
 *
 *   game record (self-delimiting: a reader consumes exactly n_plies ply
 *   blocks, so a run killed mid-write leaves only a truncated LAST record
 *   the reader detects as a short read and drops):
 *     u32      game_index
 *     u32      n_plies
 *     u8       end_kind     SP_END_* (NATURAL = rules termination,
 *                            ADJUDICATED = ply cap)
 *     i8       resigner     -1 none (no resignation in this driver; the
 *                            field keeps the shared format intact)
 *     u32      n_sims       the slow/nominal search budget (the training
 *                            cap under playout cap randomization)
 *     f64      z_white      1 / 0.5 / 0 from White's perspective
 *     char[96] start_fen    NUL-padded
 *     n_plies × ply block:
 *       u8      mover       0 white / 1 black
 *       i16     played_slot -1 on a resign ply (never written here)
 *       char[8] played_uci  NUL-padded
 *       u16     n_pi        π support: the root moves that got visits
 *       f64     outlook     root value in the mover's view (-(W/N) of
 *                            the root, whose W is parent-view) — future
 *                            resignation / loss-weighting input
 *       u16     sims        v2 only: this ply's actual search budget
 *                            (playout cap randomization: the trainer
 *                            keeps plies with sims == n_sims)
 *       n_pi ×  { u16 slot, f32 prob }   π ∝ visits, normalized to 1
 *
 * Per-ply FENs are NOT stored: the reader replays played_uci from
 * start_fen through the engine. The ply number is the block's index.
 */
#ifndef CHESS_SP_DUMP_H
#define CHESS_SP_DUMP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SP_FEN_MAX   96
#define SP_UCI_MAX   8
#define SP_PI_MAX    256   /* ≤ legal-move bound (MAX_MOVES); u16 in file */
#define SP_VERSION   2     /* v1 = pre-PCR (no per-ply sims field)        */

enum {
    SP_END_PLAYING = 0,   /* never written — reader-side state only      */
    SP_END_NATURAL,       /* state_termination fired (mate / draw rules) */
    SP_END_RESIGN,        /* never written in v1 (format parity)         */
    SP_END_ADJUDICATED    /* max_plies reached                            */
};

/* One ply as the driver records it, before the byte serialization. */
typedef struct SpPlyRec {
    uint8_t  mover;
    int16_t  played_slot;
    char     played_uci[SP_UCI_MAX];
    uint16_t n_pi;
    double   outlook;
    uint16_t sims;             /* this ply's actual search budget (v2) */
    uint16_t pi_slot[SP_PI_MAX];
    float    pi_prob[SP_PI_MAX];
} SpPlyRec;

/* One finished game. */
typedef struct SpGameRec {
    uint32_t     game_index;
    uint32_t     n_plies;
    uint8_t      end_kind;
    int8_t       resigner;        /* always -1 from this driver */
    uint32_t     n_sims;
    double       z_white;
    char         start_fen[SP_FEN_MAX];
    const SpPlyRec *plies;        /* n_plies entries */
} SpGameRec;

/* "shard-<worker>.spbin" inside `dir`. */
void sp_dump_shard_path(const char *dir, int worker, char *buf, size_t cap);

/* Create `dir` if needed, open its shard-<worker>.spbin with O_TRUNC (a
 * shard belongs to exactly one run) and write the 24-byte header.
 * Returns the fd or -1. */
int  sp_dump_open_shard(const char *dir, int worker, uint64_t unit_seed);

/* Append one finished game. 0 ok, -1 on any failed/short write — the
 * caller fails the run: a silently lost record would corrupt the
 * training set's accounting. (Convenience wrapper over the two halves
 * below; the self-play driver streams instead — its per-slot memory is
 * compact, not SpPlyRec-sized.) */
int  sp_dump_game(int fd, const SpGameRec *g);

/* Streaming halves of sp_dump_game: head, then each ply in order. */
int  sp_dump_game_head(int fd, const SpGameRec *g);
int  sp_dump_ply(int fd, const SpPlyRec *r);

/* Write DONE.json into `dir` on completion (the trainer's unit marker:
 * n_workers, unit_seed, games_done). 0 ok, -1 on write failure. */
int  sp_dump_write_done(const char *dir, int n_workers, uint64_t unit_seed,
                        uint32_t games_done);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_SP_DUMP_H */
