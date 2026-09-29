/*
 * chess_sp_dump.h — the finished-game record dump (collect-time shard
 * files, user decision 2026-09-28 — supersedes deviation 10's drain
 * callback).
 *
 * When a game finishes, its worker dumps the record to ITS OWN shard file
 * (one file per worker — no locking, no interleaved writers) and returns
 * the pool slot immediately. Nothing keeps finished games in memory: no
 * done ring, no per-epoch drain, no Python callback on the record path.
 * The trainer reads the shards back later and converts to tensors
 * (deliberately deferred — 3.4).
 *
 * THE FILE FORMAT (little-endian, packed — the trainer's reader contract)
 * =======================================================================
 * Shard = header, then games concatenated back to back:
 *
 *   shard header, 24 B:
 *     char[8]  magic      "SPSHARD1"
 *     u32      version    1
 *     u32      worker     writer index (0..n_workers-1)
 *     u64      unit_seed  the run's seed
 *
 *   game record (self-delimiting — a reader consumes exactly n_plies ply
 *   blocks, so a run killed mid-write leaves only a truncated LAST record,
 *   which the reader detects as a short read):
 *     u32      game_index
 *     u32      n_plies
 *     u8       end_kind     SP_END_*
 *     i8       resigner     -1 none / 0 white / 1 black
 *     u32      n_sims       cfg.n
 *     f64      z_white
 *     char[96] start_fen    NUL-padded (CHESS_FEN_MAX)
 *     n_plies × ply block:
 *       u8      mover       0 white / 1 black
 *       i16     played_slot -1 for the resign ply
 *       char[8] played_uci  NUL-padded, empty for the resign ply
 *       u16     n_pi        π′ support ≤ SP_REC_CANDIDATES
 *       f64     outlook     the resignation outlook
 *       n_pi ×  { u16 slot, f32 prob }
 *
 * Per-ply FENs are NOT stored (same decision as the in-memory records):
 * a reader replays played_uci from start_fen — the sequential exporter's
 * rule, so both paths consume identical information. The ply number is
 * the block's index (games start at ply 0).
 */
#ifndef CHESS_SP_DUMP_H
#define CHESS_SP_DUMP_H

#include <stddef.h>
#include <stdint.h>

#include "chess_search.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SP_DUMP_PATH_MAX 256

/* shard-<worker>.spbin, placed in `dir` — the one filename convention,
 * shared by the engine (writer), the C tests (reader) and the Python
 * reader (spengine.py mirrors the literal). */
void sp_dump_shard_path(const char *dir, int worker, char *buf,
                        size_t cap);

/* Open the worker's shard in `dir` (O_TRUNC — a shard file belongs to
 * exactly one run) and write the 24-byte header. Returns the fd or -1. */
int  sp_dump_open_shard(const char *dir, int worker, uint64_t unit_seed);

/* Append one finished game's record to `fd`. 0 ok, -1 on any short/
 * failed write (the caller fails the run — a silently lost record would
 * corrupt the training set's accounting). */
int  sp_dump_game(int fd, const SpGame *g);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_SP_DUMP_H */
