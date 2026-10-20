/*
 * chess_ffi.h — the C-ABI surface consumed by Python (cffi) and any other
 * foreign runtime (Phase 0.3 / the Phase 1 gate "perft via FFI").
 *
 * DESIGN: an *opaque handle per game*. The rules engine proper (chess.h)
 * works on caller-owned `State` structs — perfect inside C, awkward across
 * a language boundary, where mirroring a 16 KB struct layout in the guest
 * language is fragile and silent-breaking. Instead this shim owns a
 * malloc'd handle (State + the move log needed for undo), hands out only
 * pointers, and speaks small plain types: ints, uint64s, char buffers.
 * This is the one place the engine allocates; the zero-malloc guarantee
 * for the hot path is untouched.
 *
 * Conventions:
 *   - A handle is NOT thread-safe (gen_legal-style make/unmake probing
 *     happens inside calls). One handle per thread / per game; the
 *     underlying tables are read-only after chess_init, so any number of
 *     handles may be used concurrently.
 *   - Strings are NUL-terminated UTF-8/ASCII; buffers the caller allocates
 *     are sized with the CHESS_FEN_MAX / CHESS_UCI_MAX constants.
 *   - Legal moves come out as a packed char buffer, one 8-byte slot per
 *     move ("e2e4\0---"): array-out, no per-move callbacks, no format
 *     negotiation — cheap for cffi to slice.
 *   - int return of a yes/no question: 1 = yes, 0 = no; errors are -1.
 */
#ifndef CHESS_FFI_H
#define CHESS_FFI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHESS_FEN_MAX 96  /* longest FEN is ~90 chars; matches fen_serialize */
#define CHESS_UCI_MAX 8   /* "e7e8q\0" + slack; one slot per legal move      */

typedef struct ChessGame ChessGame; /* opaque: State + move log for undo */

/* Call chess_init() once per process before creating games. */
void chess_init(void);

/* Create a game from a FEN (NULL or "" = standard start position).
 * Returns NULL if the FEN is malformed. */
ChessGame *chess_game_new(const char *fen);
void       chess_game_free(ChessGame *g);

/* Replace the position (also resets the move log / repetition history).
 * 0 = ok, -1 = malformed FEN. */
int  chess_set_fen(ChessGame *g, const char *fen);

/* Serialize the current position into out (caller gives >= CHESS_FEN_MAX). */
void chess_fen(const ChessGame *g, char *out);

/* --- playing ----------------------------------------------------------- */

/* Write up to `max` legal moves as UCI strings, 8 bytes each, into `out`
 * (caller allocates >= max * CHESS_UCI_MAX). Returns the total legal-move
 * count (which may exceed `max` if the buffer was too small — then only
 * the first `max` were written). MAX_MOVES is the true bound. */
int  chess_legal_moves(ChessGame *g, char *out, int max);

/* Play `uci` (e.g. "e2e4", "e7e8q"). 0 = played, -1 = malformed/illegal. */
int  chess_push_uci(ChessGame *g, const char *uci);

/* Undo the last push. 0 = undone, -1 = no moves to undo. */
int  chess_pop(ChessGame *g);

/* --- observation ------------------------------------------------------- */

/* state_termination() codes: TERM_PLAYING / TERM_CHECKMATE / ... (chess.h).
 * TERM_CHECKMATE means the side to move lost; winner = !chess_side(g). */
int      chess_termination(ChessGame *g);
/* Every game-over reason is a TERM_* code of chess_termination() (the
 * header of chess.h lists them) — there are no separate predicates. */

int      chess_is_repetition(ChessGame *g, int count); /* occurred count times */

int      chess_in_check(ChessGame *g);
int      chess_side(ChessGame *g);             /* 0 = white, 1 = black       */
int      chess_ply(ChessGame *g);

/* Piece code on a square (color*6+type, 12 = EMPTY), for lightweight
 * evaluation in the guest language without parsing FEN. */
int      chess_piece_at(ChessGame *g, int sq);

/* --- perft (Phase 1 gate: identical numbers via the FFI) ---------------- */

uint64_t chess_perft(ChessGame *g, int depth);

/* --- NN input encoding (Phase 2.1; algorithm in nn_encode.c) -------------
 *
 * Writes the 64-token x 117-float feature tensor of the CURRENT position
 * (layout: engine/include/chess_nn.h) into `out`, which the caller sizes
 * CHESS_NN_TOTAL floats. The handle records the placement history of its
 * own pushes/pops, so encoding needs no extra bookkeeping from the guest.
 * Returns 0 (reserved for future error codes). */
#define CHESS_NN_TOTAL (64 * 117)   /* == NN_TOTAL; standalone by design   */
int  chess_encode(ChessGame *g, float *out);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_FFI_H */
