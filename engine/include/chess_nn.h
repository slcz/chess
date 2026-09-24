/*
 * chess_nn.h — NN input-feature encoder (Phase 2.1), isolated from the rules.
 *
 * WHY A SEPARATE MODULE
 * =====================
 * The rules engine (chess.h / board.c / movegen.c) is done, perft-verified,
 * and performance-tuned; the ML side of the project must not grow into it.
 * Everything NN-specific lives here: the feature layout, the history window,
 * and the encoder. The module is *pure*: it reads the current `State` and a
 * caller-maintained snapshot history, and writes floats. It never mutates a
 * State, keeps no globals, and needs nothing from the rules engine beyond
 * the public header.
 *
 * THE ENCODED TENSOR (the one authoritative spec; chess_ai/net/features.py
 * mirrors it and the tests assert the two agree bit-for-bit)
 * =====================================================================
 * 64 tokens, one per square, NN_TOKEN_DIM = 117 floats each, laid out
 * token-major: out[token * 117 + dim]. Token index = square from the CURRENT
 * MOVER's perspective (the AZ/Lc0 convention: the mover always "sits at the
 * bottom"):  token = sq            if the side to move is WHITE,
 *           token = sq ^ 56        if it is BLACK (vertical flip).
 *
 * Per-token dims:
 *   [0, 96)    board history, 8 plies x 12 piece one-hots. h = 0 is the
 *              current position, h = 1 one move earlier, ... h = 7. Within a
 *              ply: 6*is_enemy + piece_type, where "enemy" is relative to
 *              the CURRENT mover (constant across frames — the window keeps
 *              one orientation, it does not re-flip per ply). Plies older
 *              than the game itself (short games, FEN starts) are all-zero.
 *   [96, 104)  repetition flag per history ply: 1.0 if the position 8-h
 *              plies ago had already occurred EARLIER in this game. Raw
 *              Zobrist-key match over State.keys (feature-grade; the exact
 *              FIDE en-passant refinement stays in termination.c, which is
 *              the only place that must be exact).
 *   [104, 112) en-passant target FILE one-hot of the current position
 *              (all-zero when there is no ep square; a file is orientation-
 *              independent, so no flip is needed).
 *   [112, 116) castling rights of the current position, mover-first:
 *              [mover O-O, mover O-O-O, enemy O-O, enemy O-O-O].
 *   [116]      halfmove clock / 100.0 of the current position.
 *
 * HISTORY (NnHistory)
 * ===================
 * The encoder needs the piece placement of the last NN_HISTORY-1 positions;
 * State deliberately does not keep that (rules code is done), so this module
 * owns a small ring of mailbox snapshots. Whoever advances the game records
 * a snapshot on every position change:
 *
 *     nn_history_reset(h);                  // on game setup
 *     nn_history_record(h, s);              // position 0 is IN the ring
 *     ... make_move ... nn_history_record(h, s);   // each new position
 *
 * The ring keeps the last NN_HISTORY snapshots *including* the current
 * position (nn_encode still reads the placement of "now" from the State
 * itself — the ring copy would also be correct, but the State is the more
 * authoritative view and this keeps record() optional for the h=0 frame).
 * nn_encode() consumes (State, ring) with no coupling between them beyond
 * the caller keeping them in step.
 */
#ifndef CHESS_NN_H
#define CHESS_NN_H

#include "chess.h" /* State, piece-code helpers, MAX_PLY — read-only use */

#ifdef __cplusplus
extern "C" {
#endif

/* Feature-tensor constants (see the layout spec above). */
#define NN_HISTORY   8     /* plies of board history incl. the current one */
#define NN_TOKENS    64    /* one token per square                          */
#define NN_TOKEN_DIM 117   /* 96 pieces + 8 rep-flags + 8 ep + 4 castle + 1 */
#define NN_TOTAL     (NN_TOKENS * NN_TOKEN_DIM) /* 7488 floats              */

/* Ring of recent mailbox snapshots (see above). Small enough to sit by
 * value inside a game handle; zero-init/reset is a valid empty state. */
typedef struct {
    uint8_t board[NN_HISTORY][64]; /* [0] oldest … [count-1] newest         */
    int     count;                 /* snapshots held, 0 … NN_HISTORY        */
} NnHistory;

/* Empty the ring (call when a game is created / reset from a FEN). */
void nn_history_reset(NnHistory *h);

/* Snapshot `s`'s piece placement into the ring (call on arrival at every
 * new position, including the first). Oldest snapshot falls off at 8. */
void nn_history_record(NnHistory *h, const State *s);

/* Drop the newest snapshot (mirror of record, for undo; not below empty). */
void nn_history_drop(NnHistory *h);

/* Encode the current position `s` with history `h` into out (caller
 * allocates NN_TOTAL floats, token-major [64][117]). Pure function: reads
 * only. Frames for plies older than the ring (or older than the game) come
 * out zero, matching the AZ convention for missing history. */
void nn_encode(const State *s, const NnHistory *h, float out[NN_TOTAL]);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_NN_H */
