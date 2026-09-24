/*
 * nn_encode.c — implementation of the Phase 2.1 input encoder (chess_nn.h).
 *
 * Everything here is a pure read over (State, NnHistory): no make/unmake
 * walking, no mutation, no globals — so the rules engine stays untouched and
 * the encoder is safe to call on shared states from any number of games.
 *
 * The two non-trivial parts:
 *   1. Frame reconstruction. The ring holds the last 8 placements in order;
 *      history index h (h = 1 … 7) is the placement of position ply-h, which
 *      is ring[count-1-h] — one index arithmetic per frame, no copies.
 *   2. Repetition flags. For window ply t, "already occurred earlier" is a
 *      scan of State.keys[0 … t-1] for an equal key. The scan is bounded by
 *      the halfmove clock at t (an equal placement/side/castling key across
 *      a pawn move or capture is impossible — those are irreversible — so
 *      anything older than the clock window cannot match; the bound is an
 *      optimization, not a correctness assumption). The clock at position t
 *      is hist[t].halfmove (make_move stores the pre-move clock at index
 *      t; for t == ply it is s->halfmove itself).
 */
#include <string.h>

#include "chess_nn.h"

void nn_history_reset(NnHistory *h) {
    h->count = 0;
}

void nn_history_record(NnHistory *h, const State *s) {
    if (h->count == NN_HISTORY) {
        memmove(h->board[0], h->board[1],
                sizeof h->board[0] * (size_t)(NN_HISTORY - 1));
        memcpy(h->board[NN_HISTORY - 1], s->board, sizeof h->board[0]);
    } else {
        memcpy(h->board[h->count], s->board, sizeof h->board[0]);
        h->count++;
    }
}

void nn_history_drop(NnHistory *h) {
    if (h->count > 0) h->count--;
}

void nn_encode(const State *s, const NnHistory *hist, float out[NN_TOTAL]) {
    const int mover = s->side;
    const int flip  = (mover == BLACK);

    memset(out, 0, (size_t)NN_TOTAL * sizeof *out);

    /* --- per-ply repetition flags, from State.keys (see file comment) --- */
    float repflag[NN_HISTORY];
    for (int h = 0; h < NN_HISTORY; h++) {
        const int t = s->ply - h;          /* position index of frame h   */
        repflag[h] = 0.0f;
        if (t < 1) continue;               /* no earlier plies to repeat  */
        const int clock = (t == s->ply) ? s->halfmove : s->hist[t].halfmove;
        int lo = t - clock;                /* earliest possible repeat    */
        if (lo < 0) lo = 0;
        const uint64_t key = s->keys[t];
        for (int q = lo; q < t; q++) {
            if (s->keys[q] == key) { repflag[h] = 1.0f; break; }
        }
    }

    /* --- board-history frames ------------------------------------------ */
    /* Frame h = placement of position (ply - h): h = 0 from the live State,
     * h >= 1 from the ring (index count-1-h; out of range = before the game
     * = stays zero). All frames share the CURRENT mover's orientation and
     * color ordering (see the chess_nn.h layout spec). */
    for (int h = 0; h < NN_HISTORY; h++) {
        const uint8_t *placement;
        if (h == 0) {
            placement = s->board;
        } else {
            const int idx = hist->count - 1 - h;
            if (idx < 0) break;            /* ring exhausted: rest stays 0 */
            placement = hist->board[idx];
        }
        for (int sq = 0; sq < 64; sq++) {
            const int pc = placement[sq];
            if (pc == EMPTY) continue;
            const int token = flip ? (sq ^ 56) : sq;
            const int dim = h * 12
                          + (pc_color(pc) == mover ? pc_type(pc) : 6 + pc_type(pc));
            out[token * NN_TOKEN_DIM + dim] = 1.0f;
        }
    }

    /* --- global (per-position) features, broadcast to every token ------ */
    for (int token = 0; token < NN_TOKENS; token++) {
        float *row = out + token * NN_TOKEN_DIM;

        for (int h = 0; h < NN_HISTORY; h++)
            row[96 + h] = repflag[h];

        if (s->ep != NO_SQ)
            row[104 + (s->ep & 7)] = 1.0f; /* file is flip-invariant      */

        /* Castling bits: CASTLE_WK=1, WQ=2, BK=4, BQ=8 — reorder so the
         * mover's own rights come first regardless of color. */
        const int c = s->castling;
        if (mover == WHITE) {
            row[112] = (float)(c & CASTLE_WK);
            row[113] = (float)((c >> 1) & 1);
            row[114] = (float)((c >> 2) & 1);
            row[115] = (float)((c >> 3) & 1);
        } else {
            row[112] = (float)((c >> 2) & 1);
            row[113] = (float)((c >> 3) & 1);
            row[114] = (float)(c & 1);
            row[115] = (float)((c >> 1) & 1);
        }

        row[116] = (float)s->halfmove / 100.0f;
    }
}
