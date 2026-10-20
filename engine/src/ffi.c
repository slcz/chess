/*
 * ffi.c — implementation of the chess_ffi.h handle API.
 *
 * A ChessGame is a State plus the move log. Why the log: unmake_move()
 * needs the move itself to undo it, and the rules engine deliberately does
 * not store moves in State (the search makes/unmakes with moves it holds
 * in its own hands). A guest language cannot easily do that, so the handle
 * keeps the played moves and replays them backward on pop().
 *
 * chess_push_uci() validates against the legal-move list, so a guest can
 * never corrupt a game with a nonsense move.
 */
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "chess_ffi.h"
#include "chess_nn.h" /* Phase 2: NnHistory ring behind chess_encode() */

struct ChessGame {
    State    st;
    Move     moves[MAX_PLY]; /* moves[base_ply..st.ply-1]: the log of
                              * moves played since construction */
    int      base_ply;       /* st.ply at construction/set_fen — the log's
                              * origin (a FEN start lands mid-count, its
                              * ply is nonzero but nothing was played) */
    NnHistory nn;            /* placement snapshots feeding chess_encode().
                              * Kept in step with pushes/pops below, so the
                              * guest never touches it. */
};

/* Start a fresh history ring for the current position (game creation and
 * set_fen both land here — a FEN start has no past, by definition). */
static void nn_reset_and_record(ChessGame *g) {
    nn_history_reset(&g->nn);
    nn_history_record(&g->nn, &g->st);
}

ChessGame *chess_game_new(const char *fen) {
    ChessGame *g = malloc(sizeof *g);
    if (!g) return NULL;
    if (!fen || !*fen) {
        state_init_startpos(&g->st);
        g->base_ply = g->st.ply;
        nn_reset_and_record(g);
        return g;
    }
    if (fen_parse(&g->st, fen) != 0) {
        free(g);
        return NULL;
    }
    g->base_ply = g->st.ply;
    nn_reset_and_record(g);
    return g;
}

void chess_game_free(ChessGame *g) {
    free(g);
}

int chess_set_fen(ChessGame *g, const char *fen) {
    int rc = fen_parse(&g->st, fen) == 0 ? 0 : -1;
    if (rc == 0) {
        g->base_ply = g->st.ply;
        nn_reset_and_record(g);
    }
    return rc;
}

void chess_fen(const ChessGame *g, char *out) {
    fen_serialize(&g->st, out, CHESS_FEN_MAX);
}

int chess_legal_moves(ChessGame *g, char *out, int max) {
    MoveList ml;
    gen_legal(&g->st, &ml);
    int n = ml.count < max ? ml.count : max;
    for (int i = 0; i < n; i++)
        move_to_uci(ml.moves[i], out + (size_t)i * CHESS_UCI_MAX);
    return ml.count;
}

int chess_push_uci(ChessGame *g, const char *uci) {
    if (!uci) return -1;
    size_t len = strlen(uci);
    if (len < 4 || len > 5) return -1;

    MoveList ml;
    gen_legal(&g->st, &ml);
    for (int i = 0; i < ml.count; i++) {
        char candidate[CHESS_UCI_MAX];
        move_to_uci(ml.moves[i], candidate);
        if (strcmp(candidate, uci) == 0) {
            make_move(&g->st, ml.moves[i]);
            g->moves[g->st.ply - 1] = ml.moves[i];
            nn_history_record(&g->nn, &g->st); /* keep the encoder's window in step */
            return 0;
        }
    }
    return -1;
}

int chess_pop(ChessGame *g) {
    if (g->st.ply == 0) return -1;
    unmake_move(&g->st, g->moves[g->st.ply - 1]);
    nn_history_drop(&g->nn);
    return 0;
}

/* --- observation: one-line forwards, with the const-ness gen_legal needs --- */

int chess_termination(ChessGame *g)          { return state_termination(&g->st); }
/* The individual game-over predicates (checkmate/stalemate/material/
 * fifty-move/repetition) were removed with the Python GUI: every reason
 * is already a TERM_* code of chess_termination, which is all any
 * remaining consumer (tests, the web worker) ever asked. */
int chess_is_repetition(ChessGame *g, int count) { return state_is_repetition(&g->st, count); }

int chess_in_check(ChessGame *g)        { return in_check(&g->st); }
int chess_side(ChessGame *g)            { return g->st.side; }
int chess_ply(ChessGame *g)             { return g->st.ply; }
int chess_piece_at(ChessGame *g, int sq){ return (sq >= 0 && sq < 64) ? g->st.board[sq] : EMPTY; }

uint64_t chess_perft(ChessGame *g, int depth) { return perft(&g->st, depth); }

/* --- NN input encoding (Phase 2.1) ------------------------------------- */

int chess_encode(ChessGame *g, float *out) {
    if (!out) return -1;
    nn_encode(&g->st, &g->nn, out);
    return 0;
}
