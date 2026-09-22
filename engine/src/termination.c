/*
 * termination.c — game-over detection (Phase 1.3).
 *
 * Five ways a chess game ends:
 *   checkmate, stalemate, dead material, the 50/75-move clocks, and
 *   repetition (threefold claimable / fivefold automatic since 2014).
 *
 * Everything here mirrors python-chess *exactly*, because python-chess is
 * the fuzz oracle for Phase 1.5 (PLAN §10.3): at every position of every
 * random walk, our answer must equal Board.outcome(claim_draw=True) —
 * same reason, same precedence, same winner. The interesting engineering
 * is in the repetition check; see state_is_repetition below.
 *
 * Like gen_legal, several of these functions make/unmake moves on `s` to
 * test king safety (the legality filter pattern); the State is bit-identical
 * when they return, but callers must not share one State across threads
 * while a call is in flight.
 */
#include "chess.h"

/* ------------------------------------------------------------------ */
/* Position key                                                        */
/* ------------------------------------------------------------------ */

/* Recompute the position key from scratch (placement + side + castling,
 * en passant and clocks excluded — see the State.keys comment for why).
 * fen_parse uses this once for the root position; the random-game tests
 * call it every ply to verify make_move's incremental XORs agree. */
uint64_t state_poskey(const State *s) {
    uint64_t k = 0;
    for (int c = 0; c < 2; c++)
        for (int t = 0; t < 6; t++) {
            uint64_t b = s->bb[c][t];
            while (b) {
                int sq = __builtin_ctzll(b);
                b &= b - 1;
                k ^= zob_psq[pc_of(c, t)][sq];
            }
        }
    k ^= zob_castle[s->castling];
    if (s->side == BLACK) k ^= zob_side;
    return k;
}

/* ------------------------------------------------------------------ */
/* En passant, the repetition subtlety                                 */
/* ------------------------------------------------------------------ */

/*
 * FIDE Article 9.2: positions count as "the same" only if the same player
 * is to move, the same pieces stand on the same squares — and the same
 * moves, *including en passant*, are available. An ep square that is set
 * but not legally capturable therefore does NOT distinguish positions.
 * (python-chess: has_legal_en_passant().)
 *
 * Returns the ep square if an en-passant capture is legal in `s` right now,
 * NO_SQ otherwise. Probes each candidate pawn with make/unmake — at most
 * two candidates exist, and this only runs at repetition-query time, never
 * in make_move's hot path.
 */
static int effective_ep(State *s, int ep_sq) {
    if (ep_sq == NO_SQ) return NO_SQ;
    if ((occ_all(s) >> ep_sq) & 1) return NO_SQ; /* occupied target: corrupt FEN */

    const int us = s->side;
    uint64_t cand = pawn_att_tbl[us ^ 1][ep_sq] & s->bb[us][PAWN]; /* mirror trick */
    while (cand) {
        int from = __builtin_ctzll(cand);
        cand &= cand - 1;
        Move m = move_make(from, ep_sq, F_EP);
        make_move(s, m);
        bool ok = !is_attacked(s, king_sq(s, us), s->side);
        unmake_move(s, m);
        if (ok) return ep_sq;
    }
    return NO_SQ;
}

bool state_is_repetition(State *s, int count) {
    if (count <= 1) return true; /* the current position trivially counts */

    const uint64_t cur_key = s->keys[s->ply];
    const int cur_ep = effective_ep(s, s->ep);

    /* Only positions since the last irreversible move can repeat: a pawn
     * never moves backwards and material never comes back, so any repeated
     * placement lives inside the halfmove-clock window. (python-chess
     * additionally stops its walk at castling-rights changes; unnecessary
     * here because the rights are part of the key — a rights change can
     * only produce key-unequal positions, which we skip anyway.) */
    int stop = (int)s->ply - (int)s->halfmove;
    if (stop < 0) stop = 0;

    int seen = 1; /* the current position */
    for (int i = (int)s->ply - 1; i >= stop && seen < count; i--) {
        if (s->keys[i] != cur_key) continue;

        /* Raw key match: identical placement, side to move, castling
         * rights. The positions are truly "the same" iff effective en
         * passant also matches — and because the placement is identical,
         * evaluating ep legality HERE (on the current board) answers it
         * for the historical position too. hist[i].ep is the ep square
         * that position i had (it is the value saved before move i). */
        if (effective_ep(s, s->hist[i].ep) == cur_ep) seen++;
    }
    return seen >= count;
}

/* ------------------------------------------------------------------ */
/* Mate / stalemate / clocks                                           */
/* ------------------------------------------------------------------ */

static bool no_legal_moves(State *s) {
    MoveList ml;
    gen_legal(s, &ml);
    return ml.count == 0;
}

bool state_is_checkmate(State *s) {
    return in_check(s) && no_legal_moves(s);
}

bool state_is_stalemate(State *s) {
    return !in_check(s) && no_legal_moves(s);
}

bool state_halfmoves(State *s, int n) {
    if (s->halfmove < (uint16_t)n) return false;
    return !no_legal_moves(s); /* python-chess: ... and any legal moves */
}

/* ------------------------------------------------------------------ */
/* The claimable draws, with their one-move lookahead                  */
/* ------------------------------------------------------------------ */

/*
 * "Claimable" has a subtlety the automatic rules lack: FIDE lets a player
 * claim a draw by the intended MOVE — so a position where the clock sits
 * at 99 and any quiet move exists is already claimable (the claim attaches
 * to the move about to be made), and likewise a position that has occurred
 * twice is claimable if some legal move would repeat it a third time.
 * python-chess's can_claim_fifty_moves / can_claim_threefold_repetition
 * implement exactly that, and outcome(claim_draw=True) uses them, so the
 * oracle fuzz would flag any difference. (In self-play the distinction is
 * cosmetic — without the lookahead the game simply ends one ply later —
 * but parity is parity.)
 */

static bool can_claim_fifty(State *s) {
    if (state_halfmoves(s, 100)) return true;
    if (s->halfmove < 99) return false;

    MoveList ml;
    gen_legal(s, &ml);
    for (int i = 0; i < ml.count; i++) {
        make_move(s, ml.moves[i]);
        /* Clock hits 100 unless the move resets it (make_move already
         * computed that), and the opponent must still have a move — a
         * quiet move that stalemates is not a claimable fifty. */
        bool ok = s->halfmove >= 100 && !no_legal_moves(s);
        unmake_move(s, ml.moves[i]);
        if (ok) return true;
    }
    return false;
}

static bool can_claim_threefold(State *s) {
    if (state_is_repetition(s, 3)) return true;

    /* Would some legal move reach a position that has occurred twice
     * before (and thus occurs a third time)? is_repetition(3) after the
     * make answers exactly that: the post-move key can only match window
     * positions, and itself makes the third occurrence. */
    MoveList ml;
    gen_legal(s, &ml);
    for (int i = 0; i < ml.count; i++) {
        make_move(s, ml.moves[i]);
        bool ok = state_is_repetition(s, 3);
        unmake_move(s, ml.moves[i]);
        if (ok) return true;
    }
    return false;
}

/* ------------------------------------------------------------------ */
/* Insufficient material                                               */
/* ------------------------------------------------------------------ */

/* Squares of one color complex: a1, c1, e1, g1, b2, d2, ... (a1 is dark).
 * Only used to check that every bishop on the board sits on one complex. */
#define DARK_SQ_BB 0xAA55AA55AA55AA55ULL

/* Can `color` still win with this material? Faithful port of python-chess's
 * has_insufficient_material: it looks at material only (not placement), so
 * it can say "insufficient" for fortress-like positions where no win exists
 * — that's fine, it matches the oracle. is_insufficient_material() is then
 * "neither side can win". */
static bool has_insufficient_material(const State *s, int color) {
    const int them = color ^ 1;

    if (s->occ[color] & (s->bb[color][PAWN] | s->bb[color][ROOK] | s->bb[color][QUEEN]))
        return false;

    if (s->bb[color][KNIGHT]) {
        /* King alone or with one knight, and the opponent owns nothing that
         * could ever help us selfmate them (only kings/queens are inert
         * enough — their pawns/knights/bishops/rooks would allow a mate
         * around our knight). */
        return __builtin_popcountll(s->occ[color]) <= 2 &&
               !(s->occ[them] & ~(s->bb[them][KING] | s->bb[them][QUEEN]));
    }

    if (s->bb[color][BISHOP]) {
        /* Every bishop on the board (both colors) on the same square color,
         * and no pawns/knights anywhere. */
        uint64_t all_bishops = s->bb[WHITE][BISHOP] | s->bb[BLACK][BISHOP];
        if (all_bishops & DARK_SQ_BB && all_bishops & ~DARK_SQ_BB) return false;
        if (s->bb[WHITE][PAWN] | s->bb[BLACK][PAWN]) return false;
        if (s->bb[WHITE][KNIGHT] | s->bb[BLACK][KNIGHT]) return false;
        return true;
    }

    return true; /* bare king */
}

bool state_insufficient_material(const State *s) {
    return has_insufficient_material(s, WHITE) && has_insufficient_material(s, BLACK);
}

/* ------------------------------------------------------------------ */
/* The composed answer                                                 */
/* ------------------------------------------------------------------ */

int state_termination(State *s) {
    if (state_is_checkmate(s))          return TERM_CHECKMATE;
    if (state_insufficient_material(s)) return TERM_INSUFFICIENT_MATERIAL;
    if (state_is_stalemate(s))          return TERM_STALEMATE;
    if (state_halfmoves(s, 150))        return TERM_SEVENTYFIVE_MOVES;
    if (state_is_repetition(s, 5))      return TERM_FIVEFOLD_REPETITION;
    if (can_claim_fifty(s))             return TERM_FIFTY_MOVES;
    if (can_claim_threefold(s))         return TERM_THREEFOLD_REPETITION;
    return TERM_PLAYING;
}
