/*
 * movegen.c — pseudo-legal generation + legality filter (Phase 1.2).
 *
 * Two stages (see chess.h file header for the rationale):
 *
 *   gen_pseudo_legal(): everything that obeys how pieces move — pawns push,
 *   capture diagonally, promote on the last rank, double-push from their home
 *   rank; leapers use their precomputed attack tables; sliders ray-scan until
 *   a blocker (capturing if it's an enemy). Castling is generated here too,
 *   but with its full legality already verified (rights present, path empty,
 *   king not in / passing through / landing on check) because those conditions
 *   are invisible once the move has been made — the generic filter can only
 *   answer "after this move, is my king attacked?".
 *
 *   gen_legal(): runs the filter — make the move, ask exactly that question,
 *   unmake. This catches every remaining illegality with zero special cases,
 *   including the classic trap: en passant is the only move where two pieces
 *   leave the board at once (the capturing pawn and the captured pawn), which
 *   can expose the king along the rank — a case pure pattern-matching
 *   generators famously get wrong.
 */
#include "chess.h"

/* Rank/file constants for pawn set-wise generation. */
#define FILEA_BB 0x0101010101010101ULL
#define FILEH_BB 0x8080808080808080ULL
#define RANK1_BB 0x00000000000000FFULL
#define RANK3_BB 0x0000000000FF0000ULL
#define RANK6_BB 0x0000FF0000000000ULL
#define RANK8_BB 0xFF00000000000000ULL

static inline void add_move(MoveList *ml, int from, int to, int flags) {
    assert(ml->count < MAX_MOVES);
    ml->moves[ml->count++] = move_make(from, to, flags);
}

/* Iterate a bitboard of *target* squares, emitting moves from `from` to each. */
static inline void add_targets(MoveList *ml, int from, uint64_t targets, uint64_t them_bb) {
    while (targets) {
        int to = __builtin_ctzll(targets);
        targets &= targets - 1;
        add_move(ml, from, to, (them_bb >> to) & 1 ? F_CAPTURE : F_QUIET);
    }
}

/* ------------------------------------------------------------------ */
/* Pawn moves (set-wise: generate for all pawns of a color at once)     */
/* ------------------------------------------------------------------ */

static void gen_pawn_moves(const State *s, MoveList *ml, int us) {
    const int    them    = us ^ 1;
    const uint64_t own    = s->occ[us];
    const uint64_t them_bb = s->occ[them];
    const uint64_t empt   = ~(own | them_bb);
    const uint64_t pawns  = s->bb[us][PAWN];
    const uint64_t promo_rank = us == WHITE ? RANK8_BB : RANK1_BB;

    /* Pushes: shifts by 8 never wrap around files, so no file masking needed. */
    const int      fwd     = us == WHITE ? 8 : -8;
    const uint64_t one     = us == WHITE ? (pawns << 8) & empt : (pawns >> 8) & empt;
    const uint64_t two     = us == WHITE ? ((one & RANK3_BB) << 8) & empt
                                          : ((one & RANK6_BB) >> 8) & empt;
    /* Diagonal captures *do* wrap (h5→a6 would be a legal-looking +9), so the
     * a/h-file pawns are excluded per direction. */
    const uint64_t cap_left  = us == WHITE ? ((pawns & ~FILEA_BB) << 7) & them_bb
                                           : ((pawns & ~FILEA_BB) >> 9) & them_bb;
    const uint64_t cap_right = us == WHITE ? ((pawns & ~FILEH_BB) << 9) & them_bb
                                           : ((pawns & ~FILEH_BB) >> 7) & them_bb;

    /* Quiet single pushes to non-promotion squares. */
    uint64_t b = one & ~promo_rank;
    while (b) {
        int to = __builtin_ctzll(b);
        b &= b - 1;
        add_move(ml, to - fwd, to, F_QUIET);
    }
    /* Double pushes (only pawns on their home rank can produce these, which
     * the RANK3/RANK6 mask on `one` guarantees). */
    b = two;
    while (b) {
        int to = __builtin_ctzll(b);
        b &= b - 1;
        add_move(ml, to - 2 * fwd, to, F_DOUBLE_PUSH);
    }
    /* Captures to non-promotion squares. */
    b = cap_left & ~promo_rank;
    while (b) {
        int to = __builtin_ctzll(b);
        b &= b - 1;
        add_move(ml, to - (us == WHITE ? 7 : -9), to, F_CAPTURE);
    }
    b = cap_right & ~promo_rank;
    while (b) {
        int to = __builtin_ctzll(b);
        b &= b - 1;
        add_move(ml, to - (us == WHITE ? 9 : -7), to, F_CAPTURE);
    }
    /* Promotions: each push/capture to the last rank expands into 4 moves
     * (N/B/R/Q). Flag = F_PROMO_* (non-capture) / F_PROMO_CAP_* (capture);
     * the promo piece type is KNIGHT + (flag & 3), keeping the four variants
     * contiguous. */
    static const int promo_flags[4] = {F_PROMO_N, F_PROMO_B, F_PROMO_R, F_PROMO_Q};
    static const int promo_cap_flags[4] = {F_PROMO_CAP_N, F_PROMO_CAP_B, F_PROMO_CAP_R, F_PROMO_CAP_Q};

    b = one & promo_rank;
    while (b) {
        int to = __builtin_ctzll(b);
        b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - fwd, to, promo_flags[i]);
    }
    b = cap_left & promo_rank;
    while (b) {
        int to = __builtin_ctzll(b);
        b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - (us == WHITE ? 7 : -9), to, promo_cap_flags[i]);
    }
    b = cap_right & promo_rank;
    while (b) {
        int to = __builtin_ctzll(b);
        b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - (us == WHITE ? 9 : -7), to, promo_cap_flags[i]);
    }

    /* En passant: the ep target is the square jumped over, so the capturing
     * pawns are exactly `our pawns ∩ squares that attack it` — which (by the
     * mirror trick) is the pawn-attack table of the *other* color at that
     * square. An ep square occupied by anything is never generated (guards
     * against corrupt FENs). */
    if (s->ep != NO_SQ && !((own | them_bb) & (1ULL << s->ep))) {
        uint64_t attackers = pawn_att_tbl[them][s->ep] & pawns;
        while (attackers) {
            int from = __builtin_ctzll(attackers);
            attackers &= attackers - 1;
            add_move(ml, from, s->ep, F_EP);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Castling                                                            */
/* ------------------------------------------------------------------ */

static void gen_castling(const State *s, MoveList *ml, int us) {
    const int them = us ^ 1;
    const int ksq  = us == WHITE ? E1 : E8;
    const uint64_t all = occ_all(s);

    /* king_home/rook homes and the two squares between; all four king-path
     * squares (from, through, landing) must be unattacked — the landing check
     * is redundant with the generic filter but keeps castling from pseudo-
     * legal already fully legal, which simplifies reasoning about stage 1. */
    const int k_from = ksq, k_to_ks = us == WHITE ? G1 : G8, k_to_qs = us == WHITE ? C1 : C8;
    const int r_ks   = us == WHITE ? H1 : H8, r_qs  = us == WHITE ? A1 : A8;
    const int through_ks = us == WHITE ? F1 : F8;
    const int through_qs = us == WHITE ? D1 : D8;
    const int b_qs       = us == WHITE ? B1 : B8; /* rook passes; king doesn't */

    const bool wk = us == WHITE;
    if ((s->castling & (wk ? CASTLE_WK : CASTLE_BK))
        && s->board[k_from] == pc_of(us, KING) && s->board[r_ks] == pc_of(us, ROOK)
        && !(all & (1ULL << through_ks)) && !(all & (1ULL << k_to_ks))
        && !is_attacked(s, k_from, them) && !is_attacked(s, through_ks, them)
        && !is_attacked(s, k_to_ks, them)) {
        add_move(ml, k_from, k_to_ks, F_KCASTLE);
    }
    if ((s->castling & (wk ? CASTLE_WQ : CASTLE_BQ))
        && s->board[k_from] == pc_of(us, KING) && s->board[r_qs] == pc_of(us, ROOK)
        && !(all & ((1ULL << b_qs) | (1ULL << through_qs) | (1ULL << k_to_qs)))
        && !is_attacked(s, k_from, them) && !is_attacked(s, through_qs, them)
        && !is_attacked(s, k_to_qs, them)) {
        add_move(ml, k_from, k_to_qs, F_QCASTLE);
    }
}

/* ------------------------------------------------------------------ */
/* Stage 1 + stage 2                                                   */
/* ------------------------------------------------------------------ */

void gen_pseudo_legal(const State *s, MoveList *ml) {
    ml->count = 0;
    const int us = s->side, them = us ^ 1;
    const uint64_t own = s->occ[us], them_bb = s->occ[them];

    gen_pawn_moves(s, ml, us);
    gen_castling(s, ml, us);

    for (int type = KNIGHT; type <= KING; type++) {
        uint64_t pieces = s->bb[us][type];
        while (pieces) {
            int from = __builtin_ctzll(pieces);
            pieces &= pieces - 1;

            uint64_t att;
            switch (type) {
            case KNIGHT: att = knight_tbl[from];                         break;
            case BISHOP: att = bishop_attacks(from, own | them_bb);      break;
            case ROOK:   att = rook_attacks(from, own | them_bb);        break;
            case QUEEN:  att = bishop_attacks(from, own | them_bb)
                              | rook_attacks(from, own | them_bb);       break;
            default:     att = king_tbl[from];                           break;
            }
            add_targets(ml, from, att & ~own, them_bb);
        }
    }
}

void gen_legal(State *s, MoveList *ml) {
    gen_pseudo_legal(s, ml);
    const int us = s->side;
    int n = 0;

    for (int i = 0; i < ml->count; i++) {
        make_move(s, ml->moves[i]);
        /* After the make, s->side is the opponent: the mover's king must not
         * be attacked by whoever is now on the move. */
        if (!is_attacked(s, king_sq(s, us), s->side))
            ml->moves[n++] = ml->moves[i];
        unmake_move(s, ml->moves[i]);
    }
    ml->count = n;
}

/* ------------------------------------------------------------------ */
/* UCI move strings ("e2e4", "e7e8q"; castling = king from/to)          */
/* ------------------------------------------------------------------ */

void move_to_uci(Move m, char out[8]) {
    static const char promo_char[4] = {'n', 'b', 'r', 'q'};
    int i = 0;
    out[i++] = (char)('a' + (move_from(m) & 7));
    out[i++] = (char)('1' + (move_from(m) >> 3));
    out[i++] = (char)('a' + (move_to(m) & 7));
    out[i++] = (char)('1' + (move_to(m) >> 3));
    if (move_is_promo(m)) out[i++] = promo_char[move_flags(m) & 3];
    out[i] = '\0';
}
