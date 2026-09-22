/*
 * board.c — state manipulation: attack detection, make/unmake.
 *
 * make/unmake (rather than copy-make) is the classic bitboard approach: making
 * a move records a 6-byte Undo record, and unmake restores exactly what the
 * move touched. Perft and the legality filter do millions of make/unmake
 * pairs; copying the whole State per node would be ~100x more memory traffic.
 */
#include "chess.h"

/* ------------------------------------------------------------------ */
/* Bitboard attack generation                                          */
/* ------------------------------------------------------------------ */

/* One sliding ray from sq in direction (df, dr): every square up to and
 * including the first blocker. */
static uint64_t ray_attacks(int sq, uint64_t occ, int df, int dr) {
    uint64_t r = 0;
    int f = (sq & 7) + df;
    int rk = (sq >> 3) + dr;
    while (f >= 0 && f < 8 && rk >= 0 && rk < 8) {
        int t = rk * 8 + f;
        r |= 1ULL << t;
        if (occ & (1ULL << t)) break;
        f += df;
        rk += dr;
    }
    return r;
}

/* Bishop = 4 diagonal rays, rook = 4 orthogonal rays, queen = both,
 * all stopping at the first blocker (which is included as attacked). */
uint64_t bishop_attacks(int sq, uint64_t occ) {
    return ray_attacks(sq, occ, 1, 1) | ray_attacks(sq, occ, 1, -1) |
           ray_attacks(sq, occ, -1, 1) | ray_attacks(sq, occ, -1, -1);
}
uint64_t rook_attacks(int sq, uint64_t occ) {
    return ray_attacks(sq, occ, 1, 0) | ray_attacks(sq, occ, -1, 0) |
           ray_attacks(sq, occ, 0, 1) | ray_attacks(sq, occ, 0, -1);
}

bool is_attacked(const State *s, int sq, int by) {
    uint64_t occ = occ_all(s);

    /* Pawns: the squares from which a `by`-pawn attacks `sq` are exactly the
     * squares a pawn of the *opposite* color standing on `sq` would attack. */
    if (pawn_att_tbl[by ^ 1][sq] & s->bb[by][PAWN]) return true;
    if (knight_tbl[sq] & s->bb[by][KNIGHT]) return true;
    if (king_tbl[sq] & s->bb[by][KING]) return true;
    if (bishop_attacks(sq, occ) & (s->bb[by][BISHOP] | s->bb[by][QUEEN])) return true;
    if (rook_attacks(sq, occ) & (s->bb[by][ROOK] | s->bb[by][QUEEN])) return true;
    return false;
}

bool in_check(const State *s) {
    return is_attacked(s, king_sq(s, s->side), s->side ^ 1);
}

/* ------------------------------------------------------------------ */
/* Piece add / remove / move (keep mailbox and bitboards in sync)      */
/* ------------------------------------------------------------------ */

static void add_piece(State *s, int color, int type, int sq) {
    assert(s->board[sq] == EMPTY);
    uint64_t b = 1ULL << sq;
    s->bb[color][type] |= b;
    s->occ[color] |= b;
    s->board[sq] = (uint8_t)pc_of(color, type);
}

static void remove_piece(State *s, int color, int type, int sq) {
    assert(s->board[sq] == pc_of(color, type));
    uint64_t b = 1ULL << sq;
    s->bb[color][type] &= ~b;
    s->occ[color] &= ~b;
    s->board[sq] = EMPTY;
}

static void move_piece(State *s, int color, int type, int from, int to) {
    assert(s->board[from] == pc_of(color, type));
    assert(s->board[to] == EMPTY);
    uint64_t b = (1ULL << from) | (1ULL << to);
    s->bb[color][type] ^= b;
    s->occ[color] ^= b;
    s->board[from] = EMPTY;
    s->board[to] = (uint8_t)pc_of(color, type);
}

/* ------------------------------------------------------------------ */
/* make / unmake                                                       */
/* ------------------------------------------------------------------ */

void make_move(State *s, Move m) {
    const int from  = move_from(m);
    const int to    = move_to(m);
    const int flags = move_flags(m);
    const int us    = s->side;
    const int them  = us ^ 1;
    const int pc    = s->board[from];

    /* Captured piece: for en passant the victim is not on `to` but beside it. */
    const int cap = (flags == F_EP) ? pc_of(them, PAWN) : s->board[to];

    assert(s->ply < MAX_PLY);
    assert(pc != EMPTY);
    s->hist[s->ply].captured = (uint8_t)cap;
    s->hist[s->ply].castling = s->castling;
    s->hist[s->ply].ep       = s->ep;
    s->hist[s->ply].halfmove = s->halfmove;
    s->ply++;

    s->ep = NO_SQ; /* set again below for double pushes */

    if (flags == F_KCASTLE || flags == F_QCASTLE) {
        /* King slides two squares; the rook jumps to its new home. */
        int rfrom, rto;
        if (flags == F_KCASTLE) {
            rfrom = us == WHITE ? H1 : H8;
            rto   = us == WHITE ? F1 : F8;
        } else {
            rfrom = us == WHITE ? A1 : A8;
            rto   = us == WHITE ? D1 : D8;
        }
        move_piece(s, us, KING, from, to);
        move_piece(s, us, ROOK, rfrom, rto);
    } else if (flags == F_EP) {
        remove_piece(s, them, PAWN, us == WHITE ? to - 8 : to + 8);
        move_piece(s, us, PAWN, from, to);
    } else if (flags >= F_PROMO_N) {
        if (cap != EMPTY) remove_piece(s, them, pc_type(cap), to);
        remove_piece(s, us, PAWN, from); /* promotions are always pawn moves */
        add_piece(s, us, move_promo_type(m), to);
    } else {
        if (cap != EMPTY) remove_piece(s, them, pc_type(cap), to);
        move_piece(s, us, pc_type(pc), from, to);
    }

    /* Castling rights: moving from/to any king/rook home square kills the
     * corresponding rights (the mask makes rook *captures* work too). */
    s->castling &= (uint8_t)(castle_mask[from] & castle_mask[to]);

    if (flags == F_DOUBLE_PUSH)
        s->ep = (uint8_t)((from + to) / 2); /* the square jumped over */

    /* 50-move-rule clock resets on any pawn move or any capture. */
    s->halfmove = (pc_type(pc) == PAWN || cap != EMPTY) ? 0 : (uint16_t)(s->halfmove + 1);

    if (us == BLACK) s->fullmove++;
    s->side = (uint8_t)them;
}

void unmake_move(State *s, Move m) {
    const int from  = move_from(m);
    const int to    = move_to(m);
    const int flags = move_flags(m);

    s->side ^= 1;                 /* restore the mover */
    const int us   = s->side;
    const int them = us ^ 1;

    s->ply--;
    const Undo u = s->hist[s->ply];

    if (flags == F_KCASTLE || flags == F_QCASTLE) {
        int rfrom, rto;
        if (flags == F_KCASTLE) {
            rfrom = us == WHITE ? H1 : H8;
            rto   = us == WHITE ? F1 : F8;
        } else {
            rfrom = us == WHITE ? A1 : A8;
            rto   = us == WHITE ? D1 : D8;
        }
        move_piece(s, us, KING, to, from);
        move_piece(s, us, ROOK, rto, rfrom);
    } else if (flags == F_EP) {
        move_piece(s, us, PAWN, to, from);
        add_piece(s, them, PAWN, us == WHITE ? to - 8 : to + 8);
    } else if (flags >= F_PROMO_N) {
        remove_piece(s, us, move_promo_type(m), to);
        add_piece(s, us, PAWN, from);
        if (u.captured != EMPTY) add_piece(s, them, pc_type(u.captured), to);
    } else {
        move_piece(s, us, pc_type(s->board[to]), to, from);
        if (u.captured != EMPTY) add_piece(s, them, pc_type(u.captured), to);
    }

    s->castling = u.castling;
    s->ep       = u.ep;
    s->halfmove = u.halfmove;
    if (us == BLACK) s->fullmove--;
}
