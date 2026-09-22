/*
 * tables.c — read-only lookup tables, filled once by chess_init().
 *
 * These are the engine's only globals, and they are write-once: after
 * chess_init() returns, nothing ever writes them again, so any number of
 * threads and State instances can share them (the "no globals" rule in
 * PLAN.md concerns per-game mutable state, which all lives in State).
 *
 * Kept deliberately simple: knight/king/pawn attack sets are precomputed by
 * file/rank arithmetic, and sliders (bishop/rook/queen) use classic ray
 * scanning over the occupancy — no magic bitboards. Correctness is the Phase 1
 * gate; ray scanning is plenty fast for perft depth 5–6 and will only be
 * revisited if later profiling of the self-play loop demands it.
 */
#include "chess.h"

uint64_t knight_tbl[64];      /* squares a knight on sq attacks            */
uint64_t king_tbl[64];        /* squares a king on sq attacks              */
uint64_t pawn_att_tbl[2][64]; /* squares a pawn of color c on sq attacks   */
uint64_t castle_mask[64];     /* castling-rights bits that SURVIVE a move
                                 touching sq (all others are cleared)      */

#define BB(sq) (1ULL << (sq))

static void init_leaper_tables(void) {
    /* (file delta, rank delta) moves, validated against the 8x8 board. */
    static const int knight_d[8][2] = {
        {1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}
    };
    static const int king_d[8][2] = {
        {1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}
    };

    for (int sq = 0; sq < 64; sq++) {
        int f = sq & 7, r = sq >> 3;
        knight_tbl[sq] = 0;
        king_tbl[sq] = 0;
        pawn_att_tbl[WHITE][sq] = 0;
        pawn_att_tbl[BLACK][sq] = 0;

        for (int i = 0; i < 8; i++) {
            int nf = f + knight_d[i][0], nr = r + knight_d[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                knight_tbl[sq] |= BB(nr * 8 + nf);

            nf = f + king_d[i][0];
            nr = r + king_d[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8)
                king_tbl[sq] |= BB(nr * 8 + nf);
        }

        /* A white pawn on sq attacks sq+7 (file-1) and sq+9 (file+1);
         * a black pawn attacks sq-9 and sq-7. Both file and rank bounds are
         * checked. (Pawns never stand on their promotion rank in a reachable
         * position, but the table stays correct there anyway.) */
        if (r < 7) {
            if (f > 0) pawn_att_tbl[WHITE][sq] |= BB(sq + 7);
            if (f < 7) pawn_att_tbl[WHITE][sq] |= BB(sq + 9);
        }
        if (r > 0) {
            if (f > 0) pawn_att_tbl[BLACK][sq] |= BB(sq - 9);
            if (f < 7) pawn_att_tbl[BLACK][sq] |= BB(sq - 7);
        }
    }
}

static void init_castle_masks(void) {
    /* Default: touching a square keeps every right. Then remove rights that
     * die when the relevant king/rook home square is moved *from* or *to* —
     * "to" covers rook captures on those squares (e.g. a black bishop taking
     * the h1 rook kills white's O-O). */
    for (int sq = 0; sq < 64; sq++)
        castle_mask[sq] = 0xF;
    castle_mask[E1] &= ~(uint8_t)(CASTLE_WK | CASTLE_WQ);
    castle_mask[H1] &= ~(uint8_t)CASTLE_WK;
    castle_mask[A1] &= ~(uint8_t)CASTLE_WQ;
    castle_mask[E8] &= ~(uint8_t)(CASTLE_BK | CASTLE_BQ);
    castle_mask[H8] &= ~(uint8_t)CASTLE_BK;
    castle_mask[A8] &= ~(uint8_t)CASTLE_BQ;
}

void chess_init(void) {
    init_leaper_tables();
    init_castle_masks();
}
