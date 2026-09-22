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

uint64_t zob_psq[12][64];     /* Zobrist: piece code on a square            */
uint64_t zob_castle[16];      /* Zobrist: 4-bit castling-rights mask        */
uint64_t zob_side;            /* Zobrist: side to move (XOR in for BLACK)   */

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

/* SplitMix64: a tiny high-quality 64-bit PRNG. Deterministic under our fixed
 * seed, so every process (and every fuzz rerun) derives the same key set. */
static uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static void init_zobrist(void) {
    uint64_t x = 0x0123456789ABCDEFULL; /* fixed seed: reproducible keys */
    for (int pc = 0; pc < 12; pc++)
        for (int sq = 0; sq < 64; sq++)
            zob_psq[pc][sq] = splitmix64(&x);
    for (int m = 0; m < 16; m++)
        zob_castle[m] = splitmix64(&x);
    zob_side = splitmix64(&x);
}

void chess_init(void) {
    init_leaper_tables();
    init_castle_masks();
    init_zobrist();
}
