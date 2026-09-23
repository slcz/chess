/*
 * tables.c — read-only lookup tables, filled once by chess_init().
 *
 * These are the engine's only globals, and they are write-once: after
 * chess_init() returns, nothing ever writes them again, so any number of
 * threads and State instances can share them (the "no globals" rule in
 * PLAN.md concerns per-game mutable state, which all lives in State).
 *
 * Contents:
 *   - leaper attack tables (knight / king / pawns) by file/rank arithmetic
 *   - castling-rights masks
 *   - Zobrist key material (fixed seed, reproducible everywhere)
 *   - between/line square tables for the strictly-legal move generator
 *   - magic bitboards for bishop/rook slider attacks (subtask 1.6):
 *     the classic ray scanner walked 4 rays square-by-square on every query;
 *     a magic lookup answers in one mask + multiply + shift + load. Plain
 *     portable C11 — deliberately no PEXT / CRC / inline asm, so the same
 *     code is the fast path on both arm64 (dev Mac) and x86_64 (cloud).
 *     The magic multipliers were found once by a deterministic fixed-seed
 *     search and are baked in below; every chess_init() re-verifies the
 *     built tables exhaustively against the ray scanner (and re-searches
 *     any square whose constant fails), so a bad constant can never ship
 *     silently.
 */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "chess.h"

uint64_t knight_tbl[64];      /* squares a knight on sq attacks            */
uint64_t king_tbl[64];        /* squares a king on sq attacks              */
uint64_t pawn_att_tbl[2][64]; /* squares a pawn of color c on sq attacks   */
uint64_t castle_mask[64];     /* castling-rights bits that SURVIVE a move
                                 touching sq (all others are cleared)      */

uint64_t between_bb[64][64];  /* squares strictly between a and b (0 if unaligned) */
uint64_t line_bb[64][64];     /* full rank/file/diagonal through a and b (0 if unaligned) */

uint64_t zob_psq[12][64];     /* Zobrist: piece code on a square            */
uint64_t zob_castle[16];      /* Zobrist: 4-bit castling-rights mask        */
uint64_t zob_side;            /* Zobrist: side to move (XOR in for BLACK)   */

#define BB(sq) (1ULL << (sq))

/* ------------------------------------------------------------------ */
/* Init-time ray walking (also the oracle the magic tables are         */
/* verified against — the classic scanner, kept simple on purpose)     */
/* ------------------------------------------------------------------ */

/* One sliding ray from sq towards (df, dr): every square up to and including
 * the first blocker. */
static uint64_t ray_walk(int sq, uint64_t occ, int df, int dr) {
    uint64_t r = 0;
    int f = (sq & 7) + df;
    int rk = (sq >> 3) + dr;
    while (f >= 0 && f < 8 && rk >= 0 && rk < 8) {
        int t = rk * 8 + f;
        r |= BB(t);
        if (occ & BB(t)) break;
        f += df;
        rk += dr;
    }
    return r;
}

static const int bishop_dirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
static const int rook_dirs[4][2]   = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};

static uint64_t ray_attacks(int sq, uint64_t occ, const int dirs[4][2]) {
    uint64_t r = 0;
    for (int i = 0; i < 4; i++) r |= ray_walk(sq, occ, dirs[i][0], dirs[i][1]);
    return r;
}

static uint64_t bishop_ray_attacks(int sq, uint64_t occ) {
    return ray_attacks(sq, occ, bishop_dirs);
}
static uint64_t rook_ray_attacks(int sq, uint64_t occ) {
    return ray_attacks(sq, occ, rook_dirs);
}

/* ------------------------------------------------------------------ */
/* Leaper tables                                                       */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* Between / line tables (for the strictly-legal generator)            */
/* ------------------------------------------------------------------ */

static void init_between_line(void) {
    static const int all_dirs[8][2] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}
    };
    for (int a = 0; a < 64; a++) {
        for (int b = 0; b < 64; b++) {
            between_bb[a][b] = 0;
            line_bb[a][b] = 0;
        }
        for (int d = 0; d < 8; d++) {
            const int df = all_dirs[d][0], dr = all_dirs[d][1];
            /* Full line through a in this direction and its opposite. */
            uint64_t line = ray_walk(a, 0, df, dr) | ray_walk(a, 0, -df, -dr) | BB(a);
            /* Walk the direction, labeling every reached square b with the
             * line and the squares strictly between. */
            uint64_t between = 0;
            int f = (a & 7) + df, rk = (a >> 3) + dr;
            while (f >= 0 && f < 8 && rk >= 0 && rk < 8) {
                int b = rk * 8 + f;
                line_bb[a][b] = line;
                between_bb[a][b] = between;
                between |= BB(b);
                f += df;
                rk += dr;
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Zobrist                                                             */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* Magic bitboards                                                     */
/* ------------------------------------------------------------------ */
/*
 * For each slider square we precompute, for every subset of its
 * "relevant occupancy" mask (the ray squares short of the board edge —
 * blockers on the final square of a ray never change the attack set), the
 * attack bitboard. A per-square magic multiplier maps
 * (occ & mask) -> a collision-free table index via the high bits of the
 * product. Table sizes: rook 212,992 entries (~1.7 MB), bishop 5,248
 * (~41 KB) — comfortably cacheable on modern desktop/server CPUs.
 */

SliderMagic bishop_magic[64];
SliderMagic rook_magic[64];

/* Table storage: the sums of 2^relevant-bits over all squares for each
 * slider (rook: 4 corners x 4096 + 24 edges x 2048 + 36 inner x 1024 =
 * 102400 entries = 800 KB; bishop: 5248 = 41 KB). */
static uint64_t bishop_att_tbl[5248];
static uint64_t rook_att_tbl[102400];

/* Magic multipliers found by our own deterministic search (the same code
 * as the runtime fallback below), frozen into the source so chess_init()
 * stays fast. Each one is re-verified exhaustively at every init — a bad
 * constant triggers the runtime re-search instead of misbehaving. */
static const uint64_t bishop_magic_const[64] = {
    0x8004201806002840ULL,     0x0004010414048000ULL,     0x580401022a000201ULL,     0x00280a0226140100ULL,
    0x2004504000028008ULL,     0x0400882009860000ULL,     0x1101081510281004ULL,     0x0000820050020881ULL,
    0x0820c40888090400ULL,     0x000042045404084aULL,     0x0121100408a02408ULL,     0x0040422082001080ULL,
    0x008004030800000aULL,     0xb000008210400088ULL,     0x8110010410040402ULL,     0x0008009088013000ULL,
    0x1840002828882080ULL,     0x0020080282040104ULL,     0xa2501408008a1010ULL,     0x2480801808210000ULL,
    0x4841050820080000ULL,     0x4004898a00842008ULL,     0xa011108208010500ULL,     0x0010400482081124ULL,
    0x1005402120285100ULL,     0x8010112204410204ULL,     0x9108020420408904ULL,     0x001a080010081020ULL,
    0x5a81001001004002ULL,     0x00900280040a0104ULL,     0x0002020100480201ULL,     0x0000830000844800ULL,
    0x2401041000206001ULL,     0x0008080400088120ULL,     0x0204020100080050ULL,     0x0000040401080120ULL,
    0x2044004200440090ULL,     0x00e00101002a1040ULL,     0x0901010402010444ULL,     0x0048004281411088ULL,
    0x0094016090000830ULL,     0x80040088080004b4ULL,     0x0202001402004c10ULL,     0x0b04122024200800ULL,
    0x0440140104026210ULL,     0x3a04810042000100ULL,     0x0c03040400802400ULL,     0x0e61021282150105ULL,
    0x1020410420200200ULL,     0x0241006814040810ULL,     0x0008010888244004ULL,     0x0204488020880110ULL,
    0x1810025021024001ULL,     0xc1009c4810850800ULL,     0x0004081014088000ULL,     0x0820024882098080ULL,
    0x0209124110282022ULL,     0x0000020049182801ULL,     0x0400000080480800ULL,     0x2000008000840408ULL,
    0xc130000021042406ULL,     0x0012004002141108ULL,     0x00000560040c0484ULL,     0x0008825004082080ULL,
};
static const uint64_t rook_magic_const[64] = {
    0x0080008020104000ULL,     0x2a4002100020024aULL,     0x0880100280200108ULL,     0x2080041000080180ULL,
    0x0600242010020068ULL,     0x0200041802007110ULL,     0x040000c401021008ULL,     0x0100038a22084100ULL,
    0x0020800030804004ULL,     0x8042802000804000ULL,     0x4002004020120080ULL,     0x0c40801000080080ULL,
    0x2840800800040080ULL,     0x01d2000508020010ULL,     0x1511000442004100ULL,     0x0002000188640205ULL,
    0x0080004000402000ULL,     0x0100414010002002ULL,     0x001b010042200014ULL,     0x0408008080081000ULL,
    0x4018010008110005ULL,     0x1024008002000480ULL,     0x10080c0002102811ULL,     0x4000220001004084ULL,
    0x0820802080004001ULL,     0x0010024140002000ULL,     0x8004100880200080ULL,     0x0010000b00210090ULL,
    0x4010040080080080ULL,     0x0284000202000810ULL,     0x0000010080800200ULL,     0x0101000100004082ULL,
    0x0080400090800020ULL,     0x0080804002802008ULL,     0x0040200288801000ULL,     0x8101020823003000ULL,
    0x0000280081800400ULL,     0x002a800400800200ULL,     0x0000800100800200ULL,     0x0413002041000082ULL,
    0x1200400080028028ULL,     0x00a0200050004008ULL,     0x18b1002000410018ULL,     0x8400080010008080ULL,
    0x0200040008008080ULL,     0x1002000408020010ULL,     0x04019208d0140001ULL,     0x01000c00d2820025ULL,
    0x0080004088210100ULL,     0x04c00020015008c0ULL,     0xc880104022008200ULL,     0x1004201003000900ULL,
    0x0028018008040180ULL,     0xa421000208040100ULL,     0x0482004144480200ULL,     0x0000040060950200ULL,
    0x0000800425001041ULL,     0x2100211481400301ULL,     0x0006001040088422ULL,     0x41049000150020c9ULL,
    0x4401003800100215ULL,     0x0022000104100802ULL,     0x110280811022180cULL,     0x0000038024011042ULL,
};

/* Relevant-occupancy mask: the union of the four rays with each ray's final
 * (board-edge) square removed — a blocker there is indistinguishable from
 * an empty ray beyond it. */
static uint64_t sliding_mask(int sq, const int dirs[4][2]) {
    uint64_t m = 0;
    for (int i = 0; i < 4; i++) {
        const int df = dirs[i][0], dr = dirs[i][1];
        int f = (sq & 7) + df, rk = (sq >> 3) + dr;
        int last = -1;
        while (f >= 0 && f < 8 && rk >= 0 && rk < 8) {
            last = rk * 8 + f;
            f += df;
            rk += dr;
        }
        if (last >= 0) m |= ray_walk(sq, 0, df, dr) & ~BB(last);
    }
    return m;
}

/* Scratch for one square's subset/reference tables (max 4096 = 2^12). */
static uint64_t m_occs[4096], m_refs[4096];
static int      m_idx[4096];

/* Enumerate every relevant-occupancy subset of `sq` (carry-rippler) with
 * the ray scanner's answer for each. Returns 2^popcount(mask). */
static int enumerate_subsets(int sq, uint64_t mask, const int dirs[4][2]) {
    int count = 0;
    uint64_t subset = 0;
    do {
        m_occs[count] = subset;
        m_refs[count] = ray_attacks(sq, subset, dirs);
        count++;
        subset = (subset - mask) & mask; /* next subset */
    } while (subset);
    return count;
}

/* Try one specific magic for the pre-enumerated square: fill `slice`,
 * succeed iff no index receives two different attack sets (constructive
 * collisions — same set, same index — are fine). */
static bool try_magic(int bits, uint64_t magic, uint64_t *slice) {
    const int n = 1 << bits;
    for (int j = 0; j < n; j++) {
        m_idx[j] = -1;
        slice[j] = 0;
    }
    for (int i = 0; i < n; i++) {
        unsigned j = (unsigned)((m_occs[i] * magic) >> (64 - bits));
        if (m_idx[j] < 0) {
            m_idx[j] = i;
            slice[j] = m_refs[i];
        } else if (m_refs[m_idx[j]] != m_refs[i]) {
            return false;
        }
    }
    return true;
}

/* Runtime fallback (only if a baked constant fails verification): search
 * for a magic with sparse random candidates. Deterministic via the shared
 * fixed-seed PRNG. */
static uint64_t search_magic(uint64_t mask, int bits, uint64_t *slice,
                             uint64_t rng[1]) {
    for (long attempt = 0; attempt < 20000000L; attempt++) {
        uint64_t magic = splitmix64(rng) & splitmix64(rng) & splitmix64(rng);
        if (__builtin_popcountll((mask * magic) >> 56) < 6) continue; /* cheap prefilter */
        if (try_magic(bits, magic, slice)) return magic;
    }
    fprintf(stderr, "magic search failed (bits=%d) — init bug, report this\n", bits);
    abort();
    return 0;
}

/* Fill one slider's descriptors: mask/shift/slice layout, baked magic with
 * verified fallback. */
static void init_slider_sq(SliderMagic *m, int sq, uint64_t magic_const,
                           uint64_t **next, const int dirs[4][2], uint64_t rng[1]) {
    m->mask    = sliding_mask(sq, dirs);
    const int bits = __builtin_popcountll(m->mask);
    m->shift   = (unsigned)(64 - bits);
    m->attacks = *next;

    const int n = enumerate_subsets(sq, m->mask, dirs);
    assert(n == (1 << bits));

    if (try_magic(bits, magic_const, m->attacks)) m->magic = magic_const;
    else m->magic = search_magic(m->mask, bits, m->attacks, rng);

    *next += 1 << bits;
}

static void init_magics(uint64_t rng[1]) {
    uint64_t *bnext = bishop_att_tbl, *rnext = rook_att_tbl;
    for (int sq = 0; sq < 64; sq++)
        init_slider_sq(&bishop_magic[sq], sq, bishop_magic_const[sq], &bnext, bishop_dirs, rng);
    for (int sq = 0; sq < 64; sq++)
        init_slider_sq(&rook_magic[sq],   sq, rook_magic_const[sq],   &rnext, rook_dirs, rng);

    /* Exhaustive self-verification: every relevant occupancy of every square
     * must look up to exactly what the ray scanner computes (including the
     * square-mask complement, i.e. blockers on irrelevant squares too). A
     * silent table corruption cannot pass this — the process aborts. */
    for (int sq = 0; sq < 64; sq++) {
        const SliderMagic *m = &bishop_magic[sq];
        uint64_t subset = 0;
        do {
            uint64_t occ = subset | ~m->mask;
            if (bishop_attacks(sq, occ) != bishop_ray_attacks(sq, occ)) {
                fprintf(stderr, "magic table corrupt: bishop sq=%d occ=%llx "
                                "(init bug — report this)\n", sq,
                        (unsigned long long)occ);
                abort();
            }
            subset = (subset - m->mask) & m->mask;
        } while (subset);
    }
    for (int sq = 0; sq < 64; sq++) {
        const SliderMagic *m = &rook_magic[sq];
        uint64_t subset = 0;
        do {
            uint64_t occ = subset | ~m->mask;
            if (rook_attacks(sq, occ) != rook_ray_attacks(sq, occ)) {
                fprintf(stderr, "magic table corrupt: rook sq=%d occ=%llx "
                                "(init bug — report this)\n", sq,
                        (unsigned long long)occ);
                abort();
            }
            subset = (subset - m->mask) & m->mask;
        } while (subset);
    }
}

/* ------------------------------------------------------------------ */

void chess_init(void) {
    uint64_t rng = 0x1F2E3D4C5B6A7988ULL; /* fixed seed: same magics everywhere */
    init_leaper_tables();
    init_castle_masks();
    init_between_line();
    init_zobrist();
    init_magics(&rng);
}
