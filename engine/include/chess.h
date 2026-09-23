/*
 * chess.h — public types and API of the C rules engine (Phase 1).
 *
 * DESIGN OVERVIEW (read this first)
 * ================================
 * Board representation: **bitboards** — one 64-bit word per (color, piece type).
 * Square mapping is "little-endian rank-file" (same as Stockfish):
 *     square = rank * 8 + file,  a1 = 0, h1 = 7, a8 = 56, h8 = 63.
 * White pawns therefore move in the +8 direction, black pawns in -8.
 *
 * A second, redundant view is kept in parallel: `board[64]`, a mailbox of piece
 * codes. Bitboards answer "where are the knights?" in one AND; the mailbox
 * answers "what's on e4?" in one load. Move make/unmake updates both, and the
 * tests assert they never disagree.
 *
 * Zero dynamic memory: `State` is one fixed-size struct = one game. Move lists
 * live on the caller's stack; undo information lives in a fixed array inside
 * the State. No malloc anywhere, so hundreds of concurrent game instances can
 * live in one process (the self-play design) without heap contention.
 *
 * No *mutable* globals: all per-game state lives in State. The only globals are
 * read-only lookup tables filled once by chess_init() (knight/king/pawn attack
 * tables, castling-rights masks); after initialization they are never written,
 * so any number of threads/instances can share them safely.
 *
 * Legal move generation is strictly-legal ("pin-aware"), the pattern used
 * by strong engines: gen_legal() computes the pieces checking our king,
 * the pieces pinned to it, and the enemy's full attack map once, then
 * emits only moves that are legal by construction:
 *   - in single check, non-king moves must capture the checker or block
 *     its ray (a target mask); in double check only king moves exist;
 *   - a pinned piece may only move along its pin line;
 *   - king moves may not enter an enemy-attacked square (computed with the
 *     king removed from occupancy, so retreating along a checking slider's
 *     ray is still seen as attacked);
 *   - castling keeps its generation-time validation (path empty, not out
 *     of / through / into check);
 *   - en passant — the one move where two pieces leave the board at once —
 *     is verified explicitly against the post-capture occupancy, which
 *     covers the notorious ep-pin cases with zero special-casing.
 * gen_pseudo_legal() (pseudo-legal + make/unmake filter) is retained as a
 * test oracle: the unit tests cross-check the two generators against each
 * other over thousands of random positions.
 */
#ifndef CHESS_H
#define CHESS_H

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/* Basic vocabulary                                                    */
/* ------------------------------------------------------------------ */

enum { WHITE = 0, BLACK = 1 };

enum { PAWN = 0, KNIGHT, BISHOP, ROOK, QUEEN, KING, PIECE_TYPE_COUNT };

/* Piece code stored in the mailbox: color * 6 + type (0..11); EMPTY = 12. */
enum { EMPTY = 12 };

/* Castling-right bits, combinable into the 4-bit mask. */
enum {
    CASTLE_WK = 1,  /* white O-O   */
    CASTLE_WQ = 2,  /* white O-O-O */
    CASTLE_BK = 4,  /* black O-O   */
    CASTLE_BQ = 8,  /* black O-O-O */
};

/* All 64 named squares (a1 = 0 … h8 = 63), for readable tests and move code. */
enum {
    A1 = 0,  B1,  C1,  D1,  E1,  F1,  G1,  H1,
    A2,      B2,  C2,  D2,  E2,  F2,  G2,  H2,
    A3,      B3,  C3,  D3,  E3,  F3,  G3,  H3,
    A4,      B4,  C4,  D4,  E4,  F4,  G4,  H4,
    A5,      B5,  C5,  D5,  E5,  F5,  G5,  H5,
    A6,      B6,  C6,  D6,  E6,  F6,  G6,  H6,
    A7,      B7,  C7,  D7,  E7,  F7,  G7,  H7,
    A8 = 56, B8,  C8,  D8,  E8,  F8,  G8,  H8
};

#define NO_SQ    64    /* sentinel: "no en-passant target square"   */
#define MAX_PLY  1024  /* fixed undo-stack depth inside State       */
#define MAX_MOVES 256  /* legal maximum in any chess position is 218 */

#define CHESS_START_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

/* ------------------------------------------------------------------ */
/* Move: 16 bits — [flags:4][to:6][from:6]                             */
/* ------------------------------------------------------------------ */
/*
 * The 4-bit flag says what kind of move it is. It is needed to (a) undo
 * special moves correctly, (b) tell captures from quiets for move ordering
 * later, and (c) encode promotions. Flag layout follows the scheme used by
 * most engines (e.g. Stockfish):
 *
 *   0 quiet          1 double pawn push     2 king-side castle   3 queen-side castle
 *   4 capture        5 en-passant capture
 *   8..11  promotion to N/B/R/Q (non-capture)
 *   12..15 promotion to N/B/R/Q (capture)
 *
 * Promotion piece type is recoverable as KNIGHT + (flag & 3).
 */
typedef uint16_t Move;

enum {
    F_QUIET = 0,
    F_DOUBLE_PUSH,
    F_KCASTLE,
    F_QCASTLE,
    F_CAPTURE,
    F_EP,
    /* 6,7 unused */
    F_PROMO_N = 8,
    F_PROMO_B,
    F_PROMO_R,
    F_PROMO_Q,
    F_PROMO_CAP_N = 12,
    F_PROMO_CAP_B,
    F_PROMO_CAP_R,
    F_PROMO_CAP_Q
};

static inline Move move_make(int from, int to, int flags) {
    assert(from >= 0 && from < 64 && to >= 0 && to < 64 && flags >= 0 && flags < 16);
    return (Move)((flags << 12) | (to << 6) | from);
}
static inline int  move_from (Move m) { return m & 63; }
static inline int  move_to   (Move m) { return (m >> 6) & 63; }
static inline int  move_flags(Move m) { return m >> 12; }
static inline bool move_is_capture(Move m) {
    int f = move_flags(m);
    return f == F_CAPTURE || f == F_EP || f >= F_PROMO_CAP_N;
}
static inline bool move_is_promo(Move m) { return move_flags(m) >= F_PROMO_N; }
static inline int  move_promo_type(Move m) { return KNIGHT + (move_flags(m) & 3); }

/* ------------------------------------------------------------------ */
/* State: one whole game, zero heap                                    */
/* ------------------------------------------------------------------ */

/* What unmake_move() needs to know about one ply. */
typedef struct {
    uint8_t  captured;   /* piece code taken on this ply, EMPTY if none       */
    uint8_t  castling;   /* castling rights *before* the move                 */
    uint8_t  ep;         /* en-passant target *before* the move (or NO_SQ)    */
    uint16_t halfmove;   /* halfmove clock *before* the move                  */
    uint64_t key;        /* position key *before* the move (see State.keys)   */
} Undo;

typedef struct {
    /* Position: two redundant views, kept in sync by make/unmake. */
    uint8_t  board[64];       /* mailbox: piece code or EMPTY               */
    uint64_t bb[2][6];        /* bitboard per (color, piece type)           */
    uint64_t occ[2];          /* occupancy per color (OR of its 6 piece bbs) */

    /* Game clock / rights. */
    uint8_t  side;            /* side to move, WHITE or BLACK               */
    uint8_t  castling;        /* 4-bit CASTLE_* mask                        */
    uint8_t  ep;              /* en-passant target square, or NO_SQ         */
    uint16_t halfmove;        /* plies since last pawn move or capture      */
    uint32_t fullmove;        /* FEN fullmove number (starts at 1)          */

    /* Fixed undo stack (make increments ply, unmake decrements). */
    uint16_t ply;
    Undo     hist[MAX_PLY];

    /* Position-key history (Phase 1.3: threefold repetition detection).
     *
     * keys[i] is the Zobrist key of the position after i plies, so keys[ply]
     * is always the key of the *current* position. The key hashes exactly
     * python-chess's notion of "same position": piece placement, side to move,
     * castling rights — and deliberately NOT the en-passant square and NOT
     * the clocks. En passant is excluded from the hash because FIDE (and
     * python-chess) only count two positions as the same if en passant is
     * actually *capturable* — a legality question, too expensive to answer
     * inside make_move's hot path. state_is_repetition() refines raw key
     * matches with an exact en-passant-legality comparison instead (it can
     * afford make/unmake probes at query time). The same keys become the
     * transposition-table hash of the search later. */
    uint64_t keys[MAX_PLY + 1];
} State;

typedef struct {
    Move moves[MAX_MOVES];
    int  count;
} MoveList;

/* ------------------------------------------------------------------ */
/* Piece-code helpers                                                  */
/* ------------------------------------------------------------------ */

static inline int  pc_of(int color, int type) { return color * 6 + type; }
static inline int  pc_color(int pc)           { return pc / 6; }
static inline int  pc_type(int pc)            { return pc % 6; }
static inline uint64_t occ_all(const State *s){ return s->occ[WHITE] | s->occ[BLACK]; }
static inline int  king_sq(const State *s, int color) { return __builtin_ctzll(s->bb[color][KING]); }

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

/* Fill the read-only lookup tables. Call once per process before anything
 * else. (Tables are never written afterwards, so this is thread-safe to
 * *share*; just don't call chess_init concurrently with chess_* use.) */
void chess_init(void);

/* Read-only lookup tables (see tables.c; valid after chess_init()). */
extern uint64_t knight_tbl[64];
extern uint64_t king_tbl[64];
extern uint64_t pawn_att_tbl[2][64];
extern uint64_t castle_mask[64];

/* "Between" and "full line" squares, filled by chess_init():
 *   between_bb[a][b] — squares strictly between a and b (0 if not aligned)
 *   line_bb[a][b]    — every square of the rank/file/diagonal through a and b
 *                      incl. a and b themselves (0 if not aligned)
 * Used by the strictly-legal move generator: check-blocking targets and the
 * pin rays a pinned piece may not leave. */
extern uint64_t between_bb[64][64];
extern uint64_t line_bb[64][64];

/* Zobrist key material (filled by chess_init(), then never written):
 * zob_psq[piece code][square], zob_castle[4-bit rights mask], and a single
 * side-to-move key (XORed in when the side to move is BLACK). */
extern uint64_t zob_psq[12][64];
extern uint64_t zob_castle[16];
extern uint64_t zob_side;

/* Position key of `s` (== s->keys[s->ply]): placement + side + castling,
 * en passant and clocks excluded (see State.keys). */
uint64_t state_poskey(const State *s);

/* ------------------------------------------------------------------ */
/* Slider attacks: magic bitboards                                     */
/* ------------------------------------------------------------------ */
/*
 * A slider's attack set depends only on (square, relevant occupancy) — the
 * squares of the four rays up to (excluding) the board edge. Magic bitboards
 * hash that subset into a precomputed table entry with one multiply+shift:
 *
 *     attacks = tbl[sq][((occ & mask[sq]) * magic[sq]) >> shift[sq]]
 *
 * Plain portable C — a 64×64 multiply and two table loads, no architecture
 * specifics (no PEXT, no hand-tuned asm) — the same fast path on arm64 and
 * x86_64. Semantics are identical to the old ray scanner: every square up to
 * and INCLUDING the first blocker (the blocker is a capture target). Queen =
 * bishop_attacks | rook_attacks.
 *
 * The per-square descriptors below are write-once globals filled by
 * chess_init() (tables.c generates and self-verifies the tables; ~800 KB).
 */
typedef struct {
    uint64_t  mask;     /* relevant-occupancy mask (ray squares, edge-stripped) */
    uint64_t  magic;    /* the shift multiplier found at init                   */
    uint64_t *attacks;  /* this square's slice of the shared attack table       */
    unsigned  shift;    /* 64 - popcount(mask)                                  */
} SliderMagic;

extern SliderMagic bishop_magic[64];
extern SliderMagic rook_magic[64];

static inline uint64_t bishop_attacks(int sq, uint64_t occ) {
    const SliderMagic *m = &bishop_magic[sq];
    return m->attacks[((occ & m->mask) * m->magic) >> m->shift];
}
static inline uint64_t rook_attacks(int sq, uint64_t occ) {
    const SliderMagic *m = &rook_magic[sq];
    return m->attacks[((occ & m->mask) * m->magic) >> m->shift];
}

/* --- FEN ------------------------------------------------------------ */
int  fen_parse(State *s, const char *fen);          /* 0 = ok, -1 = bad FEN */
/* Serialize into `out`; n must be >= 96 (max FEN length is ~90 chars). */
void fen_serialize(const State *s, char *out, size_t n);
void state_init_startpos(State *s);                 /* convenience wrapper  */

/* --- make / unmake --------------------------------------------------- */
void make_move(State *s, Move m);
void unmake_move(State *s, Move m);

/* --- attacks ---------------------------------------------------------- */
/* Is square `sq` attacked by any piece of color `by`? */
bool is_attacked(const State *s, int sq, int by);
/* Is the side to move's king in check? */
bool in_check(const State *s);

/* --- move generation -------------------------------------------------- */
/* Pseudo-legal moves (obey piece movement, ignore own-king safety except
 * for castling's at-generation rules). Retained as a test oracle and
 * debugging aid. Pure function of the position. */
void gen_pseudo_legal(const State *s, MoveList *ml);
/* Strictly-legal generation: only legal moves, emitted directly (see the
 * file header). Does NOT mutate `s` — safe to call on a shared State. */
void gen_legal(const State *s, MoveList *ml);
/* Same legality logic, reduced to a count (popcounts instead of move
 * emission) — the perft bulk-counting fast path. Cross-checked against
 * gen_legal over thousands of random positions in the unit tests. */
int gen_legal_count(const State *s);

/* --- move ⇄ string (UCI algebraic, e.g. "e2e4", "e7e8q", castling "e1g1") --- */
void move_to_uci(Move m, char out[8]);

/* --- perft ------------------------------------------------------------- */
/* Leaf nodes at `depth`; bulk-counts the last ply via legal move counts.
 * Like gen_legal, mutates `s` temporarily (restored before returning). */
uint64_t perft(State *s, int depth);
/* Prints one line per legal root move with its subtree count (debugging). */
void perft_divide(State *s, int depth);

/* --- termination (Phase 1.3) --------------------------------------------
 *
 * The individual predicates follow python-chess exactly (it is our fuzz
 * oracle, PLAN §10.3), and state_termination() composes them with the same
 * precedence python-chess's Board.outcome(claim_draw=True) uses. Modern FIDE
 * rules have two draw tiers: *claimable* (threefold, 50-move — a player must
 * claim) and *automatic* (fivefold, 75-move — the game ends regardless). A
 * referee would distinguish them; self-play auto-adjudicates at the claimable
 * threshold, so a game never lives long enough to reach the automatic one.
 */
enum {
    TERM_PLAYING = 0,
    TERM_CHECKMATE,             /* winner: the side NOT to move              */
    TERM_INSUFFICIENT_MATERIAL, /* dead material: KvK, KN/KBvK, same-color Bs*/
    TERM_STALEMATE,
    TERM_SEVENTYFIVE_MOVES,     /* halfmove clock >= 150 (automatic draw)     */
    TERM_FIVEFOLD_REPETITION,
    TERM_FIFTY_MOVES,           /* halfmove clock >= 100 (claimable draw)     */
    TERM_THREEFOLD_REPETITION,  /* current position occurred >= 3 times       */
};

/* No legal moves + king in check. */
bool state_is_checkmate(State *s);
/* No legal moves + king NOT in check. */
bool state_is_stalemate(State *s);
/* Neither side can force mate with the material on the board (FIDE "dead
 * material" — exactly python-chess's is_insufficient_material). */
bool state_insufficient_material(const State *s);
/* Halfmove clock >= n and at least one legal move exists (python-chess's
 * _is_halfmoves; use n=100 for the claimable rule, 150 for the automatic). */
bool state_halfmoves(State *s, int n);
/* Has the CURRENT position occurred `count` times (including now)? Uses the
 * key history + exact FIDE en-passant refinement; only scans back within the
 * halfmove-clock window, since a pawn move or capture can never repeat.
 * Like gen_legal, temporarily mutates `s` (restored before returning). */
bool state_is_repetition(State *s, int count);

/* First applicable reason the game is over, or TERM_PLAYING. Precedence and
 * semantics mirror python-chess Board.outcome(claim_draw=True):
 *   checkmate > insufficient material > stalemate > 75 moves > fivefold
 *   > 50 moves (claimable) > threefold (claimable).
 * The claimable legs include their one-move lookahead (a draw can be
 * claimed by the intended move — see termination.c). Temporarily mutates
 * `s` (restored before returning). */
int state_termination(State *s);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_H */
