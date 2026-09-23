/*
 * movegen.c — strictly-legal move generation (rewritten in subtask 1.6).
 *
 * The original Phase 1.2 design generated pseudo-legal moves and filtered
 * them by make / "is our king attacked?" / unmake — simple, obviously
 * correct, and ~1 make-unmake-attack cycle per candidate move per node.
 * This file keeps that simple generator (gen_pseudo_legal, used by tests
 * as a cross-check oracle), but gen_legal itself now emits only legal
 * moves DIRECTLY, the way strong engines do, from three per-node facts:
 *
 *   checkers   enemy pieces currently attacking our king
 *   pinned     our pieces that may not leave the king-pinner line
 *   danger     every square the enemy attacks *with our king removed*
 *              (so a king retreating along a checking slider's ray is
 *              correctly seen as still attacked — the x-ray)
 *
 * From these, legality of every ordinary move is two bit-ANDs:
 *   - while in single check, a non-king move must CAPTURE the checker or
 *     BLOCK its ray: its target must lie in capture_mask
 *     (= checkers | between(king, checker));
 *   - a pinned piece may only move along its pin line line_bb[king, from].
 *   (Both at once is provably impossible: the pin line and the check ray
 *   are two different lines through the king, which meet only at the king
 *   itself — see the comment in gen_legal.)
 *
 * King moves use `danger` instead of a per-square is_attacked probe.
 * Castling keeps its generation-time validation (rights, path empty, not
 * through/into check) — reusing `danger` for the attacked squares.
 *
 * En passant is the one move this scheme cannot cover with masks, because
 * it is the only move where TWO pieces leave the board at once (the pawn
 * moves diagonally, the victim disappears from a different square), which
 * can open a rank/diagonal that no single-line pin analysis sees. EP
 * candidates therefore get an explicit verification: rebuild the occupancy
 * after the capture and ask whether any enemy slider attacks the king.
 * EP is rare, so this costs nothing measurable and needs zero special
 * cases beyond it.
 *
 * gen_legal no longer mutates the State at all — it is now a pure function
 * of the position (the old make/unmake filter temporarily did).
 */
#include "chess.h"

/* Rank/file constants for set-wise pawn generation. */
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
/* Attack queries on explicit occupancy (no State mutation)            */
/* ------------------------------------------------------------------ */

/* All enemy-of-`by`-nothing squares: pieces of color `by` attacking `sq`
 * given occupancy `occ`. (Their king included — needed for the danger map.) */
static inline uint64_t attackers_to(const State *s, int sq, int by, uint64_t occ) {
    return (pawn_att_tbl[by ^ 1][sq] & s->bb[by][PAWN])
         | (knight_tbl[sq] & s->bb[by][KNIGHT])
         | (king_tbl[sq] & s->bb[by][KING])
         | (bishop_attacks(sq, occ) & (s->bb[by][BISHOP] | s->bb[by][QUEEN]))
         | (rook_attacks(sq, occ) & (s->bb[by][ROOK] | s->bb[by][QUEEN]));
}

/* The full attack map of every piece of color `by` under occupancy `occ`.
 * Used with occ = occupancy minus our king, giving the king's danger map:
 * squares it may not step onto (including x-rays behind itself). */
static uint64_t attack_map(const State *s, int by, uint64_t occ) {
    uint64_t att, b;

    b = s->bb[by][PAWN];
    att = by == WHITE
        ? (((b & ~FILEA_BB) << 7) | ((b & ~FILEH_BB) << 9))
        : (((b & ~FILEA_BB) >> 9) | ((b & ~FILEH_BB) >> 7));

    b = s->bb[by][KNIGHT];
    while (b) { att |= knight_tbl[__builtin_ctzll(b)]; b &= b - 1; }

    b = s->bb[by][BISHOP] | s->bb[by][QUEEN];
    while (b) { att |= bishop_attacks(__builtin_ctzll(b), occ); b &= b - 1; }

    b = s->bb[by][ROOK] | s->bb[by][QUEEN];
    while (b) { att |= rook_attacks(__builtin_ctzll(b), occ); b &= b - 1; }

    att |= king_tbl[__builtin_ctzll(s->bb[by][KING])];
    return att;
}

/* ------------------------------------------------------------------ */
/* Pawn moves                                                          */
/* ------------------------------------------------------------------ */

/* Pawns need both a set-wise path (the common case: no pin, no check
 * restriction beyond the mask) and a per-pawn path for pinned pawns,
 * whose every target must additionally stay on the pin line. */
static void gen_pawn_moves(const State *s, MoveList *ml, int us,
                           uint64_t pinned, uint64_t capture_mask, int ksq) {
    const int      them    = us ^ 1;
    const uint64_t own     = s->occ[us];
    const uint64_t them_bb = s->occ[them];
    const uint64_t occ     = own | them_bb;
    const uint64_t empt    = ~occ;
    const uint64_t pawns   = s->bb[us][PAWN];
    const uint64_t promo_rank = us == WHITE ? RANK8_BB : RANK1_BB;
    const int      fwd     = us == WHITE ? 8 : -8;
    const int      home_rank = us == WHITE ? 1 : 6; /* rank index of double-push start */

    static const int promo_flags[4]     = {F_PROMO_N, F_PROMO_B, F_PROMO_R, F_PROMO_Q};
    static const int promo_cap_flags[4] = {F_PROMO_CAP_N, F_PROMO_CAP_B, F_PROMO_CAP_R, F_PROMO_CAP_Q};

    /* --- free (unpinned) pawns: four set-wise streams -------------------- */
    const uint64_t free = pawns & ~pinned;

    const uint64_t one = us == WHITE ? (free << 8) & empt : (free >> 8) & empt;
    /* Double pushes come from `one` restricted to the intermediate rank, so
     * the intermediate square being empty is already guaranteed; only the
     * LANDING square needs to be empty (and in capture_mask). */
    const uint64_t two = us == WHITE
        ? ((one & RANK3_BB) << 8) & empt : ((one & RANK6_BB) >> 8) & empt;
    const uint64_t cap_left  = us == WHITE
        ? ((free & ~FILEA_BB) << 7) & them_bb : ((free & ~FILEA_BB) >> 9) & them_bb;
    const uint64_t cap_right = us == WHITE
        ? ((free & ~FILEH_BB) << 9) & them_bb : ((free & ~FILEH_BB) >> 7) & them_bb;

    uint64_t b;

    b = one & ~promo_rank & capture_mask;
    while (b) { int to = __builtin_ctzll(b); b &= b - 1; add_move(ml, to - fwd, to, F_QUIET); }

    b = two & capture_mask;
    while (b) { int to = __builtin_ctzll(b); b &= b - 1; add_move(ml, to - 2 * fwd, to, F_DOUBLE_PUSH); }

    b = cap_left & ~promo_rank & capture_mask;
    while (b) { int to = __builtin_ctzll(b); b &= b - 1; add_move(ml, to - (us == WHITE ? 7 : -9), to, F_CAPTURE); }

    b = cap_right & ~promo_rank & capture_mask;
    while (b) { int to = __builtin_ctzll(b); b &= b - 1; add_move(ml, to - (us == WHITE ? 9 : -7), to, F_CAPTURE); }

    b = one & promo_rank & capture_mask;
    while (b) {
        int to = __builtin_ctzll(b); b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - fwd, to, promo_flags[i]);
    }
    b = cap_left & promo_rank & capture_mask;
    while (b) {
        int to = __builtin_ctzll(b); b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - (us == WHITE ? 7 : -9), to, promo_cap_flags[i]);
    }
    b = cap_right & promo_rank & capture_mask;
    while (b) {
        int to = __builtin_ctzll(b); b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - (us == WHITE ? 9 : -7), to, promo_cap_flags[i]);
    }

    /* --- pinned pawns: one at a time, targets on the pin line only ------- */
    b = pawns & pinned;
    while (b) {
        const int from = __builtin_ctzll(b);
        b &= b - 1;
        const uint64_t line = line_bb[ksq][from];

        /* Pushes (and the double push, whose intermediate square lies on the
         * same file between from and target, hence on the line already). */
        const int t1 = from + fwd;
        if (t1 >= 0 && t1 < 64 && (empt & (1ULL << t1))
            && (line & (1ULL << t1)) && (capture_mask & (1ULL << t1))) {
            if (promo_rank & (1ULL << t1)) {
                for (int i = 0; i < 4; i++) add_move(ml, from, t1, promo_flags[i]);
            } else {
                add_move(ml, from, t1, F_QUIET);
                if ((from >> 3) == home_rank) {
                    const int t2 = from + 2 * fwd;
                    if ((empt & (1ULL << t2)) && (capture_mask & (1ULL << t2)))
                        add_move(ml, from, t2, F_DOUBLE_PUSH);
                }
            }
        }

        /* Captures: the two diagonal targets, if occupied by them, on the
         * pin line (a diagonal pin allows capturing the pinner; a file pin
         * allows nothing but pushes; a rank pin allows nothing). */
        for (int d = 0; d < 2; d++) {
            const int df = d ? 1 : -1;
            const int f = (from & 7) + df, r = (from >> 3) + (us == WHITE ? 1 : -1);
            if (f < 0 || f > 7 || r < 0 || r > 7) continue;
            const int t = r * 8 + f;
            const uint64_t tb = 1ULL << t;
            if (!(them_bb & tb) || !(line & tb) || !(capture_mask & tb)) continue;
            if (promo_rank & tb)
                for (int i = 0; i < 4; i++) add_move(ml, from, t, promo_cap_flags[i]);
            else
                add_move(ml, from, t, F_CAPTURE);
        }
    }
}

/* En passant: explicit verification (see file header). Candidate pawns are
 * the ones attacking the ep target; a candidate is legal iff
 *   - in check: the victim IS the checker, or the landing square blocks the
 *     checking ray (necessarily a slider — knight/pawn checks can be
 *     neither blocked nor ep-captured);
 *   - always: no enemy slider attacks our king on the post-capture
 *     occupancy (from, victim gone; our pawn on the ep square). This one
 *     test covers the pin on the capturing pawn AND the notorious
 *     two-pawns-leave-a-rank case. */
static void gen_ep_moves(const State *s, MoveList *ml, int us, int ksq,
                         uint64_t checkers, uint64_t capture_mask) {
    if (s->ep == NO_SQ) return;

    const int      them  = us ^ 1;
    const uint64_t occ   = occ_all(s);
    const uint64_t ep_bb = 1ULL << s->ep;
    if (occ & ep_bb) return; /* never set in a legal position; guards bad FENs */

    const int      victim_sq = us == WHITE ? s->ep - 8 : s->ep + 8;
    const uint64_t victim_bb = 1ULL << victim_sq;
    const uint64_t bq = s->bb[them][BISHOP] | s->bb[them][QUEEN];
    const uint64_t rq = s->bb[them][ROOK]   | s->bb[them][QUEEN];

    uint64_t attackers = pawn_att_tbl[them][s->ep] & s->bb[us][PAWN];
    while (attackers) {
        const int from = __builtin_ctzll(attackers);
        attackers &= attackers - 1;

        if (checkers) {
            if (checkers != victim_bb) {
                /* not capturing the checking pawn: legal only if our pawn
                 * lands between checker and king, and only a slider check
                 * can be blocked that way */
                if (!(checkers & (bq | rq)) || !(capture_mask & ep_bb)) continue;
            }
        }

        const uint64_t occ_after = (occ ^ (1ULL << from) ^ victim_bb) | ep_bb;
        if (bishop_attacks(ksq, occ_after) & bq) continue;
        if (rook_attacks(ksq, occ_after) & rq) continue;

        add_move(ml, from, s->ep, F_EP);
    }
}

/* ------------------------------------------------------------------ */
/* Castling (validated at generation: cannot be re-detected later)     */
/* ------------------------------------------------------------------ */

/* Danger map note: `danger` is computed with our king removed from the
 * occupancy, so sliders x-ray through the king's square. For the castling
 * through/landing squares this can only add false "attacked" marks along
 * rays that pass THROUGH the king — and every such ray means the king is
 * itself attacked, i.e. we are in check, and castling is not generated at
 * all. So reusing `danger` here is exact. */
static void gen_castling(const State *s, MoveList *ml, int us, uint64_t danger) {
    const int ksq  = us == WHITE ? E1 : E8;
    const uint64_t occ = occ_all(s);

    const int k_from = ksq, k_to_ks = us == WHITE ? G1 : G8, k_to_qs = us == WHITE ? C1 : C8;
    const int r_ks   = us == WHITE ? H1 : H8, r_qs  = us == WHITE ? A1 : A8;
    const int through_ks = us == WHITE ? F1 : F8;
    const int through_qs = us == WHITE ? D1 : D8;
    const int b_qs       = us == WHITE ? B1 : B8; /* rook passes; king doesn't */

    const bool wk = us == WHITE;
    if ((s->castling & (wk ? CASTLE_WK : CASTLE_BK))
        && s->board[k_from] == pc_of(us, KING) && s->board[r_ks] == pc_of(us, ROOK)
        && !(occ & ((1ULL << through_ks) | (1ULL << k_to_ks)))
        && !(danger & ((1ULL << through_ks) | (1ULL << k_to_ks)))) {
        add_move(ml, k_from, k_to_ks, F_KCASTLE);
    }
    if ((s->castling & (wk ? CASTLE_WQ : CASTLE_BQ))
        && s->board[k_from] == pc_of(us, KING) && s->board[r_qs] == pc_of(us, ROOK)
        && !(occ & ((1ULL << b_qs) | (1ULL << through_qs) | (1ULL << k_to_qs)))
        && !(danger & ((1ULL << through_qs) | (1ULL << k_to_qs)))) {
        add_move(ml, k_from, k_to_qs, F_QCASTLE);
    }
}

/* ------------------------------------------------------------------ */
/* Pseudo-legal generation (Phase 1.2's stage 1, kept as a test oracle  */
/* and debugging aid; gen_legal no longer calls it)                    */
/* ------------------------------------------------------------------ */

static void gen_pseudo_pawn_moves(const State *s, MoveList *ml, int us) {
    const int    them    = us ^ 1;
    const uint64_t own    = s->occ[us];
    const uint64_t them_bb = s->occ[them];
    const uint64_t empt   = ~(own | them_bb);
    const uint64_t pawns  = s->bb[us][PAWN];
    const uint64_t promo_rank = us == WHITE ? RANK8_BB : RANK1_BB;

    const int      fwd     = us == WHITE ? 8 : -8;
    const uint64_t one     = us == WHITE ? (pawns << 8) & empt : (pawns >> 8) & empt;
    const uint64_t two     = us == WHITE ? ((one & RANK3_BB) << 8) & empt
                                          : ((one & RANK6_BB) >> 8) & empt;
    const uint64_t cap_left  = us == WHITE ? ((pawns & ~FILEA_BB) << 7) & them_bb
                                           : ((pawns & ~FILEA_BB) >> 9) & them_bb;
    const uint64_t cap_right = us == WHITE ? ((pawns & ~FILEH_BB) << 9) & them_bb
                                           : ((pawns & ~FILEH_BB) >> 7) & them_bb;

    uint64_t b = one & ~promo_rank;
    while (b) { int to = __builtin_ctzll(b); b &= b - 1; add_move(ml, to - fwd, to, F_QUIET); }
    b = two;
    while (b) { int to = __builtin_ctzll(b); b &= b - 1; add_move(ml, to - 2 * fwd, to, F_DOUBLE_PUSH); }
    b = cap_left & ~promo_rank;
    while (b) { int to = __builtin_ctzll(b); b &= b - 1; add_move(ml, to - (us == WHITE ? 7 : -9), to, F_CAPTURE); }
    b = cap_right & ~promo_rank;
    while (b) { int to = __builtin_ctzll(b); b &= b - 1; add_move(ml, to - (us == WHITE ? 9 : -7), to, F_CAPTURE); }

    static const int promo_flags[4] = {F_PROMO_N, F_PROMO_B, F_PROMO_R, F_PROMO_Q};
    static const int promo_cap_flags[4] = {F_PROMO_CAP_N, F_PROMO_CAP_B, F_PROMO_CAP_R, F_PROMO_CAP_Q};

    b = one & promo_rank;
    while (b) {
        int to = __builtin_ctzll(b); b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - fwd, to, promo_flags[i]);
    }
    b = cap_left & promo_rank;
    while (b) {
        int to = __builtin_ctzll(b); b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - (us == WHITE ? 7 : -9), to, promo_cap_flags[i]);
    }
    b = cap_right & promo_rank;
    while (b) {
        int to = __builtin_ctzll(b); b &= b - 1;
        for (int i = 0; i < 4; i++) add_move(ml, to - (us == WHITE ? 9 : -7), to, promo_cap_flags[i]);
    }

    if (s->ep != NO_SQ && !((own | them_bb) & (1ULL << s->ep))) {
        uint64_t attackers = pawn_att_tbl[them][s->ep] & pawns;
        while (attackers) {
            int from = __builtin_ctzll(attackers);
            attackers &= attackers - 1;
            add_move(ml, from, s->ep, F_EP);
        }
    }
}

static void gen_pseudo_castling(const State *s, MoveList *ml, int us) {
    const int them = us ^ 1;
    const int ksq  = us == WHITE ? E1 : E8;
    const uint64_t all = occ_all(s);

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

void gen_pseudo_legal(const State *s, MoveList *ml) {
    ml->count = 0;
    const int us = s->side, them = us ^ 1;
    const uint64_t own = s->occ[us], them_bb = s->occ[them];

    gen_pseudo_pawn_moves(s, ml, us);
    gen_pseudo_castling(s, ml, us);

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

/* ------------------------------------------------------------------ */
/* Strictly-legal generation                                           */
/* ------------------------------------------------------------------ */

void gen_legal(const State *s, MoveList *ml) {
    ml->count = 0;
    const int      us      = s->side;
    const int      them    = us ^ 1;
    const uint64_t own     = s->occ[us];
    const uint64_t them_bb = s->occ[them];
    const uint64_t occ     = own | them_bb;
    const int      ksq     = king_sq(s, us);

    /* King moves first: they are generated even in double check, the one
     * case where nothing else may move at all. */
    const uint64_t danger = attack_map(s, them, occ ^ (1ULL << ksq));
    add_targets(ml, ksq, king_tbl[ksq] & ~own & ~danger, them_bb);

    /* Checkers; double check → king moves only, done. */
    const uint64_t checkers = attackers_to(s, ksq, them, occ);
    if (__builtin_popcountll(checkers) > 1) return;

    /* Non-king moves must capture the checker or block its ray. */
    const uint64_t capture_mask = checkers
        ? (between_bb[ksq][__builtin_ctzll(checkers)] | checkers)
        : ~0ULL;

    /* Pinned pieces: enemy sliders that *would* attack our king on an empty
     * board ("snipers"), with exactly one blocker between — ours → pin.
     * (Zero blockers would make the sniper a checker, already found above;
     * their single blocker only blocks, no pin.) */
    uint64_t pinned = 0;
    uint64_t snipers = (bishop_attacks(ksq, 0) & (s->bb[them][BISHOP] | s->bb[them][QUEEN]))
                     | (rook_attacks(ksq, 0)   & (s->bb[them][ROOK]   | s->bb[them][QUEEN]));
    while (snipers) {
        const int ssq = __builtin_ctzll(snipers);
        snipers &= snipers - 1;
        const uint64_t blockers = between_bb[ksq][ssq] & occ;
        if (blockers && !(blockers & (blockers - 1)) && (blockers & own))
            pinned |= blockers;
    }

    /* Pawns (set-wise for free pawns; individually for pinned pawns) and
     * en passant (explicit verification — see file header). */
    gen_pawn_moves(s, ml, us, pinned, capture_mask, ksq);
    gen_ep_moves(s, ml, us, ksq, checkers, capture_mask);

    /* Knights. A pinned knight can never move: knight jumps (±1,±2)/(±2,±1)
     * are never collinear with a rank/file/diagonal, so the pin-line mask
     * below always yields the empty set for it — no special case needed. */
    uint64_t b = s->bb[us][KNIGHT];
    while (b) {
        const int from = __builtin_ctzll(b);
        b &= b - 1;
        uint64_t att = knight_tbl[from] & ~own & capture_mask;
        if (pinned & (1ULL << from)) att &= line_bb[ksq][from];
        add_targets(ml, from, att, them_bb);
    }

    /* Diagonal sliders. Pinned + in check at once cannot leave a legal
     * move (pin line and check ray are distinct lines through the king),
     * but applying both masks is still exact — their AND is just empty. */
    b = s->bb[us][BISHOP];
    while (b) {
        const int from = __builtin_ctzll(b);
        b &= b - 1;
        uint64_t att = bishop_attacks(from, occ) & ~own & capture_mask;
        if (pinned & (1ULL << from)) att &= line_bb[ksq][from];
        add_targets(ml, from, att, them_bb);
    }
    b = s->bb[us][ROOK];
    while (b) {
        const int from = __builtin_ctzll(b);
        b &= b - 1;
        uint64_t att = rook_attacks(from, occ) & ~own & capture_mask;
        if (pinned & (1ULL << from)) att &= line_bb[ksq][from];
        add_targets(ml, from, att, them_bb);
    }
    b = s->bb[us][QUEEN];
    while (b) {
        const int from = __builtin_ctzll(b);
        b &= b - 1;
        uint64_t att = (bishop_attacks(from, occ) | rook_attacks(from, occ)) & ~own & capture_mask;
        if (pinned & (1ULL << from)) att &= line_bb[ksq][from];
        add_targets(ml, from, att, them_bb);
    }

    /* Castling: never legal while in check; through/landing squares must
     * be out of `danger` (see gen_castling for why the king-removed
     * occupancy is still exact here). */
    if (!checkers) gen_castling(s, ml, us, danger);
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
