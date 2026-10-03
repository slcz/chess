/*
 * chess_az.h — the AZ-PUCT evaluation search (debug/az-eval branch).
 *
 * WHAT THIS IS
 * ============
 * The C port of scripts/az_eval.py — the clean-room AlphaZero-style PUCT
 * evaluator that acts as the second opinion on net strength, next to the
 * Gumbel search (chess_search.h). EVALUATION ONLY: it never produces
 * training records, runs no RNG of its own (opening sampling stays in the
 * Python driver), and is driven move-by-move: the caller asks for a search,
 * reads the root statistics, picks the move in Python, and plays it here.
 *
 * The search itself, ported line-for-line from az_eval.py's search():
 *
 *   select    argmax_i [ ±q_i + c_puct · P_i · sqrt(ΣN) / (1 + N_i) ]
 *             over the node's legal moves in movegen enumeration order
 *             (ties break by enumeration order, strict > — Python's dict
 *             insertion order is the same enumeration);
 *   expand    every leaf gets ALL children on first visit: priors are the
 *             net's policy_logp over the legal slots, renormalized by a
 *             masked softmax (exp(logp)/Σexp, enumeration order sums);
 *   value     v = pW − pL from the leaf mover's view (wdl[0] − wdl[2]);
 *   terminal  a terminal leaf is scored ±1/0 by the rules (state_termination,
 *             one-hot WDL like the Gumbel search's make_child) and never
 *             asks the net for anything;
 *   backup    v sign-flipped per ply up the path — every node's value_sum
 *             accumulates its OWN mover's view (same convention as
 *             gumbel.py / search.c);
 *   budget    exactly `sims` simulations, one NN evaluation per fresh leaf,
 *             the tree is rebuilt from scratch every move (az_eval.py
 *             discards the root between moves — no handoff, no reuse).
 *
 * THE Q SIGN (read before changing q_parent_view)
 * ===============================================
 * az_eval.py's selection reads the child's q UNNEGATED:
 *
 *     u_score = c.q + c_puct * c.prior * sqrt(total) / (1 + c.visits)
 *
 * where c.q is the child's own-mover value — i.e. the OPPOSING player's
 * view of the move. The AlphaGo paper's Q(s,a) is the selecting player's
 * view, which with own-mover storage is −c.q (what gumbel.py's Eq. 14
 * does). The measured fleet result (v3-r70 vs seed = 0.6125) was produced
 * by the verbatim formula, so the port defaults to it and exposes the
 * paper's sign behind az_think()'s q_parent_view flag:
 *
 *     q_parent_view = 0   verbatim az_eval.py (child view)  — default
 *     q_parent_view = 1   AlphaGo paper Q(s,a) (parent view, negated)
 *
 * THE STATE MACHINE (why this shape)
 * ==================================
 * Same park/deliver protocol as the Gumbel search, one park point (the
 * fresh leaf, including the fresh root): az_step() runs descents until a
 * position needs the net, then returns AZ_EVENT_NEED_EVAL; the caller
 * reads the request (az_request_*), gets (logp, wdl) from ANY evaluator,
 * and resumes with az_deliver(). This is the pull model cffi ABI mode
 * needs — and it is exactly what lets many games run as Python threads
 * while a batcher pools their requests into GPU forwards (the ladder's
 * BatchedInference serves both engines unchanged). Per-game search is
 * single-threaded by design.
 *
 * FAITHFULNESS CONTRACT vs az_eval.py
 * ===================================
 * Identical: the PUCT expression (same f64 evaluation order; the build is
 * -ffp-contract=off like the L2 contract), expansion/backup semantics,
 * enumeration-order tie-breaks, per-move tree reset, root statistics
 * (visits/prior/q per legal move; q is the child's own-mover view, as the
 * Python analyse output prints).
 *
 * Deliberate deviations (documented, semantics-preserving):
 *   - a terminal-on-first-visit leaf never calls the net (az_eval.py
 *     evaluates first and discards the value — pure waste);
 *   - no root Dirichlet noise (match mode never uses it; az_eval.py's
 *     analyse-only --noise stays a Python-side feature);
 *   - under real nets the two engines differ by fp32 batch-shape rounding
 *     (the L3d effect) — statistically equivalent, not bit-exact. With a
 *     deterministic fixture evaluator the walk itself is expected to match
 *     the Python walk exactly (tests/test_az.c + tests/test_az_c.py pin
 *     this).
 *
 * MEMORY
 * ======
 * One malloc'd handle per game (like the FFI's ChessGame — this is an eval
 * path, not the pooled parallel engine, so no caller-provided slot). The
 * node arena is a single bump region inside the handle, reset every
 * az_think(); nodes are u32-indexed pointers into it. Each simulation
 * expands at most one fresh leaf, so a search allocates ≤ sims + 1 nodes;
 * az_think() asserts sims against the table/arena caps. The region is
 * committed lazily (only touched pages cost memory), so the generous cap
 * is nearly free. No mutable globals: any number of handles run
 * concurrently on separate threads.
 */
#ifndef CHESS_AZ_H
#define CHESS_AZ_H

#include <stddef.h>

#include "chess.h"
#include "chess_ffi.h"   /* CHESS_FEN_MAX / CHESS_UCI_MAX */

#ifdef __cplusplus
extern "C" {
#endif

/* Arena geometry. Nodes ≤ sims + 1; the widest legal position (218 moves)
 * makes a 48 + 218*16 = 3536-byte node, so the worst legal search of
 * AZ_MAX_SIMS sims fits with room to spare. */
#define AZ_MAX_SIMS   4095                    /* nodes ≤ sims+1 ≤ table    */
#define AZ_MAX_NODES  4096
#define AZ_ARENA_CAP  (8u * 1024u * 1024u)    /* per-search bump region    */

/* What one az_step() call did:
 *   NEED_EVAL — the search parked on a fresh leaf: fulfil the pending
 *               request (az_request_* / az_deliver), then step again.
 *   DONE      — all sims ran: read the root via az_root_*, then pick a
 *               move in the caller and az_play_uci() it. */
enum {
    AZ_EVENT_NEED_EVAL = 0,
    AZ_EVENT_DONE     = 1,
};

typedef struct AzGame AzGame; /* opaque: State, arena, search state */

/* Create a game (NULL or "" = standard start). Returns NULL on a malformed
 * FEN. Call chess_init() once per process first. One handle per game; one
 * thread per handle. */
AzGame *az_game_new(const char *fen);
void    az_game_free(AzGame *g);

/* ------------------------------------------------------------------ */
/* One move's search                                                    */
/* ------------------------------------------------------------------ */

/* Start a fresh search on the current position (must be PLAYING; the board
 * must be at the game head — not parked inside a previous search). sims is
 * in [1, AZ_MAX_SIMS]; c_puct is the exploration constant (az_eval.py
 * default 1.25); q_parent_view selects the selection Q's sign (see the
 * header). After az_think, call az_step() until it returns AZ_EVENT_DONE. */
void az_think(AzGame *g, int sims, double c_puct, int q_parent_view);

/* Advance the search to the next park point or to completion. */
int az_step(AzGame *g);

/* The pending request (valid between NEED_EVAL and az_deliver). Planes are
 * the 7488-float nn_encode tensor of the parked position (mover-relative,
 * 8-ply history — same as chess_encode); slots are the legal policy slots
 * in movegen enumeration order, the order the delivered logp must align
 * to (also the order of az_root_* indexing). */
const float    *az_request_planes(const AzGame *g);
const uint16_t *az_request_slots(const AzGame *g);
int  az_request_count(const AzGame *g);
int  az_request_mover(const AzGame *g);     /* 0 white / 1 black          */

/* Fulfil the pending request: logp[count] aligned to az_request_slots()
 * (the net's policy_logp at those slots), wdl[3] = (pW, pD, pL) from the
 * parked position's mover's view. Fills priors, backs v = pW − pL up, and
 * resumes; the next az_step() continues the search. */
void az_deliver(AzGame *g, const double *logp, const double *wdl);

/* Convenience for C drivers/tests: run the whole search synchronously,
 * fulfilling every request through (fn, ctx) — the mirror of the Gumbel
 * search's sp_play(). Returns 0. */
typedef void (*AzEvalFn)(void *ctx, const AzGame *g, double *logp,
                         double *wdl);
void az_think_sync(AzGame *g, int sims, double c_puct, int q_parent_view,
                   AzEvalFn fn, void *ctx);

/* ------------------------------------------------------------------ */
/* Root statistics (valid after AZ_EVENT_DONE)                          */
/* ------------------------------------------------------------------ */

/* Children are the root's legal moves in movegen enumeration order.
 * Unvisited children report visits 0 / q 0.0 (exactly the Python Node.q
 * property's convention). q is the child's OWN mover's view — the same
 * number az_eval.py's analyse mode prints (flip the sign yourself if you
 * want the mover-of-the-root's view). */
int    az_root_count(const AzGame *g);
void   az_root_uci(const AzGame *g, int i, char out[CHESS_UCI_MAX]);
int    az_root_visits(const AzGame *g, int i);
double az_root_prior(const AzGame *g, int i);
double az_root_q(const AzGame *g, int i);
/* The root's overall value estimate — az_eval.py's effective search()
 * second return: its root_v is dead code (never assigned), so the
 * reference surfaces root.q, the visits-weighted backup value. Mirror. */
double az_root_value(const AzGame *g);

/* ------------------------------------------------------------------ */
/* The game (the caller owns move selection)                            */
/* ------------------------------------------------------------------ */

/* Play a move by UCI string, validated against the legal-move list (0 ok,
 * -1 illegal/malformed). Keeps the encoder history ring in step. */
int  az_play_uci(AzGame *g, const char *uci);

/* Position state. az_winner: -1 none (draw/ongoing), else 0 white / 1
 * black — chess_ai.engine Board.winner's convention. The first two take a
 * mutable handle because state_termination probes the position (restored
 * before returning) — ffi.c's convention. */
int  az_termination(AzGame *g);        /* TERM_* from chess.h            */
int  az_winner(AzGame *g);
int  az_side(const AzGame *g);          /* 0 white / 1 black              */
int  az_ply(const AzGame *g);
void az_fen(const AzGame *g, char *out); /* ≥ CHESS_FEN_MAX bytes         */

#ifdef __cplusplus
}
#endif

#endif /* CHESS_AZ_H */
