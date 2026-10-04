/*
 * chess_puct.h — Phase 1 PUCT MCTS (puct_mcts_plan.md, this branch only).
 *
 * Single-thread, single-game search. One PuctGame owns a State + NN history
 * ring + one fixed node block (64 MB default) that is bump-allocated until
 * it runs out (no compaction yet — the OOM counter records the give-up).
 *
 * Node layout (D1/D6): W (value_sum) is stored in the PARENT-mover's view,
 * so selection is sign-free and backup flips the sign once per ply. Children
 * are created eagerly at expansion, contiguous in the block. The tree
 * persists across moves: puct_play makes the played child the new root and
 * keeps its subtree (no renormalization needed — root W is never read).
 *
 * Eval seam (D4): requests are built entirely C-side (planes via nn_encode,
 * mask via gen_legal + chess_policy_slot); the callback only runs the
 * forward. A NULL evaluator is the dummy path: uniform priors, value 0.
 */
#ifndef CHESS_PUCT_H
#define CHESS_PUCT_H

#include "chess.h"
#include "chess_nn.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PUCT_POLICY_SIZE 4240            /* 4096 from-to + 72 + 72 promos */
#define PUCT_DEFAULT_BLOCK (64u << 20)   /* 64 MB node block per game     */

typedef struct PuctNode {    /* 48 bytes; u32 handles, no pointers */
    uint32_t parent;         /* block idx; 0 = none (root's parent) */
    uint32_t first_child;    /* children contiguous; 0 = childless  */
    uint32_t n_children;
    uint32_t visits;         /* N */
    double   value_sum;      /* W, parent-mover's view */
    float    prior;          /* P */
    float    wdl[3];         /* terminal nodes only, mover's view */
    uint16_t move;           /* move from the parent */
    uint8_t  flags;
    uint8_t  pad;
} PuctNode;

#define PUCT_TERMINAL 1

typedef struct PuctGame PuctGame;

/* planes: n x NN_TOTAL, masks: n x PUCT_POLICY_SIZE (1 = legal),
 * logp: n x PUCT_POLICY_SIZE out, wdl: n x 3 out (mover's view). */
typedef void (*PuctEvalFn)(void *ctx, int n, const float *planes,
                           const uint8_t *masks, float *logp, float *wdl);

/* block_bytes = 0 selects PUCT_DEFAULT_BLOCK. NULL/malformed FEN other than
 * NULL/"" → NULL. Call chess_init() once per process first. */
PuctGame *puct_game_new(const char *fen, size_t block_bytes);
void      puct_game_free(PuctGame *g);
void      puct_set_eval(PuctGame *g, PuctEvalFn fn, void *ctx);
void      puct_set_cpuct(PuctGame *g, float c);

/* Eval-seam request builders for the current position. */
void puct_request_planes(PuctGame *g, float out[NN_TOTAL]);
void puct_request_mask(PuctGame *g, uint8_t out[PUCT_POLICY_SIZE]);

/* Move → 4240-slot policy index, mover-relative (chess_ai/net/features.py
 * uci_to_index is the spec). Returns -1 for impossible promotion geometry. */
int  chess_policy_slot(const State *s, Move m);

/* Run `sims` PUCT simulations from the root; returns with the game position
 * exactly as it was. Gives up early (oom counter) if the block runs out. */
void puct_think(PuctGame *g, int sims);

/* Argmax visits among root children; 0 if the root has no children. */
Move puct_best_move(const PuctGame *g);

/* Advance the game by `m` (must be legal). If `m` is a root child the tree
 * persists (subtree kept); otherwise a fresh root is allocated. 0 = ok. */
int  puct_play(PuctGame *g, Move m);

uint32_t    puct_oom_count(const PuctGame *g);
const State *puct_state(const PuctGame *g);

/* Root children in generation order; any output pointer may be NULL.
 * q is W/N in the root mover's view (0 while N = 0). Returns the count. */
int puct_root_children(const PuctGame *g, uint16_t *moves, uint32_t *visits,
                       double *q, float *prior);

/* The root node's own visit count and value sum (parent-mover's view). */
void puct_root_stats(const PuctGame *g, uint32_t *visits, double *value_sum);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_PUCT_H */
