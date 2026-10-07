/*
 * chess_puct.h — Phase 1 PUCT MCTS (puct_mcts_plan.md, this branch only).
 *
 * Single-thread, single-game search. One PuctGame owns a State + NN history
 * ring + one mmap'd node block (64 MB default) that is bump-allocated and
 * packed into a fresh block whenever it fills (see Memory below).
 *
 * Node layout (D1/D6): W (value_sum) is stored in the PARENT-mover's view,
 * so selection is sign-free and backup flips the sign once per ply. Children
 * are created eagerly at expansion, contiguous in the block. The tree
 * persists across moves: puct_play makes the played child the new root and
 * keeps its subtree (no renormalization needed — root W is never read).
 *
 * Memory (D6): one mmap'd block per game, bump-allocated. When an
 * allocation no longer fits, the search packs: a fresh block is mapped, the
 * subtree reachable from the root is copied over with parent/first_child
 * remapped (dead subtrees of earlier moves are dropped), the old block is
 * munmap'd, and the allocation is retried once. If even a packed fresh
 * block cannot fit the request, the OOM counter ticks and the search gives
 * up for this think.
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
#define PUCT_MAX_WALKERS 8               /* in-flight evals per game      */
#define PUCT_BANKS 2                     /* landing-zone banks (double buffer) */

typedef struct PuctNode {    /* u32 handles only, no pointers (pack-safe) */
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
#define PUCT_SLEEPING 2          /* a walker is parked on this node; the
                                    parked indices live in the walkers
                                    themselves, compaction scans them */

typedef struct PuctGame PuctGame;
typedef struct PuctMux PuctMux;      /* multi-game multiplexer           */

/* planes: bank-relative rows. n x NN_TOTAL, masks: n x PUCT_POLICY_SIZE
 * (1 = legal), logp: n x PUCT_POLICY_SIZE out, wdl: n x 3 out (mover's
 * view). bank = which landing-zone bank the batch belongs to. */
typedef enum {
    PUCT_EVAL_PUSH,   /* H2D + launch compute; async — must not block on
                         the GPU (the io thread keeps serving the other
                         bank so computes run back to back) */
    PUCT_EVAL_PULL,   /* enqueue D2H for the bank's results; async */
    PUCT_EVAL_WAIT    /* block until the bank's results are readable in
                         logp/wdl (the only GPU-blocking op) */
} PuctEvalOp;

/* Synchronous evaluators (dummy, MPS, scripted tests) do the whole
 * forward at PUSH and treat PULL/WAIT as no-ops. */
typedef void (*PuctEvalFn)(void *ctx, PuctEvalOp op, int bank, int n,
                           const float *planes, const uint8_t *masks,
                           float *logp, float *wdl);

/* block_bytes = 0 selects PUCT_DEFAULT_BLOCK. NULL/malformed FEN other than
 * NULL/"" → NULL. Call chess_init() once per process first. */
PuctGame *puct_game_new(const char *fen, size_t block_bytes);
void      puct_game_free(PuctGame *g);
void      puct_set_eval(PuctEvalFn fn, void *ctx);   /* the one evaluator */
void      puct_set_cpuct(PuctGame *g, float c);

/* Recycle in place: fresh startpos game — state, history, tree and walkers
 * reset; the mmap'd block, the zone and the OOM/compaction stats are kept.
 * This is how the mux starts a new game without freeing memory. */
void      puct_game_reset(PuctGame *g);

/* ---- the landing zone (global, Phase 3b) ---------------------------------
 * Eval rows shared by every game in the process, in TWO banks (double
 * buffer): the walker thread stages into the active bank while the io
 * thread evaluates the other, so an eval no longer blocks walking. Row ids
 * are global across banks (bank = row / cap). Plus the evaluator and its
 * single io thread — the strict ping-pong of the single-game design,
 * unchanged; a walker claims a row when it parks and remembers its row id,
 * so delivery needs no row→game table. Row storage is permanent (static,
 * paged in only for used rows; PUCT_MAX_ROWS caps a bank at compile time)
 * — nothing here allocates. */
void puct_zone_init(int max_rows);   /* rows PER BANK, clamped to
                                        PUCT_MAX_ROWS; safe to call again
                                        between rounds to resize */
void puct_zone_set_buffers(int bank, float *planes, uint8_t *mask,
                           float *logp, float *wdl);  /* nn_eval.c seam:
                                             point the bank at caller-owned
                                             pinned tensors (e.g. sized
                                             rows = zone cap) */

/* Eval-seam request builders for the current position. */
void puct_request_planes(PuctGame *g, float out[NN_TOTAL]);
void puct_request_mask(PuctGame *g, uint8_t out[PUCT_POLICY_SIZE]);

/* Move → 4240-slot policy index, mover-relative (chess_ai/net/features.py
 * uci_to_index is the spec). Returns -1 for impossible promotion geometry. */
int  chess_policy_slot(const State *s, Move m);

/* Run PUCT simulations from the root until `sims` backups are counted;
 * returns with the game position exactly as it was. W walkers (default 8,
 * see puct_set_walkers) descend the tree concurrently on the calling
 * thread; a walker that reaches a childless leaf expands it in place with
 * uniform placeholder priors, claims an eval row and parks. The landing
 * zone's io thread runs the evaluator over the staged batch; delivery
 * fills in the real priors and backs the value up. Only the calling
 * thread touches the tree. Gives up per-walker (oom counter) if the block
 * cannot be packed further. */
void puct_think(PuctGame *g, int sims);

/* puct_think decomposed for multi-game drivers: */
bool puct_walk(PuctGame *g, int sims);   /* descend/park until all walkers
                                            are parked, the sims target is
                                            met, or the active bank is full;
                                            true = think complete */
bool puct_think_done(const PuctGame *g); /* true between a completed think
                                            and the next walk's fresh start */

/* Number of concurrent walkers / in-flight evals (1..PUCT_MAX_WALKERS).
 * More walkers fill bigger batches but pick nodes more speculatively. */
void puct_set_walkers(PuctGame *g, int n);

/* Argmax visits among root children; 0 if the root has no children. */
Move puct_best_move(const PuctGame *g);

/* Advance the game by `m` (must be legal). If `m` is a root child the tree
 * persists (subtree kept); otherwise a fresh root is allocated. 0 = ok. */
int  puct_play(PuctGame *g, Move m);

uint32_t    puct_oom_count(const PuctGame *g);
uint32_t    puct_compact_count(const PuctGame *g);

/* Per-think leaf-arrival telemetry (reset each fresh think); any output
 * pointer may be NULL. parks = every sim ending at a childless node (eval
 * parks + terminal arrivals); terminal_hits = parks backed up from a
 * terminal. Eval rows staged = parks - terminal_hits - (1 if the root was
 * expanded this think). */
void puct_park_stats(const PuctGame *g, uint32_t *parks,
                     uint32_t *terminal_hits);
const State *puct_state(const PuctGame *g);

/* Root children in generation order; any output pointer may be NULL.
 * q is W/N in the root mover's view (0 while N = 0). Returns the count. */
int puct_root_children(const PuctGame *g, uint16_t *moves, uint32_t *visits,
                       double *q, float *prior);

/* The root node's own visit count and value sum (parent-mover's view). */
void puct_root_stats(const PuctGame *g, uint32_t *visits, double *value_sum);

/* ---- multi-game multiplexer (Phase 3a) ----------------------------------
 * Owns a pool of at most games_max games (all start from the start
 * position), created one at a time: whenever every live game's walkers are
 * parked (all sleeping) and the batch still has room, the next step starts
 * one more game. Full or stuck banks are shipped to the io thread
 * asynchronously; a step ends when every row it staged is delivered, so
 * the driver can play moves between steps (puct_think_done +
 * puct_best_move + puct_play) and recycle finished games via
 * puct_game_reset. block_bytes = 0 selects PUCT_DEFAULT_BLOCK. */
PuctMux  *puct_mux_new(int games_max, int rows_max, size_t block_bytes);
void      puct_mux_free(PuctMux *m);
void      puct_mux_step(PuctMux *m, int sims);  /* one fill → eval → deliver */
int       puct_mux_games(const PuctMux *m);
PuctGame *puct_mux_game(const PuctMux *m, int i);
uint64_t  puct_mux_rows(const PuctMux *m);      /* cumulative eval rows */

#ifdef __cplusplus
}
#endif

#endif /* CHESS_PUCT_H */
