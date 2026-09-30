/*
 * chess_sp.h — the parallel self-play engine (Phase 3.3b).
 *
 * ARCHITECTURE (PARALLEL_DESIGN.md, settled over review rounds R1–R7)
 * ===================================================================
 * cfg.n_workers MCTS worker threads, and the sp_engine_wait() caller
 * plays the evaluator between epochs (no dedicated evaluator thread —
 * see PARALLEL_DESIGN §8, deviation 7). A game lives on one
 * worker for its whole life (tree, working State, records — no shared
 * trees, no virtual loss); workers advance their games round-robin and
 * every parked descent's request goes into one of the worker's two buffer
 * slots (B requests each, sealed on full or no-work). All synchronization
 * concentrates at the batch boundary: once EVERY active worker has a
 * sealed buffer (the per-epoch barrier), the evaluator takes all sealed
 * slots, runs one batched inference, and hands each worker its results;
 * workers wake and deliver into their own games (the evaluator never
 * writes trees, F7). The game machines are exactly the wave-mode state
 * machines of chess_search.h — sp_game_step_wave / sp_deliver_node are
 * the design's game_step and result delivery.
 *
 * Liveness (designed-in, §5; asserted by tests):
 *   - no-work seal: a worker whose games cannot progress seals its
 *     nonempty slot and posts; new games are started freely while the
 *     atomic games_started reservation is below the budget (zero game
 *     overshoot — the reservation is the only admission control);
 *   - deferred parks: once the buffer seals, the worker keeps
 *     stepping games, parking their requests into a deferred list, which
 *     drains first into freed slots after the epoch;
 *   - exit, don't post: a worker with no games, no sealed slots, no
 *     deferred requests and an exhausted budget leaves; the evaluator's
 *     predicate tracks the shrinking active_workers and exits at zero.
 *
 * Deterministic-interleave mode (R6/L3): the same fill/gather/evaluate/
 * deliver machinery driven by one OS thread in fixed worker order — the
 * full parallel semantics (waves, barriers, batching) with a
 * deterministic schedule. Bit-reproducible run-to-run; with a
 * batch-invariant evaluator the threaded and interleave runs also agree
 * with each other and with the sequential mode exactly, which is the
 * engine's race-detector gate (any divergence points at a data race).
 */
#ifndef CHESS_SP_H
#define CHESS_SP_H

#include <stdint.h>

#include "chess_search.h"
#include "chess_sp_eval.h"
#include "chess_sp_dump.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Static capacity caps (deviation 9: the engine allocates nothing). All
 * storage — game pool, worker buffers, deferred lists, batch staging — is
 * embedded in ONE block of SP_ENGINE_STATIC_SIZE bytes that the CALLER
 * provides (C tests: BSS; Python: an anonymous mmap). The block is lazily
 * committed: unused pool slots cost no physical memory.
 *
 * TWO DIFFERENT BOUNDS — do not conflate (deviation 10):
 *   - game_budget: CUMULATIVE games this run plays — training
 *     granularity (10k, 100k, ...), NOT memory-bound.
 *   - SP_ENGINE_MAX_OUTSTANDING: CONCURRENT in-flight games per worker —
 *     memory-bound (pool slots). A finished game is dumped to its
 *     worker's shard file and its slot is recycled IMMEDIATELY
 *     (chess_sp_dump.h), so the pool bounds concurrency, never the
 *     budget. */
#define SP_ENGINE_MAX_WORKERS     8
#define SP_ENGINE_MAX_OUTSTANDING 128   /* in-flight games per worker      */
#define SP_ENGINE_MAX_BUFFER      256   /* B: epoch buffer requests/worker */

/* Upper bound of the engine's static footprint (pool + buffers + staging
 * + slack); sp_engine_init asserts the caller's block is at least this. */
#define SP_ENGINE_STATIC_SIZE                                             \
    ((size_t)SP_ENGINE_MAX_WORKERS * SP_ENGINE_MAX_OUTSTANDING           \
      * SP_GAME_SLOT_SIZE + (size_t)256u * 1024u * 1024u)

typedef struct SpEngine SpEngine;

/* The evaluator is injected as plain C function pointers: the embedded
 * CPython implementation (sp_evaluator.c) is one; C fixtures in the tests
 * are another. init is called once before the first batch (it may use
 * Python), free_fn once after the run ends. batch must fill the outputs
 * for every request and return 0 (nonzero fails the run). */
typedef int  (*SpEvalInitFn)(void *ctx);
typedef int  (*SpEvalBatchFn)(void *ctx, SpEvalBatch *b);
typedef void (*SpEvalFreeFn)(void *ctx);

typedef struct {
    void         *ctx;
    SpEvalInitFn  init;      /* may be NULL  */
    SpEvalBatchFn batch;     /* required     */
    SpEvalFreeFn  free_fn;   /* may be NULL  */
} SpEvaluator;

/* The record consumer is a FILE, not a callback (user decision
 * 2026-09-28): each worker dumps finished games to its own shard in
 * record_dir (chess_sp_dump.h) at collect time and recycles the slot
 * immediately — no drain hook, no Python on the record path. The trainer
 * reads the shards back later (3.4). Empty string = no dump (runs that
 * don't need records, e.g. the failure-path test). */

typedef struct {
    SpConfig game_cfg;         /* per-game search knobs                       */
    uint64_t unit_seed;        /* game i is seeded game_seed(unit_seed, i)    */
    int32_t  game_budget;      /* CUMULATIVE games to play this run —
                                  training granularity (zero overshoot via
                                  the CAS reservation); NOT memory-bound      */
    int32_t  n_workers;        /* worker threads ≤ SP_ENGINE_MAX_WORKERS; the
                                  wait() caller plays the evaluator between
                                  epochs                                      */
    int32_t  buffer_requests;  /* B: epoch buffer requests/worker
                                  (≤ SP_ENGINE_MAX_BUFFER; design: 256)       */
    int32_t  max_plies;        /* adjudication bound per game (< 0 unbounded) */
    int32_t  interleave;       /* deterministic single-thread mode            */
    char     record_dir[SP_DUMP_PATH_MAX];   /* shard dump dir; empty = off */
    /* Diverse game init (3.5 finding: startpos-only self-play self-
     * confirms). Game i starts from start_fens[i % n_start_fens]; NULL/0
     * = every game from the standard start position. The STRINGS are the
     * caller's and must outlive the run (sp_engine_init copies only the
     * config struct); picking by index consumes no game RNG draws, so the
     * "first draw is move 1's Gumbels" contract and (unit_seed, game_index)
     * determinism are intact. A malformed pool FEN fails the run. */
    const char *const *start_fens;
    int32_t  n_start_fens;
} SpEngineConfig;

void sp_engine_config_init(SpEngineConfig *cfg);

/* Lifecycle. sp_engine_init initializes CALLER-PROVIDED storage (≥
 * SP_ENGINE_STATIC_SIZE bytes, 64-aligned; C tests declare it in BSS,
 * Python mmaps it) — the engine allocates nothing, ever. It copies the
 * config and evaluator and returns the engine handle.
 * sp_engine_start spawns the worker threads (or, in interleave mode,
 * arranges the single-threaded run) and returns immediately.
 * sp_engine_wait DRIVES the run — its caller plays the evaluator between
 * epochs — and returns only when the run ends; check sp_engine_error
 * after. (The engine makes no progress unless wait() is running.)
 * sp_engine_destroy releases nothing (storage is the caller's); it runs
 * the evaluator's free hook. */
SpEngine *sp_engine_init(void *storage, size_t storage_size,
                         const SpEngineConfig *cfg, const SpEvaluator *ev);
void      sp_engine_destroy(SpEngine *e);

/* SP_ENGINE_STATIC_SIZE, as a function (same drift-proofing). */
size_t sp_engine_static_size(void);
int       sp_engine_start(SpEngine *e);   /* 0 = ok                        */
void      sp_engine_wait(SpEngine *e);
int       sp_engine_error(const SpEngine *e);   /* 0 ok, 1 failed         */

/* Telemetry. */
int  sp_engine_games_started(const SpEngine *e);
int  sp_engine_games_done(const SpEngine *e);
long sp_engine_evals(const SpEngine *e);      /* requests evaluated         */
long sp_engine_epochs(const SpEngine *e);     /* batched forward passes     */
int  sp_engine_max_batch(const SpEngine *e);  /* largest batch seen         */

/* Convenience factory for the embedded-CPython evaluator: fills `ev` with
 * function pointers that import `module` (e.g.
 * "chess_ai.selfplay.sp_evaluator"), call its make_evaluator(`spec`) once,
 * and dispatch eval_batch per epoch via memoryviews. Only available in
 * libchess_sp (the build with Python bindings). */
int sp_py_evaluator_make(const char *module, const char *spec,
                         SpEvaluator *ev);

#ifdef __cplusplus
}
#endif

#endif /* CHESS_SP_H */
