/*
 * sp_engine.c — the parallel self-play engine (chess_sp.h).
 *
 * Threads: cfg.n_workers worker threads, all spawned by sp_engine_start,
 * plus the sp_engine_wait() CALLER, which plays the evaluator between
 * epochs (no dedicated evaluator thread — workers and the evaluator never
 * compute simultaneously in the lock-step epoch, so a reserved thread
 * would idle through every fill phase; see PARALLEL_DESIGN §8, deviation
 * 7). The evaluator is injected as C function pointers — the
 * embedded-CPython implementation lives in sp_evaluator.c, C fixtures in
 * the tests. Locking: one mutex + two condvars, used ONLY at the
 * per-epoch handshake (R2, exactly); the fill sweeps, tree work,
 * deliveries and record dumps are lock-free per worker.
 *
 * Flow per epoch:
 *   1. each worker fills its epoch buffer (capacity 2·B; sweeping games
 *      round-robin, parking one descent per game per sweep, starting new
 *      games while the atomic games_started reservation allows), seals it
 *      if nonempty, posts, and waits on cv_workers;
 *   2. when EVERY active worker posted, the evaluator gathers all sealed
 *      requests into the batch staging (taking the slots), runs one
 *      inference outside the lock, and appends each request's result
 *      (offsets into the output staging) to its owning worker;
 *   3. the evaluator broadcasts; workers wake, take their result list and
 *      deliver outside the lock (sp_deliver_node), then fill again.
 *
 * The result staging is single-buffered and safe: a worker re-posts only
 * after delivering, so the evaluator's next gather overwrites the staging
 * only after every worker finished delivering the previous epoch.
 *
 * Finished games never wait in memory: collect_game dumps the record to
 * the worker's shard file (chess_sp_dump.h) and returns the pool slot to
 * the worker's own free list immediately (same thread — no race), so the
 * pool bounds concurrency, never the budget.
 *
 * Deferred parks: once the buffer seals, the worker keeps stepping its
 * games; parked requests overflow into a per-worker deferred list, which
 * drains first into the buffer after the epoch (R1's deferred-on-full
 * park).
 *
 * Interleave mode (R6/L3): the identical machinery, driven by one thread
 * in fixed worker order (fill all → gather → evaluate → deliver all), so
 * the full parallel semantics run with a deterministic schedule.
 */
#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "chess_sp.h"
#include "chess_sp_eval.h"

#define NN_TOTAL_PLanes  (64 * 117)

typedef struct {
    SpGame  *game;       /* the parked request's owner                    */
    uint32_t node;       /* delivery key                                  */
    int32_t  logp_off;   /* offset into the engine's out_logp staging     */
    int32_t  wdl_off;    /* offset into the engine's out_wdl staging      */
    uint16_t n_legal;
} SpResEntry;

typedef struct {
    SpEngine *engine;

    /* Games owned by this worker (round-robin order) — pool slots. */
    SpGame  *games[SP_ENGINE_MAX_OUTSTANDING];
    int      n_games;
    int      rr;                      /* sweep rotation cursor            */

    /* This worker's pool: free slots, and the in-rotation set. A slot is
     * held from start_game until the run ends (3.4 returns it earlier by
     * streaming records out). */
    SpGame  *free_slots[SP_ENGINE_MAX_OUTSTANDING];
    int      n_free;

    /* The epoch request buffer: B requests (SpReq ≈ 7.9 KB each), sealed
     * on full-or-no-work. Per-worker QUEUE CAPACITY — the evaluator takes
     * the whole buffer every epoch (no slot rotation, deviation 6). */
    SpReq    buf[SP_ENGINE_MAX_BUFFER];
    int      n_buf;
    uint8_t  sealed;
    uint8_t  posted;

    /* Deferred-on-full parks. Bound: every in-flight game parks at most
     * m_cap (≤ 16) descents, so outstanding × 16 covers the worst case. */
    SpReq    deferred[SP_ENGINE_MAX_OUTSTANDING * SP_REC_CANDIDATES];
    int      n_def;

    /* Results awaiting delivery (offsets into the engine's staging). */
    SpResEntry results[SP_ENGINE_MAX_BUFFER];
    int        n_res;

    uint8_t   epoch;            /* set + broadcast at each epoch's end:
                                 * wakes workers even with no results, so
                                 * an empty-posted worker refills */
    pthread_t thread;
    uint8_t   started, active;
    int       dump_fd;          /* this worker's shard file (-1: dump off) */
} SpWorker;

struct SpEvalBatch {
    int32_t  size, total_legal;
    uint8_t *planes;      /* [size × 7488]  u8                            */
    uint16_t *slots;      /* [total_legal]  u16, enum order, concatenated */
    int32_t *offsets;     /* [size + 1]     i32 cumulative                */
    int32_t *mover;       /* [size]         i32                           */
    float   *out_logp;    /* [total_legal]  f32 (evaluator fills)         */
    float   *out_wdl;     /* [size × 3]     f32 (evaluator fills)         */
};

struct SpEngine {
    SpEngineConfig cfg;
    SpEvaluator    ev;

    SpWorker workers[SP_ENGINE_MAX_WORKERS];
    int      n_workers;

    pthread_mutex_t lock;
    pthread_cond_t  cv_eval, cv_workers;
    int      active_workers;
    int      posted_workers;

    atomic_int games_started;
    atomic_int n_done;         /* completed games — telemetry only (the
                                * records themselves go to the shard
                                * files at collect time) */
    long       evals, epochs;
    int        max_batch;
    /* Workers set this (dump failures) while the evaluator reads it —
     * atomic because it crosses threads without the epoch mutex. */
    atomic_bool failed;
    uint8_t    shutdown, running;

    /* Batch staging, embedded — the evaluator reads it via the batch
     * accessors; results reference offsets into it. Sized for the worst
     * epoch (every worker's buffer full). */
    struct SpEvalBatch batch;
    uint8_t  st_planes[SP_ENGINE_MAX_WORKERS * SP_ENGINE_MAX_BUFFER * 7488];
    uint16_t st_slots[SP_ENGINE_MAX_WORKERS * SP_ENGINE_MAX_BUFFER
                      * SP_REQ_MAX_LEGAL];
    int32_t  st_offsets[SP_ENGINE_MAX_WORKERS * SP_ENGINE_MAX_BUFFER + 1];
    int32_t  st_mover[SP_ENGINE_MAX_WORKERS * SP_ENGINE_MAX_BUFFER];
    float    st_logp[SP_ENGINE_MAX_WORKERS * SP_ENGINE_MAX_BUFFER
                     * SP_REQ_MAX_LEGAL];
    float    st_wdl[SP_ENGINE_MAX_WORKERS * SP_ENGINE_MAX_BUFFER * 3];

    /* The game pool: fixed slots, reset per start — never allocated.
     * Raw bytes because SpGame is opaque to this file; SP_GAME_SLOT_SIZE
     * (2 MB, a 64 multiple) bounds sizeof(SpGame) — search.c
     * static-asserts it — so every slot is correctly aligned and large
     * enough. _Alignas keeps the array's own offset 64-aligned. */
    _Alignas(64) char pool[SP_ENGINE_MAX_WORKERS][SP_ENGINE_MAX_OUTSTANDING]
                          [SP_GAME_SLOT_SIZE];
};

/* ------------------------------------------------------------------ */
/* Batch accessors (chess_sp_eval.h)                                   */
/* ------------------------------------------------------------------ */

int  sp_batch_size(const SpEvalBatch *b)        { return b->size; }
int  sp_batch_total_legal(const SpEvalBatch *b) { return b->total_legal; }
const uint8_t  *sp_batch_planes(const SpEvalBatch *b)  { return b->planes; }
const uint16_t *sp_batch_slots(const SpEvalBatch *b)   { return b->slots; }
const int32_t  *sp_batch_offsets(const SpEvalBatch *b) { return b->offsets; }
const int32_t  *sp_batch_mover(const SpEvalBatch *b)   { return b->mover; }
float *sp_batch_out_logp(SpEvalBatch *b) { return b->out_logp; }
float *sp_batch_out_wdl(SpEvalBatch *b)  { return b->out_wdl; }

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

void sp_engine_config_init(SpEngineConfig *cfg) {
    memset(cfg, 0, sizeof *cfg);
    sp_config_init(&cfg->game_cfg);
    cfg->n_workers = 8;          /* reference sizing: 8 threads; the wait()
                                    caller plays the evaluator between epochs */
    cfg->buffer_requests = 256;  /* B (user, R3)                            */
    cfg->max_plies = -1;
}

static void worker_remove_game(SpWorker *w, SpGame *g) {
    for (int i = 0; i < w->n_games; i++) {
        if (w->games[i] == g) {
            w->games[i] = w->games[--w->n_games];
            return;
        }
    }
    assert(0 && "removing a game the worker does not own");
}

/* A worker may leave only when it holds nothing pending and the budget is
 * exhausted (no new games can start). See the header's liveness notes. */
static int worker_exit_ok(SpEngine *e, SpWorker *w) {
    /* Pure end-of-run condition (R3): the budget is exhausted and nothing
     * is in flight. An empty free list alone is NOT exit-worthy — slots
     * return at collect time, so it is transient. */
    return w->n_games == 0 && w->n_buf == 0
        && w->n_def == 0
        && atomic_load(&e->games_started) >= e->cfg.game_budget;
}

/* ------------------------------------------------------------------ */
/* Worker-side: game lifecycle, fill phase, delivery                   */
/* ------------------------------------------------------------------ */

/* Start a game from this worker's pool free list. Returns 0 if the pool
 * is exhausted (the worker then seals and waits; the budget bound —
 * game_budget ≤ pool capacity — means this cannot stall the run). */
static int start_game(SpEngine *e, SpWorker *w, int32_t game_index) {
    if (w->n_free == 0)
        return 0;
    SpGame *g = w->free_slots[--w->n_free];
    sp_game_init(g, &e->cfg.game_cfg, NULL, e->cfg.unit_seed,
                 (uint32_t)game_index);
    sp_game_set_wave_mode(g);
    sp_game_set_owner(g, (int)(w - e->workers));
    sp_set_max_plies(g, e->cfg.max_plies);
    w->games[w->n_games++] = g;
    return 1;
}

/* Collect a finished game: out of the worker's rotation, record dumped to
 * the worker's shard file, pool slot returned to the worker's OWN free
 * list (push and pop both happen on this worker's thread — no race), so
 * a completed game never holds its slot past this call. */
static void collect_game(SpEngine *e, SpWorker *w, SpGame *g) {
    worker_remove_game(w, g);
    if (w->dump_fd >= 0 && sp_dump_game(w->dump_fd, g) != 0)
        atomic_store(&e->failed, 1);   /* a lost record fails the run */
    w->free_slots[w->n_free++] = g;
    atomic_fetch_add(&e->n_done, 1);
}

/* Reserve the next game index (CAS: the counter never exceeds the budget —
 * the zero-overshoot property the R3 reservation promises, exactly). */
static int reserve_game_index(SpEngine *e, int32_t *out) {
    int32_t cur = atomic_load(&e->games_started);
    for (;;) {
        if (cur >= e->cfg.game_budget)
            return 0;
        if (atomic_compare_exchange_weak(&e->games_started, &cur, cur + 1)) {
            *out = cur;
            return 1;
        }
    }
}

/* Per-worker epoch-buffer capacity: 2·B (the unit B stays the config
 * knob; the epoch takes the whole buffer, so the 2 is pure headroom over
 * the ≈B-per-epoch equilibrium). */
static int worker_buf_cap(const SpEngine *e) {
    return e->cfg.buffer_requests;
}

/* One sweep over the worker's games (round-robin order): one wave-step per
 * game. Parks go into the epoch buffer (stops when the buffer is full) or,
 * in the deferred-on-full phase, into the deferred list. Returns whether
 * anything progressed. A finished game is collected immediately (its
 * rotation slot is re-examined). */
static int sweep_games(SpEngine *e, SpWorker *w, int to_deferred) {
    int progressed = 0;
    for (int i = 0; i < w->n_games; ) {
        int gi = (w->rr + i) % w->n_games;
        SpGame *g = w->games[gi];
        SpReq *req;
        if (!to_deferred) {
            if (w->n_buf >= worker_buf_cap(e))
                break;
            req = &w->buf[w->n_buf];
        } else {
            /* bound: outstanding × m_cap parked descents, statically */
            assert(w->n_def < SP_ENGINE_MAX_OUTSTANDING * SP_REC_CANDIDATES);
            req = &w->deferred[w->n_def];
        }
        int ev = sp_game_step_wave(g, req);
        if (ev == SP_STEP_DONE) {
            collect_game(e, w, g);
            continue;      /* no i++: the swap-remove brought a game here */
        }
        i++;
        if (ev == SP_STEP_PARKED) {
            req->game = g;
            if (!to_deferred) w->n_buf++; else w->n_def++;
            progressed = 1;
            if (!to_deferred && w->n_buf >= worker_buf_cap(e))
                break;
        } else if (ev == SP_STEP_PROGRESS) {
            progressed = 1;
        }
    }
    return progressed;
}

/* One fill phase: drain deferred requests first, then sweep, starting new
 * games whenever nothing can progress and the budget allows (the design's
 * §4.3 worker loop), until the buffer is full or no work remains — then
 * seal it if nonempty. With the buffer sealed, the deferred-on-full park
 * (R1) keeps the games busy: whole descents park into the deferred list
 * until a sweep makes no progress, so the next epoch's buffer fills
 * without re-sweeping. */
static void fill_phase(SpEngine *e, SpWorker *w) {
    while (w->n_buf < worker_buf_cap(e)) {
        if (w->n_def > 0) {
            w->buf[w->n_buf++] = w->deferred[--w->n_def];
            continue;
        }
        int progressed = sweep_games(e, w, 0);
        w->rr++;
        if (!progressed) {
            int32_t idx;
            /* free-slot check FIRST: a reservation must never be wasted */
            if (w->n_free > 0 && reserve_game_index(e, &idx)
                && start_game(e, w, idx)) {
                progressed = 1;
            } else {
                break;   /* no work anywhere: seal what we have */
            }
        }
    }
    if (w->n_buf > 0)
        w->sealed = 1;
    if (w->sealed) {
        for (;;) {
            int progressed = sweep_games(e, w, 1);
            w->rr++;
            if (!progressed)
                break;
        }
    }
}

static void deliver_results(SpEngine *e, const SpResEntry *res, int n) {
    double logp[SP_REQ_MAX_LEGAL], wdl[3];
    for (int i = 0; i < n; i++) {
        const float *lp = e->batch.out_logp + res[i].logp_off;
        const float *wd = e->batch.out_wdl + res[i].wdl_off;
        for (int j = 0; j < res[i].n_legal; j++)
            logp[j] = lp[j];
        for (int k = 0; k < 3; k++)
            wdl[k] = wd[k];
        sp_deliver_node(res[i].game, res[i].node, logp, wdl);
    }
}

static void *worker_main(void *arg) {
    SpWorker *w = arg;
    SpEngine *e = w->engine;
    /* active_workers was set to n_workers by sp_engine_start before the
     * threads spawned (no startup race with the evaluator's predicate);
     * workers only decrement it at exit. */
    for (;;) {
        fill_phase(e, w);

        pthread_mutex_lock(&e->lock);
        /* Exit, don't post (R3) — checked under the lock so the
         * evaluator's active-workers denominator is always exact. */
        if (worker_exit_ok(e, w)) {
            e->active_workers--;
            w->active = 0;
            pthread_cond_broadcast(&e->cv_eval);
            pthread_mutex_unlock(&e->lock);
            return NULL;
        }
        /* Post EVERY round, sealed or empty — the design's "seal on full
         * or NO-WORK" (R1). An empty post is how a worker with nothing
         * to seal still counts toward the barrier: the epoch fires and
         * the broadcast wakes it to refill (its recycled slots came
         * back at collect time). */
        if (!w->posted) {
            w->posted = 1;
            e->posted_workers++;
            pthread_cond_signal(&e->cv_eval);
        }
        /* Wake per EPOCH (not per result): results may legitimately be
         * zero (empty buffer this round) while the worker still must
         * refill. */
        while (!w->epoch && !atomic_load(&e->failed) && !e->shutdown)
            pthread_cond_wait(&e->cv_workers, &e->lock);
        int failed = atomic_load(&e->failed) || e->shutdown;
        w->epoch = 0;
        int n = w->n_res;
        w->n_res = 0;               /* logically consumed; the static array
                                       stays valid until the next gather */
        pthread_mutex_unlock(&e->lock);

        if (failed)
            return NULL;            /* the run is over — leave quietly */
        if (n > 0)
            deliver_results(e, w->results, n);
    }
}

/* ------------------------------------------------------------------ */
/* Evaluator-side: gather, dispatch, distribute                        */
/* ------------------------------------------------------------------ */

/* Take every sealed buffer into the batch staging; append each request's
 * result entry to its owning worker. Called with the lock held — safe
 * because the evaluator only fires when every active worker posted (is
 * waiting), so no worker mutates its buffer concurrently. Clears the taken
 * buffers and the posted flags. Returns the batch size. */
static int gather_batch(SpEngine *e) {
    int size = 0, total = 0;
    e->batch.offsets[0] = 0;
    for (int wi = 0; wi < e->n_workers; wi++) {
        SpWorker *w = &e->workers[wi];
        {
            if (!w->sealed)
                continue;
            for (int r = 0; r < w->n_buf; r++) {
                const SpReq *req = &w->buf[r];
                memcpy(e->batch.planes + (size_t)size * NN_TOTAL_PLanes,
                       req->planes_u8, NN_TOTAL_PLanes);
                memcpy(e->batch.slots + total, req->slots,
                       (size_t)req->n_legal * sizeof *req->slots);
                e->batch.mover[size] = req->mover;
                e->batch.offsets[size + 1] = total + req->n_legal;

                SpWorker *dst = &e->workers[sp_game_owner(req->game)];
                assert(dst->n_res < SP_ENGINE_MAX_BUFFER);
                dst->results[dst->n_res++] = (SpResEntry){
                    .game = req->game, .node = req->node,
                    .logp_off = total, .wdl_off = size * 3,
                    .n_legal = req->n_legal,
                };
                total += req->n_legal;
                size++;
            }
            w->n_buf = 0;
            w->sealed = 0;
        }
        w->posted = 0;
    }
    e->posted_workers = 0;
    e->batch.size = size;
    e->batch.total_legal = total;
    e->epochs++;
    e->evals += size;
    if (size > e->max_batch)
        e->max_batch = size;
    return size;
}

/* Drop undelivered result entries (engine failure path — they point at
 * staging that will never be filled). */
static void discard_results(SpEngine *e) {
    for (int wi = 0; wi < e->n_workers; wi++) {
        e->workers[wi].n_res = 0;
    }
}

/* The evaluator loop, run by the sp_engine_wait() CALLER between epochs —
 * there is no dedicated evaluator thread. Workers and the evaluator never
 * compute simultaneously in the lock-step epoch (every game is parked
 * while the batch runs), so reserving a thread that idles through the
 * fill phase wastes one filler; the caller plays the role instead and all
 * cfg.n_workers threads fill. The GIL is re-acquired per batch inside
 * ev.batch (the wait caller released it via the cffi boundary). */
static int run_evaluator_loop(SpEngine *e) {
    if (e->ev.init && e->ev.init(e->ev.ctx) != 0) {
        pthread_mutex_lock(&e->lock);
        atomic_store(&e->failed, 1);
        discard_results(e);
        pthread_cond_broadcast(&e->cv_workers);
        pthread_mutex_unlock(&e->lock);
        return -1;
    }

    pthread_mutex_lock(&e->lock);
    for (;;) {
        while (e->posted_workers < e->active_workers && e->active_workers > 0
               && !atomic_load(&e->failed) && !e->shutdown)
            pthread_cond_wait(&e->cv_eval, &e->lock);
        if (e->active_workers == 0 || atomic_load(&e->failed) || e->shutdown)
            break;

        gather_batch(e);
        pthread_mutex_unlock(&e->lock);

        int rc = 0;
        if (e->batch.size > 0)
            rc = e->ev.batch(e->ev.ctx, &e->batch);   /* GIL lives inside */

        pthread_mutex_lock(&e->lock);
        if (rc != 0) {
            atomic_store(&e->failed, 1);
            discard_results(e);
            pthread_cond_broadcast(&e->cv_workers);
            break;
        }
        for (int wi = 0; wi < e->n_workers; wi++)
            e->workers[wi].epoch = 1;    /* wake for refill/deliver/exit */
        pthread_cond_broadcast(&e->cv_workers);
    }
    pthread_mutex_unlock(&e->lock);
    return atomic_load(&e->failed) ? -1 : 0;
}

/* ------------------------------------------------------------------ */
/* Deterministic-interleave mode (R6/L3): one thread, fixed order      */
/* ------------------------------------------------------------------ */

static void run_interleave(SpEngine *e) {
    if (e->ev.init && e->ev.init(e->ev.ctx) != 0) {
        atomic_store(&e->failed, 1);
        return;
    }
    for (;;) {
        for (int wi = 0; wi < e->n_workers; wi++)
            fill_phase(e, &e->workers[wi]);

        int all_done = 1;
        for (int wi = 0; wi < e->n_workers; wi++) {
            SpWorker *w = &e->workers[wi];
            if (!worker_exit_ok(e, w))
                all_done = 0;
        }
        if (all_done)
            break;   /* budget exhausted, nothing in flight anywhere */

        pthread_mutex_lock(&e->lock);
        gather_batch(e);
        pthread_mutex_unlock(&e->lock);

        if (e->batch.size > 0 && e->ev.batch(e->ev.ctx, &e->batch) != 0) {
            atomic_store(&e->failed, 1);
            discard_results(e);
            break;
        }
        for (int wi = 0; wi < e->n_workers; wi++) {
            SpWorker *w = &e->workers[wi];
            int n = w->n_res;
            w->n_res = 0;
            deliver_results(e, w->results, n);
        }
    }
    /* NOTE: the evaluator's free hook runs ONCE, in sp_engine_destroy —
     * calling it here too double-frees its context (py_eval_free frees
     * the strdup'd strings and the ctx itself). */
}

/* ------------------------------------------------------------------ */
/* Lifecycle                                                           */
/* ------------------------------------------------------------------ */

SpEngine *sp_engine_init(void *storage, size_t storage_size,
                         const SpEngineConfig *cfg, const SpEvaluator *ev) {
    /* Everything is embedded in the caller's static block (deviation 9):
     * the game pool, worker buffers, deferred lists, results, batch
     * staging. NO allocation happens here or ever —
     * init only wires pointers and resets scalars. The bulk (pool slots,
     * staging) is NOT touched, so untouched capacity commits no physical
     * memory. */
    assert(storage != NULL);
    assert(storage_size >= SP_ENGINE_STATIC_SIZE);
    assert(((uintptr_t)storage & 63u) == 0);
    assert(cfg->n_workers >= 1 && cfg->n_workers <= SP_ENGINE_MAX_WORKERS);
    assert(cfg->buffer_requests >= 1
           && cfg->buffer_requests <= SP_ENGINE_MAX_BUFFER);
    assert(cfg->game_budget >= 0);   /* cumulative — NOT pool-bounded */
    assert(ev && ev->batch);
    _Static_assert(sizeof(SpEngine) <= SP_ENGINE_STATIC_SIZE,
                   "engine outgrew SP_ENGINE_STATIC_SIZE");

    SpEngine *e = storage;
    /* Selective reset — NEVER memset the whole struct: it would commit
     * every pool/staging page (gigabytes) instead of leaving them lazily
     * mapped. Everything below is scalars and small arrays; the bulk
     * (pool, staging content) is written before it is ever read. */
    e->active_workers = 0;
    e->posted_workers = 0;
    atomic_init(&e->failed, 0);
    e->shutdown = 0;
    e->running = 0;

    e->cfg = *cfg;
    e->ev = *ev;
    e->n_workers = cfg->n_workers;
    atomic_init(&e->games_started, 0);
    atomic_init(&e->n_done, 0);
    e->evals = 0;
    e->epochs = 0;
    e->max_batch = 0;
    pthread_mutex_init(&e->lock, NULL);
    pthread_cond_init(&e->cv_eval, NULL);
    pthread_cond_init(&e->cv_workers, NULL);

    for (int wi = 0; wi < e->n_workers; wi++) {
        SpWorker *w = &e->workers[wi];
        w->engine = e;
        w->n_games = 0;
        w->rr = 0;
        w->n_free = SP_ENGINE_MAX_OUTSTANDING;
        for (int s = 0; s < SP_ENGINE_MAX_OUTSTANDING; s++)
            w->free_slots[s] = (SpGame *)(void *)&e->pool[wi][s][0];
        w->n_buf = 0;
        w->sealed = 0;
        w->posted = 0;
        w->n_def = 0;
        w->n_res = 0;
        w->epoch = 0;
        w->started = 0;
        w->active = 0;
        w->dump_fd = -1;
    }

    size_t cap = (size_t)e->n_workers * (size_t)cfg->buffer_requests;
    e->batch.size = 0;
    e->batch.total_legal = 0;
    e->batch.planes  = e->st_planes;
    e->batch.slots   = e->st_slots;
    e->batch.offsets = e->st_offsets;
    e->batch.mover   = e->st_mover;
    e->batch.out_logp = e->st_logp;
    e->batch.out_wdl  = e->st_wdl;
    (void)cap;
    return e;
}

int sp_engine_start(SpEngine *e) {
    assert(!e->running);
    e->running = 1;
    /* Open the per-worker shard files BEFORE any thread runs (both
     * modes): an unopenable dump fails the run at startup, loudly. */
    if (e->cfg.record_dir[0]) {
        for (int wi = 0; wi < e->n_workers; wi++) {
            e->workers[wi].dump_fd =
                sp_dump_open_shard(e->cfg.record_dir, wi, e->cfg.unit_seed);
            if (e->workers[wi].dump_fd < 0) {
                for (int wj = 0; wj < wi; wj++) {
                    close(e->workers[wj].dump_fd);
                    e->workers[wj].dump_fd = -1;
                }
                atomic_store(&e->failed, 1);
                e->running = 0;
                return -1;
            }
        }
    }
    if (e->cfg.interleave)
        return 0;   /* the run happens inside sp_engine_wait */
    /* The epoch fire/exit predicate counts down active_workers; set it
     * BEFORE spawning so a worker can never observe a transient 0. */
    e->active_workers = e->n_workers;
    for (int wi = 0; wi < e->n_workers; wi++) {
        SpWorker *w = &e->workers[wi];
        w->started = 1;
        if (pthread_create(&w->thread, NULL, worker_main, w) != 0) {
            w->started = 0;
            atomic_store(&e->failed, 1);
            /* wake everyone so wait() returns; the failed flag stops them */
            pthread_mutex_lock(&e->lock);
            e->active_workers = 0;
            pthread_cond_broadcast(&e->cv_workers);
            pthread_cond_broadcast(&e->cv_eval);
            pthread_mutex_unlock(&e->lock);
            return -1;
        }
    }
    return 0;
}

void sp_engine_wait(SpEngine *e) {
    if (e->cfg.interleave) {
        run_interleave(e);
    } else {
        /* The CALLER plays the evaluator between epochs (see
         * run_evaluator_loop): all cfg.n_workers spawned threads are
         * workers, and they fill while the caller idles between batches. */
        run_evaluator_loop(e);
        for (int wi = 0; wi < e->n_workers; wi++) {
            SpWorker *w = &e->workers[wi];
            if (w->started) {
                pthread_join(w->thread, NULL);
                w->started = 0;
            }
        }
    }
    /* Shard files (both paths): close — no fsync; durability is the
     * caller's concern (3.4 / HF backup). */
    for (int wi = 0; wi < e->n_workers; wi++) {
        if (e->workers[wi].dump_fd >= 0) {
            close(e->workers[wi].dump_fd);
            e->workers[wi].dump_fd = -1;
        }
    }
}

int sp_engine_error(const SpEngine *e) { return atomic_load(&e->failed); }

int sp_engine_games_started(const SpEngine *e) {
    return atomic_load(&e->games_started);
}
int sp_engine_games_done(const SpEngine *e) {
    return atomic_load(&e->n_done);
}
long sp_engine_evals(const SpEngine *e)     { return e->evals; }
long sp_engine_epochs(const SpEngine *e)    { return e->epochs; }
int  sp_engine_max_batch(const SpEngine *e) { return e->max_batch; }

size_t sp_engine_static_size(void) { return SP_ENGINE_STATIC_SIZE; }

void sp_engine_destroy(SpEngine *e) {
    if (!e) return;
    assert(!e->running || e->workers[0].started == 0);  /* wait() first */
    /* The evaluator's free hook fires exactly once, HERE (the run loops
     * never call it — see run_interleave). The storage block is the
     * CALLER's — nothing else to free; finished games' records live in
     * the shard files, their slots were recycled at collect time. */
    if (e->ev.free_fn)
        e->ev.free_fn(e->ev.ctx);
    pthread_mutex_destroy(&e->lock);
    pthread_cond_destroy(&e->cv_eval);
    pthread_cond_destroy(&e->cv_workers);
}
