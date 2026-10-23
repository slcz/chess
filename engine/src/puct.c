/* puct.c — Phase 1/2 PUCT MCTS + Phase 3a multi-game batching (chess_puct.h). */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include "chess_puct.h"
#include "chess_rng.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* A walker = one concurrent descent. It owns a scratch State copied fresh
 * from the game root at every sim (no rewind: the copy IS the reset); the
 * copy doubles as the parked leaf's position at delivery, where only the
 * side to move is read (for the policy-slot gather). A walker parks by
 * claiming an eval row and flagging its leaf PUCT_SLEEPING so compaction
 * keeps its idx current. A parked leaf is expanded on the spot, so no two
 * walkers ever park on one node. `dead` is the only other per-walker
 * life-cycle flag. */
typedef struct PuctWalker {
    State      st;
    NnHistory  hist;
    uint32_t   idx;      /* current leaf; remapped by compaction */
    int        row;      /* staged row while parked, -1 while walking */
    bool       dead;
} PuctWalker;

enum { WALKERS_DEFAULT = 8 };   /* walkers per game unless puct_set_walkers */

/* Bank life cycle: the waker stages rows (FREE→QUEUED via zone_submit);
 * the io thread pushes it (H2D + async compute launch, QUEUED→IN_QUEUE),
 * enqueues its D2H once the peer bank is pushed too — or earlier if the
 * waker flags it flush (starved: no more rows will come until this bank
 * delivers) — (IN_QUEUE→PULLED), blocks on the D2H event and publishes
 * the results (PULLED→READY). Delivery by the waker returns the bank to
 * FREE. A bank may only be staged into or re-submitted when it is FREE
 * again — the in-flight states are NOT free. Pushes always precede pulls,
 * so the copy FIFO runs H2D(A) H2D(B) D2H(A) D2H(B): every H2D unblocked,
 * the two forwards back to back. */
#ifdef __EMSCRIPTEN__
/* The web build is single-threaded and synchronous: the zone has no
 * concurrent caller, so the landing-zone API takes no lock there. The
 * native build's io thread does, and keeps the real mutex. */
#define ZONE_LOCK()   ((void)0)
#define ZONE_UNLOCK() ((void)0)
#else
#define ZONE_LOCK()   pthread_mutex_lock(&Z.mu)
#define ZONE_UNLOCK() pthread_mutex_unlock(&Z.mu)
#endif

#define BANK_FREE 0
#define BANK_QUEUED 1
#define BANK_IN_QUEUE 2
#define BANK_PULLED 3
#define BANK_READY 4

/* Permanent row storage: ONE mapping, partitioned into the four arrays ×
 * two banks, sized once by puct_zone_init (pages materialize on touch,
 * so a small cap costs nothing). The nn seam (puct_zone_set_buffers)
 * can instead point a bank's pointers at pinned tensors — this storage
 * is then simply bypassed, never freed. */
static void *z_mem;        /* also the zone's initialized flag */
static int    z_rows;      /* rows per bank */

/* One row's storage: planes + logp + wdl floats, then the mask bytes. */
#define ZONE_ROW_BYTES \
    (((size_t)NN_TOTAL + PUCT_POLICY_SIZE + 3) * sizeof(float) \
     + PUCT_POLICY_SIZE)

/* The landing zone — exactly one per system, two banks (double buffer).
 * The walker thread stages rows into the active bank while the io thread
 * evaluates the other; a bank is shipped asynchronously (queued) and its
 * results wait for delivery (ready). A walker claims a row when it parks
 * and remembers its global row id (bank = row / cap), so delivery needs no
 * row→game table. */
static struct {
    struct {
        float  *planes;   /* cap × NN_TOTAL, into z_planes or pinned */
        uint8_t *mask;    /* cap × PUCT_POLICY_SIZE */
        float  *logp;     /* cap × PUCT_POLICY_SIZE */
        float  *wdl;      /* cap × 3 */
        int     used;     /* rows staged; frozen once the bank leaves
                             FREE (only the active bank is staged into) */
        int     state;    /* BANK_FREE .. BANK_READY, guarded by Z.mu */
        bool    flush;    /* waker is stuck on this bank: pull it now
                             instead of waiting for the peer's push */
    } banks[PUCT_BANKS];
    int     active;       /* bank the walkers stage into (waker only) */
    PuctEvalFn eval;
    void   *eval_ctx;
    pthread_t       io_thread;
    pthread_mutex_t mu;
    pthread_cond_t  io_cv, walker_cv;
    bool            io_quit;
} Z;

struct PuctGame {
    State      state;
    NnHistory  hist;
    PuctNode  *nodes;
    size_t     block_bytes;
    size_t     cap;      /* node slots in the block */
    uint32_t   used;
    uint32_t   root;
    float      cpuct;
    uint64_t   seed0;      /* this game's key: stream = f(seed0, think_no),
                              never drawn from a shared stream, so a
                              transcript cannot depend on task scheduling */
    uint32_t   think_no;   /* fresh thinks this game has started */
    uint64_t   rng;        /* per-game sampling stream, re-keyed at every
                              fresh think (walker threads never touch the
                              process stream while sampling) */
    uint32_t   oom;
    uint32_t   compactions;
    PuctWalker *walkers;    /* parked idx fields are the ground truth;
                               PUCT_SLEEPING gates the compaction scan */
    int        n_walkers;
    bool       think_done; /* completed (awaiting the driver's play) or
                              never-thought; see puct_walk */
    bool       suspended;  /* skip in walks: the game waits on something
                              external (e.g. the opponent's UCI reply) */
    /* mux step status: written only by the task walking this game (or
     * by the driver between rounds, while every task sleeps) — never
     * under a lock; the round barrier orders the two */
    uint8_t    step_done;  /* this step's think is complete */
    uint8_t    stuck;      /* parked on an in-flight bank until delivery */
    uint64_t   done_key;   /* position key at completion; 0 = fresh game */
    int        sims_done;
    uint32_t   parks, terminal_hits;
    uint32_t   pending_parks;  /* parks that co-slept on a pending row */
};

/* Parked walkers point at nodes by plain index; compaction rewrites the
 * matching fields when it moves a flagged node. A stale idx on an awake
 * walker is harmless — it is overwritten at the next park. */
static void sleep_remap(PuctGame *g, uint32_t old_idx, uint32_t new_idx) {
    for (int i = 0; i < g->n_walkers; i++)
        if (g->walkers[i].idx == old_idx)
            g->walkers[i].idx = new_idx;
}

/* Drop own park; clear the node's flag only when the last sleeper leaves
 * (co-sleepers may share a pending node and still need remapping). */
static void sleep_clear(PuctGame *g, PuctWalker *self) {
    for (int i = 0; i < g->n_walkers; i++) {
        PuctWalker *w = &g->walkers[i];
        if (w != self && w->row >= 0 && w->idx == self->idx)
            return;
    }
    g->nodes[self->idx].flags &= ~(uint8_t)PUCT_SLEEPING;
}

#ifdef __EMSCRIPTEN__
/* wasm has no mmap (Emscripten's returns ENOSYS): the block is plain
 * heap. The engine treats a block as opaque private storage, so the only
 * contract is "a fresh writable region that can be freed" — exactly what
 * the allocator gives. unmap_block keeps the free path symmetric for
 * both builds. */
static void *map_block(size_t bytes) {
    return aligned_alloc(8, (bytes + 7) & ~(size_t)7);
}
static void unmap_block(void *p, size_t bytes) { (void)bytes; free(p); }
#else
static void *map_block(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}
static void unmap_block(void *p, size_t bytes) { (void)bytes; munmap(p, bytes); }
#endif

/* Packing stage: fresh block, copy the subtree reachable from the root,
 * remapping parent/first_child. The new block doubles as the BFS queue:
 * a copied node's first_child field still points into the old block, so
 * draining new[hd] reads its old children block, bulk-copies it to the
 * tail and rewires first_child + the children's parent. Dead subtrees of
 * earlier moves are dropped; the old block is munmap'd.
 * Every copied node's sleep chain is remapped, so parked walkers (and any
 * in-flight expand) read their new index straight from their own field. */
static bool compact(PuctGame *g) {
    PuctNode *nb = map_block(g->block_bytes);
    if (!nb)
        return false;
    const PuctNode *ob = g->nodes;
    nb[0].n_children = 1;         /* the sentinel's one child IS the root: */
    nb[0].first_child = g->root;  /* the BFS packs it like any other node */
    size_t hd = 0, tl = 1;
    while (hd < tl) {
        uint32_t old_fc = nb[hd].first_child;
        uint32_t n = nb[hd].n_children;
        if (n > 0) {
            memcpy(&nb[tl], &ob[old_fc], (size_t)n * sizeof(PuctNode));
            nb[hd].first_child = (uint32_t)tl;
            for (uint32_t i = 0; i < n; i++) {
                if (nb[tl + i].flags & PUCT_SLEEPING)
                    sleep_remap(g, old_fc + i, (uint32_t)tl + i);
                nb[tl + i].parent = (uint32_t)hd;
            }
            tl += n;
        }
        hd++;
    }
    unmap_block(g->nodes, g->block_bytes);
    g->nodes = nb;
    g->used = (uint32_t)tl;
    g->root = 1;
    g->compactions++;
    return true;
}

static uint32_t alloc_nodes(PuctGame *g, int n) {
    if ((size_t)g->used + (size_t)n > g->cap &&
        (!compact(g) || (size_t)g->used + (size_t)n > g->cap)) {
        g->oom++;
        return 0;
    }
    uint32_t idx = g->used;
    memset(&g->nodes[idx], 0, (size_t)n * sizeof(PuctNode));
    g->used += (uint32_t)n;
    return idx;
}

static void set_terminal(PuctNode *n, int term) {
    n->flags |= PUCT_TERMINAL;
    if (term == TERM_CHECKMATE) {
        n->wdl[0] = 0; n->wdl[1] = 0; n->wdl[2] = 1;
    } else {
        n->wdl[0] = 0; n->wdl[1] = 1; n->wdl[2] = 0;
    }
}

/* ---- landing zone -------------------------------------------------------- */

static void *io_main(void *arg);
static void walker_deliver(PuctGame *g, PuctWalker *w);

/* Default evaluator until puct_set_eval swaps in a real one: the whole
 * "forward" is uniform logp + draw. */
static void dummy_eval(void *ctx, PuctEvalOp op, int bank, int n,
                       const float *planes, const uint8_t *masks,
                       float *logp, float *wdl) {
    (void)ctx; (void)bank; (void)planes; (void)masks;
    if (op != PUCT_EVAL_PUSH)
        return;
    for (int r = 0; r < n; r++) {
        memset(logp + (size_t)r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = wdl + (size_t)r * 3;
        w[0] = 0; w[1] = 1; w[2] = 0;
    }
}

void puct_zone_init(int max_rows) {
    assert(!z_mem);      /* one-time setup: the zone is process-global */
    if (max_rows < 1)
        max_rows = 1;
    size_t rows = (size_t)max_rows;
    z_mem = map_block((size_t)PUCT_BANKS * rows * ZONE_ROW_BYTES);
    assert(z_mem);       /* row storage is the pipeline's life support */
    float *planes = z_mem;
    float *logp = planes + (size_t)PUCT_BANKS * rows * NN_TOTAL;
    float *wdl = logp + (size_t)PUCT_BANKS * rows * PUCT_POLICY_SIZE;
    uint8_t *mask = (uint8_t *)(wdl + (size_t)PUCT_BANKS * rows * 3);
    for (int b = 0; b < PUCT_BANKS; b++)
        puct_zone_set_buffers(b,
                              planes + (size_t)b * rows * NN_TOTAL,
                              mask + (size_t)b * rows * PUCT_POLICY_SIZE,
                              logp + (size_t)b * rows * PUCT_POLICY_SIZE,
                              wdl + (size_t)b * rows * 3);
    z_rows = max_rows;
    if (!Z.eval)
        Z.eval = dummy_eval;   /* thinkable out of the box */
    pthread_mutex_init(&Z.mu, NULL);
    pthread_cond_init(&Z.io_cv, NULL);
    pthread_cond_init(&Z.walker_cv, NULL);
#ifndef __EMSCRIPTEN__
    assert(pthread_create(&Z.io_thread, NULL, io_main, &Z) == 0);
#endif
}

void puct_zone_set_buffers(int bank, float *planes, uint8_t *mask,
                           float *logp, float *wdl) {
    assert(bank >= 0 && bank < PUCT_BANKS);
    Z.banks[bank].planes = planes;
    Z.banks[bank].mask = mask;
    Z.banks[bank].logp = logp;
    Z.banks[bank].wdl = wdl;
}

void puct_set_eval(PuctEvalFn fn, void *ctx) {
    Z.eval = fn;
    Z.eval_ctx = ctx;
}

#ifndef __EMSCRIPTEN__
/* Sleep condition for the io thread: work it can make progress on. A bank
 * IN_QUEUE without flush is NOT pending — its D2H deliberately waits for
 * the peer's push so the copy FIFO stays H2D(A) H2D(B) D2H(A) D2H(B). */
static bool io_pending(void) {
    for (int b = 0; b < PUCT_BANKS; b++) {
        int st = Z.banks[b].state;
        if (st == BANK_QUEUED || st == BANK_PULLED ||
            (st == BANK_IN_QUEUE && Z.banks[b].flush))
            return true;
    }
    return false;
}
#endif  /* !__EMSCRIPTEN__ */

#ifndef __EMSCRIPTEN__
static void bank_eval(PuctEvalOp op, int b, int n) {
    assert(Z.eval);
    Z.eval(Z.eval_ctx, op, b, n, Z.banks[b].planes, Z.banks[b].mask,
           Z.banks[b].logp, Z.banks[b].wdl);
}

/* Drive every bank that can make progress through its next phase: push
 * queued banks, pull pushed ones (once the peer is pushed too, or flush
 * is flagged — the copy FIFO keeps H2D(A) H2D(B) D2H(A) D2H(B)), then
 * wait+publish pulled ones. Mutex HELD on entry (released around each
 * evaluator call). Native: called by the io thread's loop. Inline
 * (__EMSCRIPTEN__, below): called directly by the waker — no io thread.
 * Strict hand-off either way — it never touches the trees or the
 * walkers; delivering the results is the waker's job (zone_pump). */
static void io_drive(void) {
    /* 1. push: H2D + launch compute for every queued bank (async) */
    for (int b = 0; b < PUCT_BANKS; b++) {
        if (Z.banks[b].state != BANK_QUEUED)
            continue;
        Z.banks[b].state = BANK_IN_QUEUE;
        int n = Z.banks[b].used;   /* frozen since submit: only the
                                      active (FREE) bank is staged */
        ZONE_UNLOCK();
        bank_eval(PUCT_EVAL_PUSH, b, n);
        ZONE_LOCK();
        pthread_cond_broadcast(&Z.walker_cv);   /* wake zone_wait so a
                                    starved waker can flag flush */
    }
    /* 2. pull: enqueue D2H for pushed banks — but only once the peer
     * is pushed too (or the waker flagged us flush), keeping the copy
     * FIFO H2D(A) H2D(B) D2H(A) D2H(B) */
    for (int b = 0; b < PUCT_BANKS; b++) {
        if (Z.banks[b].state != BANK_IN_QUEUE)
            continue;
        if (!Z.banks[b].flush && Z.banks[b ^ 1].state == BANK_FREE)
            continue;      /* peer not pushed yet: keep the FIFO order */
        Z.banks[b].state = BANK_PULLED;
        Z.banks[b].flush = false;
        int n = Z.banks[b].used;
        ZONE_UNLOCK();
        bank_eval(PUCT_EVAL_PULL, b, n);
        ZONE_LOCK();
    }
    /* 3. publish: block on each pulled bank's D2H, in order */
    for (int b = 0; b < PUCT_BANKS; b++) {
        if (Z.banks[b].state != BANK_PULLED)
            continue;
        int n = Z.banks[b].used;
        ZONE_UNLOCK();
        bank_eval(PUCT_EVAL_WAIT, b, n);
        ZONE_LOCK();
        Z.banks[b].state = BANK_READY;
        pthread_cond_broadcast(&Z.walker_cv);
    }
}

/* io thread (native builds only): wait for work, then io_drive it. The
 * wasm build has no io thread at all: the think loop hands QUEUED banks
 * to JS (puct_think_start below) and the page runs the evaluator. */
static void *io_main(void *arg) {
    (void)arg;
    ZONE_LOCK();
    for (;;) {
        while (!Z.io_quit && !io_pending())
            pthread_cond_wait(&Z.io_cv, &Z.mu);
        if (Z.io_quit)
            break;
        io_drive();
    }
    ZONE_UNLOCK();
    return NULL;
}
#endif  /* !__EMSCRIPTEN__ */

static bool bank_busy(int b) {
    ZONE_LOCK();
    bool busy = Z.banks[b].state != BANK_FREE;
    ZONE_UNLOCK();
    return busy;
}

/* Submit a bank for evaluation; its rows stay staged for delivery.
 * Native: the io thread takes it from here. Wasm: nothing runs — the
 * bank sits QUEUED until puct_think_start reports it to JS, the page
 * runs the forward, and puct_zone_publish marks the results ready. */
static void zone_submit(int b) {
    ZONE_LOCK();
    assert(Z.banks[b].used > 0 && Z.banks[b].state == BANK_FREE);
    Z.banks[b].state = BANK_QUEUED;
    pthread_cond_signal(&Z.io_cv);
    ZONE_UNLOCK();
}

/* Block until bank b's results have arrived. A pushed-but-unpulled bank
 * means the waker is starved (its games all park on it) — flag it flush so
 * the io thread pulls it instead of waiting for the peer's push. */
static void zone_wait(int b) {
    ZONE_LOCK();
    while (Z.banks[b].state != BANK_READY) {
        if (Z.banks[b].state == BANK_IN_QUEUE && !Z.banks[b].flush) {
            Z.banks[b].flush = true;
            pthread_cond_signal(&Z.io_cv);
        }
        pthread_cond_wait(&Z.walker_cv, &Z.mu);
    }
    ZONE_UNLOCK();
}

/* Deliver bank b's rows into their games and free the bank (it becomes
 * the active one). bank_busy must have been true — the caller waited. */
static int deliver_bank(int b, PuctGame **games, int n_games) {
    for (int i = 0; i < n_games; i++) {
        PuctGame *g = games[i];
        for (int k = 0; k < g->n_walkers; k++) {
            PuctWalker *w = &g->walkers[k];
            if (w->row >= 0 && w->row / z_rows == b)
                walker_deliver(g, w);
        }
    }
    ZONE_LOCK();
    assert(Z.banks[b].state == BANK_READY);
    int rows = Z.banks[b].used;
    Z.banks[b].state = BANK_FREE;
    Z.banks[b].used = 0;
    Z.active = b;   /* under Z.mu like every zone-field write: the walk
                       tasks read it under the same lock while staging */
    ZONE_UNLOCK();
    return rows;
}

/* ---- game ---------------------------------------------------------------- */

/* Fresh game in place at the CURRENT state: history, tree and walkers
 * reset; the mmap'd block and the lifetime stats are kept. */
static void game_fresh(PuctGame *g) {
    nn_history_reset(&g->hist);
    nn_history_record(&g->hist, &g->state);
    memset(g->nodes, 0, 2 * sizeof(PuctNode));
    g->used = 2;
    g->root = 1;
    g->think_no = 0;
    g->think_done = true;
    g->done_key = 0;
    for (int i = 0; i < g->n_walkers; i++)
        g->walkers[i].row = -1;
}

PuctGame *puct_game_new(size_t block_bytes) {
    if (block_bytes == 0)
        block_bytes = PUCT_DEFAULT_BLOCK;
    PuctGame *g = calloc(1, sizeof *g);
    if (!g)
        return NULL;
    g->nodes = map_block(block_bytes);
    if (!g->nodes) {
        free(g);
        return NULL;
    }
    g->block_bytes = block_bytes;
    g->cap = block_bytes / sizeof(PuctNode);
    g->cpuct = 1.25f;
    g->seed0 = rng_next();     /* single-threaded here; the mux re-seeds */
    state_init_startpos(&g->state);
    if (g->cap < 3) {
        unmap_block(g->nodes, g->block_bytes);
        free(g);
        return NULL;
    }
    game_fresh(g);   /* walkers stay NULL until set (or defaulted in walk) */
    return g;
}

void puct_game_free(PuctGame *g) {
    if (g) {
        unmap_block(g->nodes, g->block_bytes);
        free(g->walkers);
        free(g);
    }
}

/* Fresh startpos game in place (see game_fresh). */
void puct_game_reset(PuctGame *g) {
    state_init_startpos(&g->state);
    game_fresh(g);
}

/* The one non-startpos override: replant the game at `fen` in place
 * (see game_fresh). Call with no walker in flight — e.g. right after
 * puct_mux_new, or to recycle a finished match game at its book
 * position. 0 = ok. */
int puct_game_set_fen(PuctGame *g, const char *fen) {
    State s;
    if (!fen || !*fen || fen_parse(&s, fen) != 0)
        return -1;
    g->state = s;
    game_fresh(g);
    return 0;
}

void puct_set_walkers(PuctGame *g, int n) {
    if (n < 1)
        n = 1;
    assert(!g->walkers);   /* set once, before the first think */
    g->walkers = calloc((size_t)n, sizeof *g->walkers);
    assert(g->walkers);    /* walker setup is a startup decision */
    for (int i = 0; i < n; i++)
        g->walkers[i].row = -1;   /* row 0 is a valid row id */
    g->n_walkers = n;
}

void puct_set_suspended(PuctGame *g, bool on) { g->suspended = on; }

/* Key a game's sampling stream: stream_t = f(seed, think t), so it depends
 * on the game and its own move number, never on when it ran. */
void puct_game_seed(PuctGame *g, uint64_t seed) {
    g->seed0 = seed ? seed : 1;
    g->think_no = 0;
    g->rng = g->seed0;
}


int chess_policy_slot(const State *s, Move m) {
    int frm = move_from(m), to = move_to(m);
    if (s->side == BLACK) {
        frm ^= 56;
        to ^= 56;
    }
    if (move_is_promo(m)) {
        int pt = move_promo_type(m);
        if (pt == QUEEN)
            return frm * 64 + to;
        static const int dirs[3] = {7, 8, 9};   /* cap-left, push, cap-right */
        int d = to - frm;
        for (int k = 0; k < 3; k++) {
            if (d == dirs[k] && frm >= 48 && frm <= 55)
                return 4096 + (frm - 48) * 9 + k * 3 + (pt - KNIGHT);
            if (d == -dirs[k] && frm >= 8 && frm <= 15)
                return 4168 + (frm - 8) * 9 + k * 3 + (pt - KNIGHT);
        }
        return -1;
    }
    return frm * 64 + to;
}

typedef double v4d __attribute__((vector_size(32)));   /* gcc and clang */

static int node_pi_bar(const PuctGame *g, const PuctNode *r,
                       uint16_t *moves, float *pi, int cap);

/* SEARCH (Grill et al. 2020 sec 4.2): the descending move is sampled from
 * pi-bar, the same regularized policy LEARN trains on and ACT plays from.
 * There is no deterministic PUCT path and no root noise: pi-bar carries the
 * prior and the search carries the exploration. */
static uint32_t select_child(PuctGame *g, const PuctNode *n) {
    uint16_t mv[MAX_MOVES];
    float pi[MAX_MOVES];
    if (n->n_children < 2)
        return n->first_child;
    int m = node_pi_bar(g, n, mv, pi, MAX_MOVES);
    if (m <= 1)
        return n->first_child;
    double w[MAX_MOVES];
    for (int i = 0; i < m; i++)
        w[i] = (double)pi[i];
    /* pi-bar lists children in order: the sampled index is the child
     * index. The draw comes from the game's own stream — concurrent
     * walker threads never contend on the process-global one. */
    return n->first_child + (uint32_t)sample_index_stream(&g->rng, w, m,
                                                          1.0);
}

/* Expand the childless leaf *idx on the spot: one child per legal move,
 * uniform placeholder priors (the real softmax arrives with the eval row).
 * The leaf must be flagged
 * PUCT_SLEEPING with *idx aliasing the parked walker's field, so a pack
 * inside the allocation keeps *idx current. */
static bool expand_uniform(PuctGame *g, uint32_t *idx, const MoveList *ml) {
    assert(ml->count > 0);
    uint32_t kids = alloc_nodes(g, ml->count);
    if (kids == 0)
        return false;
    float prior = 1.0f / (float)ml->count;
    for (int i = 0; i < ml->count; i++) {
        PuctNode *c = &g->nodes[kids + i];
        c->parent = *idx;
        c->move = ml->moves[i];
        c->prior = prior;
    }
    g->nodes[*idx].flags |= PUCT_PENDING;
    g->nodes[*idx].first_child = kids;
    g->nodes[*idx].n_children = (uint32_t)ml->count;
    return true;
}

/* Fill in the parked leaf's real priors: softmax over the row's logp,
 * gathered at each child's policy slot. chess_policy_slot reads only the
 * side to move, and w->st still holds the leaf position at delivery. */
static void fill_priors(PuctGame *g, uint32_t idx, const float *logp,
                        const State *st) {
    PuctNode *n = &g->nodes[idx];
    int slots[MAX_MOVES];
    double max = -INFINITY, sum = 0.0;
    for (uint32_t i = 0; i < n->n_children; i++) {
        slots[i] = chess_policy_slot(st, g->nodes[n->first_child + i].move);
        if (logp[slots[i]] > max)
            max = logp[slots[i]];
    }
    for (uint32_t i = 0; i < n->n_children; i++)
        sum += exp((double)logp[slots[i]] - max);
    for (uint32_t i = 0; i < n->n_children; i++)
        g->nodes[n->first_child + i].prior =
            (float)(exp((double)logp[slots[i]] - max) / sum);
    n->flags &= ~(uint8_t)PUCT_PENDING;
}

/* Backup at delivery or terminal arrival: W and one visit land on every
 * path node; nothing is counted during the descent. */
static void backup(PuctGame *g, uint32_t leaf, float v) {
    double vs = -(double)v;
    for (uint32_t i = leaf; i != 0; i = g->nodes[i].parent) {
        g->nodes[i].value_sum += vs;
        g->nodes[i].visits++;   /* the root is on this path too, so it needs
                                 * no separate increment */
        vs = -vs;
    }
}

/* One descent attempt: fresh State from the root, down the tree by pi-bar
 * sampling, ending in a terminal backup, or a park — the leaf is expanded
 * the moment we find it (uniform placeholder priors) and its eval row
 * staged; real priors and the value are filled in at delivery. The leaf
 * index is parked (flagged
 * and aliased to w->idx) BEFORE the expansion so a pack inside the
 * allocation keeps it current. Returns false ONLY when the park found
 * the active bank full (the walker stays live; the caller stops the
 * pass and ships the bank — in the threaded mux the other walk tasks
 * may have filled it since this pass began). */
static bool walker_step(PuctGame *g, PuctWalker *w) {
    w->st = g->state;
    w->hist = g->hist;
    uint32_t idx = g->root;
    while (g->nodes[idx].n_children > 0 &&
           !(g->nodes[idx].flags & PUCT_PENDING) &&
           w->st.ply + 2 < MAX_PLY) {
        uint32_t c = select_child(g, &g->nodes[idx]);
        make_move(&w->st, g->nodes[c].move);
        nn_history_record(&w->hist, &w->st);
        idx = c;
    }
    PuctNode *n = &g->nodes[idx];
    if (n->flags & PUCT_PENDING) {
        /* expanded but unevaluated: co-sleep on its in-flight row. Its
         * children are uniform-prior placeholders — descending through
         * them would spend budget on moves the policy never ranked. */
        w->idx = idx;
        w->row = -1;
        for (int i = 0; i < g->n_walkers; i++)
            if (g->walkers[i].row >= 0 && g->walkers[i].idx == idx) {
                w->row = g->walkers[i].row;
                break;
            }
        assert(w->row >= 0);
        g->parks++;
        g->pending_parks++;   /* shared in-flight row: no new eval request */
        return true;
    }
    if (n->n_children > 0) {
        /* ply wall at an evaluated interior node (children present, no row
         * in flight): no value will ever arrive — give the walker up,
         * like the OOM path */
        w->dead = true;
        return true;
    }
    int term;
    if (n->flags & PUCT_TERMINAL ||
        (term = state_termination(&w->st)) != TERM_PLAYING) {
        if (!(n->flags & PUCT_TERMINAL))
            set_terminal(n, term);   /* fresh leaf: record its wdl */
        backup(g, idx, n->wdl[0] - n->wdl[2]);
        g->parks++;
        g->terminal_hits++;
        g->sims_done++;
        return true;
    }
    /* park: reserve a row FIRST, then expand here and stage into it —
     * from now on the leaf has children, so later walkers descend
     * through it and park on fresh leaves; a park is always the sole
     * owner of its row. The reserve is the one zone write that races
     * (the mux's walk tasks all stage into the active bank), so
     * bank + used move as one section under Z.mu — the same lock the
     * io thread and the pump already use. Row CONTENT is written
     * outside it: rows are walker-disjoint, and the pump barrier (every
     * task asleep) orders them before the io thread reads. A failed
     * expand after a successful reserve (oom) orphans the row — one
     * garbage eval nobody claims, harmless. */
    ZONE_LOCK();
    int b = Z.active;
    if (Z.banks[b].used >= z_rows) {   /* a caller-sized zone must fit
                                        * every walker of a bank */
        ZONE_UNLOCK();
        return false;
    }
    int off = Z.banks[b].used++;
    ZONE_UNLOCK();
    w->idx = idx;
    g->nodes[w->idx].flags |= PUCT_SLEEPING;
    MoveList ml;
    gen_legal(&w->st, &ml);
    if (!expand_uniform(g, &w->idx, &ml)) {    /* may pack; w->idx follows */
        g->nodes[w->idx].flags &= ~(uint8_t)PUCT_SLEEPING;
        w->dead = true;
        return true;
    }
    w->row = b * z_rows + off;             /* global row id */
    nn_encode(&w->st, &w->hist, Z.banks[b].planes + (size_t)off * NN_TOTAL);
    uint8_t *m = Z.banks[b].mask + (size_t)off * PUCT_POLICY_SIZE;
    memset(m, 0, PUCT_POLICY_SIZE);
    for (int i = 0; i < ml.count; i++)
        m[chess_policy_slot(&w->st, ml.moves[i])] = 1;
    g->parks++;
    return true;
}

/* Resume a parked walker with its row's result: fill the leaf's real
 * priors (its children were created at park) and back the value up along
 * the path. The root's row fills priors only — the root has no parent. */
static void walker_deliver(PuctGame *g, PuctWalker *w) {
    int b = w->row / z_rows;
    int off = w->row - b * z_rows;
    assert(g->nodes[w->idx].n_children > 0);
    /* first delivery only: co-sleepers share the row */
    if (g->nodes[w->idx].flags & PUCT_PENDING)
        fill_priors(g, w->idx,
                    Z.banks[b].logp + (size_t)off * PUCT_POLICY_SIZE, &w->st);
    if (w->idx != g->root) {
        const float *wd = Z.banks[b].wdl + (size_t)off * 3;
        backup(g, w->idx, wd[0] - wd[2]);
        g->sims_done++;
    }
    sleep_clear(g, w);
    w->row = -1;
}

/* puct_think decomposed for the mux driver (both live in this file): one
 * walking pass — descend/park until all walkers are parked, the sims
 * target is met, or the active bank is full. The return is why the pass
 * stopped: DONE (think complete), STUCK (every walker parked or sims
 * met — only a delivery unblocks it) or FULL (live walkers remain but
 * the active bank has no rows — a pump flips it walkable again). */
enum { WALK_STUCK = 0, WALK_DONE = 1, WALK_FULL = 2 };

static int puct_walk(PuctGame *g, int sims) {
    if (g->suspended)
        return WALK_DONE; /* waiting on the outside: not done, not parked */
    if (g->think_done) {
        /* A completed think stays complete until the driver plays a move
         * — puct_play clears done_key, because a repetition cycle
         * returns the same poskey while still deserving a fresh think.
         * The mux re-walks every game on every pass, and re-searching an
         * un-played root would corrupt the per-think stats.
         * done_key == 0 marks a fresh, never-thought game — its first
         * walk starts the fresh think below. */
        if (g->done_key == state_poskey(&g->state))
            return WALK_DONE;
        g->think_done = false;              /* fresh think: per-move reset */
        g->sims_done = 0;
        g->parks = g->terminal_hits = g->pending_parks = 0;
        g->rng = g->seed0 + 0x9E3779B97F4A7C15ull * (++g->think_no);
        for (int i = 0; i < g->n_walkers; i++) {
            g->walkers[i].dead = false;
            g->walkers[i].row = -1;
        }
    }
    if (g->nodes[g->root].flags & PUCT_TERMINAL) {
        g->think_done = true;
        g->done_key = state_poskey(&g->state);
        return WALK_DONE;
    }
    if (!g->walkers)
        puct_set_walkers(g, WALKERS_DEFAULT);   /* unless set: 8 walkers */
    assert(z_mem);   /* the caller sized the zone (puct_zone_init /
                        puct_attach_nn) before the first think */
    for (;;) {
        bool any_live = false, any_parked = false, all_parked = true;
        for (int i = 0; i < g->n_walkers; i++) {
            const PuctWalker *w = &g->walkers[i];
            if (w->dead)
                continue;
            any_live = true;
            if (w->row >= 0) any_parked = true; else all_parked = false;
        }
        if (!any_live || (g->sims_done >= sims && !any_parked)) {
            g->think_done = true;
            g->done_key = state_poskey(&g->state);
            return WALK_DONE;
        }
        if (all_parked || g->sims_done >= sims)
            return WALK_STUCK;   /* batch staged (or needs its flush) */
        for (int i = 0; i < g->n_walkers; i++) {
            PuctWalker *w = &g->walkers[i];
            if (w->dead || w->row >= 0)
                continue;
            if (!walker_step(g, w))
                return WALK_FULL;    /* the park found the active bank full
                                        (another task may have filled it
                                        since this pass began): the caller
                                        ships it */
        }
    }
}

/* Move rows toward delivery after a walk pass left games parked. Invariant
 * on entry: the active bank is FREE (only zone_pump submits, and every
 * path below leaves active pointing at the bank it just freed), so the
 * rows staged this pass are either here — ship them — or nowhere: every
 * parked walker's row already sits in the in-flight peer — reclaim it.
 * Either way the walk resumes into a free bank and the next pump ships
 * what it staged. */
static int zone_pump(PuctGame **games, int n_games) {
    if (Z.banks[Z.active].used > 0) {
        int ship = Z.active;
        zone_submit(ship);
        ZONE_LOCK();
        Z.active = ship ^ 1;   /* staged rows ship: walk the peer bank */
        ZONE_UNLOCK();
        if (bank_busy(Z.active)) {       /* both banks were occupied */
            zone_wait(Z.active);
            return deliver_bank(Z.active, games, n_games);
        }
        return 0;
    }
    /* nothing stageable: parked rows must sit in the other, in-flight bank */
    assert(bank_busy(Z.active ^ 1));
    zone_wait(Z.active ^ 1);
    return deliver_bank(Z.active ^ 1, games, n_games);
}

bool puct_think_done(const PuctGame *g) { return g->think_done; }

/* True when the position is decided (mate/draw): a terminal root thinks
 * instantly and has no move to play — drivers stop asking here (playing
 * puct_best_move's 0 on a finished game would corrupt the state). */
bool puct_game_over(const PuctGame *g) {
    return state_termination((State *)&g->state) != TERM_PLAYING;
}

void puct_think(PuctGame *g, int sims) {
    uint64_t key0 = state_poskey(&g->state);
    /* puct_think always runs a fresh sims budget on the current position
     * (back-to-back calls keep searching the same tree); the done-guard of
     * puct_walk is the mux driver's contract, not this one. */
    g->think_done = true;
    g->done_key = 0;
    while (puct_walk(g, sims) != WALK_DONE) {
        PuctGame *gg = g;
        zone_pump(&gg, 1);
    }
    assert(state_poskey(&g->state) == key0);
}

/* Play m (must be legal): a root child keeps its subtree; any other legal
 * move — an opponent's reply in an eval game — gets a fresh root. */
int puct_play(PuctGame *g, Move m) {
    PuctNode *r = &g->nodes[g->root];
    uint32_t nr = 0;
    for (uint32_t i = 0; i < r->n_children; i++)
        if (g->nodes[r->first_child + i].move == m) {
            nr = r->first_child + i;
            break;
        }
    if (nr == 0)
        nr = alloc_nodes(g, 1);   /* may pack; play replaces the root anyway */
    if (nr == 0)
        return -1;
    make_move(&g->state, m);
    nn_history_record(&g->hist, &g->state);
    g->root = nr;
    g->nodes[nr].parent = 0;
    g->done_key = 0;   /* a played move invalidates the done-guard — a
                          repetition cycle returns the same poskey and
                          would otherwise skip the next think */
    return 0;
}

Move puct_best_move(const PuctGame *g) {
    const PuctNode *r = &g->nodes[g->root];
    uint32_t best = r->first_child, bv = 0;   /* ties keep the first child */
    for (uint32_t i = 0; i < r->n_children; i++) {
        uint32_t c = r->first_child + i;
        if (g->nodes[c].visits > bv) {
            bv = g->nodes[c].visits;
            best = c;
        }
    }
    return best ? g->nodes[best].move : 0;
}

uint32_t puct_oom_count(const PuctGame *g) { return g->oom; }

uint32_t puct_compact_count(const PuctGame *g) { return g->compactions; }

/* Per-think leaf-arrival telemetry (reset by the next fresh think): parks
 * counts every sim ending at a childless node — eval parks and terminal
 * arrivals; terminal_hits is the subset backed up from a terminal;
 * pending_parks is the subset that co-slept on another walker's in-flight
 * row (a virtual visit only — no new eval request was staged). */
void puct_park_stats(const PuctGame *g, uint32_t *parks,
                     uint32_t *terminal_hits, uint32_t *pending_parks) {
    if (parks)
        *parks = g->parks;
    if (terminal_hits)
        *terminal_hits = g->terminal_hits;
    if (pending_parks)
        *pending_parks = g->pending_parks;
}

const State *puct_state(const PuctGame *g) { return &g->state; }

int puct_root_children(const PuctGame *g, uint16_t *moves, uint32_t *visits,
                       double *q, float *prior) {
    const PuctNode *r = &g->nodes[g->root];
    for (uint32_t i = 0; i < r->n_children; i++) {
        const PuctNode *c = &g->nodes[r->first_child + i];
        if (moves)
            moves[i] = c->move;
        if (visits)
            visits[i] = c->visits;
        if (q)
            q[i] = c->visits ? c->value_sum / c->visits : 0.0;
        if (prior)
            prior[i] = c->prior;
    }
    return (int)r->n_children;
}

void puct_root_stats(const PuctGame *g, uint32_t *visits, double *value_sum) {
    const PuctNode *r = &g->nodes[g->root];
    if (visits)
        *visits = r->visits;
    if (value_sum)
        *value_sum = r->value_sum;
}

/* ---- multi-game multiplexer ----------------------------------------------
 * ONE mux per process, like the zone: mux_games[0..mux_games_max) is
 * allocated up front at the start position (games_max IS the memory
 * budget — one mmap'd node block per game) and handed to the caller
 * once; idle slots are idled via puct_set_suspended. puct_mux_step runs
 * in ROUNDS — open one, give every eligible game one puct_walk pass,
 * pump, repeat until every live game's think is complete, ending fully
 * delivered so the driver can play and recycle between steps. The scan
 * order is fixed: every offset is claimed every round, so order only
 * decides who stages first when the bank fills — the rest run as the
 * earlier games complete their thinks.
 *
 * Locking: the only mutexes are the zone's (Z.mu — banks and row
 * staging) and the pool's, which exists solely as pthread_cond_wait's
 * mandatory companion for the round handshake (open, drain, quit) —
 * touched once per task per round, never on the walk path. The walk
 * path is lock-free: games are claimed off an atomic scan cursor (each
 * game at most one pass per round, so its step_done/stuck bytes have
 * exactly one writer), and the round barrier orders those bytes against
 * the driver's resets. Walk threads are opt-in (PUCT_TASKS; none by
 * default): with none — and always on wasm — the driver runs the same
 * pull loop inline as the only task. */

static PuctGame **mux_games;
static int        mux_games_max;

static atomic_int  mux_scan;      /* round cursor: one offset per claim */
static atomic_bool mux_full;      /* a pass found the active bank full */

/* the walk-task pool — thread plumbing only */
static struct {
    pthread_t       *tids;
    int              n_tasks;     /* 0 = inline (wasm, or no PUCT_TASKS) */
    int              quit;
    int              sims;        /* the current step's budget */
    int              epoch;       /* bumps per round: tasks walk when
                                     behind; active/epoch under mu */
    int              active;      /* tasks still inside the round */
    pthread_mutex_t  mu;          /* the condvar's companion, nothing else */
    pthread_cond_t   cv;
} P;

/* This task's memory of the last round it ran — thread-local, so a
 * task needs no identity beyond the globals it was spawned against. */
static _Thread_local int t_epoch;

/* One task's round (and, with no tasks, the driver's): claim scan
 * offsets until dry, giving each eligible game one puct_walk pass. */
static void mux_pull(void) {
    for (;;) {
        int off = atomic_fetch_add_explicit(&mux_scan, 1,
                                            memory_order_relaxed);
        if (off >= mux_games_max ||
            atomic_load_explicit(&mux_full, memory_order_relaxed))
            return;              /* round dry, or the bank filled */
        PuctGame *g = mux_games[off];
        if (g->suspended || g->step_done || g->stuck)
            continue;
        switch (puct_walk(g, P.sims)) {
        case WALK_DONE:  g->step_done = 1; break;
        case WALK_STUCK: g->stuck = 1; break;
        default:    atomic_store_explicit(&mux_full, true,
                                          memory_order_relaxed);
        }
    }
}

static void *mux_task_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&P.mu);
    for (;;) {
        while (!P.quit && P.epoch == t_epoch)
            pthread_cond_wait(&P.cv, &P.mu);
        if (P.quit)
            break;
        t_epoch = P.epoch;
        pthread_mutex_unlock(&P.mu);
        mux_pull();
        pthread_mutex_lock(&P.mu);
        if (--P.active == 0)
            pthread_cond_broadcast(&P.cv);  /* round drained */
    }
    pthread_mutex_unlock(&P.mu);
    return NULL;
}

/* The fixed game array is handed to the caller ONCE; there are no
 * per-slot accessors. 0 = ok, -1 = a game failed to allocate (games is
 * then untouched). Size the zone first (puct_zone_init or
 * puct_attach_nn) — the mux does not. */
int puct_mux_new(int games_max, size_t block_bytes, PuctGame **games) {
    if (games_max < 1)
        games_max = 1;
    mux_games = calloc((size_t)games_max, sizeof *mux_games);
    assert(mux_games);           /* startup allocation */
    mux_games_max = games_max;
    int n = 0;    /* no walk tasks: the driver pulls inline; PUCT_TASKS
                     opts into threads (wasm never has any) */
#ifndef __EMSCRIPTEN__
    const char *env = getenv("PUCT_TASKS");
    if (env && *env) {
        int v = atoi(env);
        if (v >= 1)
            n = v;
    }
#endif
    P.tids = calloc((size_t)(n > 0 ? n : 1), sizeof *P.tids);
    assert(P.tids);
    pthread_mutex_init(&P.mu, NULL);
    pthread_cond_init(&P.cv, NULL);
    uint64_t base = rng_next();   /* one draw, before any task exists */
    for (int i = 0; i < games_max; i++) {
        mux_games[i] = puct_game_new(block_bytes);
        if (!mux_games[i]) {
            puct_mux_free();    /* clean: 0 tasks, mutex initialized */
            return -1;
        }
        puct_game_seed(mux_games[i], base ^ (uint64_t)(i + 1));
        games[i] = mux_games[i];
    }
    fprintf(stderr, "mux: %d walk task(s)\n", n);   /* provenance: the
                                    * shard's task count is a run property */
    for (int i = 0; i < n; i++)    /* n == 0: nothing spawns */
        assert(pthread_create(&P.tids[i], NULL, mux_task_main,
                              NULL) == 0);
    P.n_tasks = n;
    return 0;
}

void puct_mux_free(void) {
    pthread_mutex_lock(&P.mu);
    P.quit = 1;
    pthread_cond_broadcast(&P.cv);
    pthread_mutex_unlock(&P.mu);
    for (int i = 0; i < P.n_tasks; i++)
        pthread_join(P.tids[i], NULL);
    P.n_tasks = 0;
    P.quit = 0;
    P.epoch = 0;    /* fresh tasks (t_epoch 0) must sleep until a step */
    pthread_mutex_destroy(&P.mu);
    pthread_cond_destroy(&P.cv);
    for (int i = 0; i < mux_games_max; i++)
        puct_game_free(mux_games[i]);
    free(mux_games);
    mux_games = NULL;
    mux_games_max = 0;
    free(P.tids);
    P.tids = NULL;
}

void puct_mux_step(int sims) {
    assert(z_mem);   /* the caller sized the zone before stepping */
    for (int i = 0; i < mux_games_max; i++)
        mux_games[i]->step_done = mux_games[i]->stuck = 0;
    P.sims = sims;   /* tasks read it only after the round handshake */
    for (;;) {
        atomic_store_explicit(&mux_scan, 0, memory_order_relaxed);
        atomic_store_explicit(&mux_full, false, memory_order_relaxed);
        if (P.n_tasks > 0) {
            pthread_mutex_lock(&P.mu);
            P.active = P.n_tasks;
            P.epoch++;
            pthread_cond_broadcast(&P.cv);  /* open the round */
            while (P.active > 0)
                pthread_cond_wait(&P.cv, &P.mu);
            pthread_mutex_unlock(&P.mu);
        } else {
            mux_pull();    /* no threads: the driver is the only task */
        }
        bool all_done = true;
        for (int i = 0; i < mux_games_max; i++)
            if (!mux_games[i]->step_done && !mux_games[i]->suspended) {
                all_done = false;
                break;
            }
        if (all_done)
            break;             /* every live think complete */
        if (zone_pump(mux_games, mux_games_max) > 0)
            for (int i = 0; i < mux_games_max; i++)
                mux_games[i]->stuck = 0;  /* the delivery unparked every
                                    waiter on the bank; ones still waiting
                                    elsewhere re-flag on one cheap pass */
    }
    for (int b = 0; b < PUCT_BANKS; b++)
        assert(Z.banks[b].used == 0 && !bank_busy(b));
}

float puct_cpuct(const PuctGame *g) { return g->cpuct; }

/* pi-bar (Grill et al. 2020): the exact solution of
 * argmax_y [q^T y - lambda KL[pi_theta, y]], i.e.
 *     pi(a) = lambda*pi_theta(a) / (alpha - q(a)),
 * lambda = cpuct*sqrt(N)/(|A|+N), alpha the zero of sum pi = 1 by Newton
 * from the right. Unvisited children take the node's own value. */
static int node_pi_bar(const PuctGame *g, const PuctNode *r,
                       uint16_t *moves, float *pi, int cap) {
    int n = (int)r->n_children;
    if (n <= 0 || cap <= 0)
        return 0;
    if (n > cap)
        n = cap;
    double N = (double)r->visits;
    double qmax = -INFINITY, sum = 0.0;
    double lam = N > 0.0 ? (double)g->cpuct * sqrt(N)
                               / (N + (double)r->n_children) : 0.0;
    if (lam == 0.0) {
        /* nothing visited: the regularized policy is the prior itself */
        double ps = 0.0;
        for (int i = 0; i < n; i++)
            ps += (double)g->nodes[r->first_child + i].prior;
        for (int i = 0; i < n; i++) {
            const PuctNode *c = &g->nodes[r->first_child + i];
            moves[i] = c->move;
            pi[i] = (float)((double)c->prior / ps);
        }
        return n;
    }
    /* W is parent-view, children's q are mover-view: the fallback is the
     * node's own mean, negated */
    double v_node = N > 0.0 ? -(r->value_sum / N) : 0.0;
    /* lane-aligned so the Newton loop reads whole vectors */
    v4d q4[MAX_MOVES / 4 + 1], lp4[MAX_MOVES / 4 + 1];
    double *const q = (double *)q4, *const lp = (double *)lp4;
    int nv = (n + 3) / 4;
    for (int i = 0; i < nv * 4; i++) {
        if (i < n) {
            const PuctNode *c = &g->nodes[r->first_child + i];
            double qi = c->visits ? c->value_sum / (double)c->visits : v_node;
            q[i] = qi;
            lp[i] = lam * (double)c->prior;
            if (qi > qmax)
                qmax = qi;
        } else {
            /* zero weight kills the lane; q pads from a real child — a
             * 0.0 pad could NaN via 0*inf when a == 0 */
            q[i] = q[n - 1];
            lp[i] = 0.0;
        }
    }
    /* prop. 4(iii): sum pi <= 1 already at qmax + lam — safe start */
    double a = qmax + lam;
    for (int it = 0; it < 64; it++) {
        double f = -1.0, df = 0.0;
        /* Newton on sum lp/(a-q) = 1; one reciprocal feeds both f and df
         * (lp/(d*d) is lp*r*r) */
        v4d av = {a, a, a, a}, fv = {0.0, 0.0, 0.0, 0.0};
        v4d dfv = {0.0, 0.0, 0.0, 0.0};
        for (int k = 0; k < nv; k++) {
            v4d rr = 1.0 / (av - q4[k]);
            fv += lp4[k] * rr;
            dfv -= lp4[k] * rr * rr;
        }
        f += fv[0] + fv[1] + fv[2] + fv[3];
        df += dfv[0] + dfv[1] + dfv[2] + dfv[3];
        double step = f / df;
        double next = a - step;
        a = next > qmax ? next : 0.5 * (a + qmax);
        if (step < 0.0 ? -step < 1e-15 : step < 1e-15)
            break;
    }
    for (int i = 0; i < n; i++) {
        const PuctNode *c = &g->nodes[r->first_child + i];
        moves[i] = c->move;
        pi[i] = (float)(lp[i] / (a - q[i]));
        sum += pi[i];
    }
    for (int i = 0; i < n; i++)
        pi[i] = (float)((double)pi[i] / sum);
    return n;
}

int puct_learn_pi(const PuctGame *g, uint16_t *moves, float *pi, int cap) {
    return node_pi_bar(g, &g->nodes[g->root], moves, pi, cap);
}


#ifdef __EMSCRIPTEN__
/* ---- single-threaded eval stepping (the web build's evaluator seam) ----
 *
 * Walk passes run until the think completes (returns 1) or a bank is QUEUED
 * (returns 0) with its rows staged in *out and the walkers parked. JS runs
 * the net, then puct_zone_publish delivers the rows and unparks the walkers.
 * Nothing runs between calls, so parked walkers are simply safe. puct_think
 * is not exported in this build: it would block forever on a bank only JS
 * can fill. */
int puct_think_start(PuctGame *g, int sims, int *out) {
    assert(Z.banks[0].state == BANK_FREE);
    if (!puct_walk(g, sims)) {
        assert(Z.banks[0].used > 0);
        Z.banks[0].state = BANK_QUEUED;
        out[0] = Z.banks[0].used;
        out[1] = (int)(uintptr_t)Z.banks[0].planes;
        out[2] = (int)(uintptr_t)Z.banks[0].mask;
        out[3] = (int)(uintptr_t)Z.banks[0].logp;
        out[4] = (int)(uintptr_t)Z.banks[0].wdl;
        return 0;
    }
    return 1;              /* think complete */
}

void puct_zone_publish(PuctGame *g) {
    assert(Z.banks[0].state == BANK_QUEUED);
    Z.banks[0].state = BANK_READY;
    Z.banks[0].flush = false;
    deliver_bank(0, &g, 1);
}

/* Size the zone; no C evaluator is installed (JS is the evaluator). */
int puct_attach_js(int rows) {
    puct_zone_init(rows);
    return 0;
}
#endif  /* __EMSCRIPTEN__ */
