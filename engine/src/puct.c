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

/* A walker = one concurrent descent. Its scratch State is copied fresh
 * from the game root at every sim and doubles as the parked leaf's
 * position at delivery (only the side to move is read). idx points at
 * the parked leaf; a parked leaf is expanded on the spot, so no two
 * walkers ever park on one node. */
typedef struct PuctWalker {
    State      st;
    NnHistory  hist;
    uint32_t   idx;      /* current leaf; remapped by compaction */
    int        row;      /* staged row while parked, -1 while walking */
    bool       dead;
} PuctWalker;

enum { WALKERS_DEFAULT = 8 };   /* walkers per game unless puct_set_walkers */

/* Bank life cycle (guarded by Z.mu): FREE → QUEUED (zone_submit) →
 * IN_QUEUE (pushed) → PULLED (D2H enqueued) → READY (published) → FREE
 * (delivered). Only a FREE bank is staged into. */
#ifdef __EMSCRIPTEN__
/* The wasm build is single-threaded: the zone API takes no lock there. */
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

/* Permanent row storage: one mapping, partitioned into the four arrays ×
 * banks, sized once by puct_zone_init (pages materialize on touch);
 * puct_zone_set_buffers can instead point a bank at pinned tensors —
 * this storage is then bypassed, never freed. */
static void *z_mem;        /* also the zone's initialized flag */
static int    z_rows;      /* rows per bank */

/* One row's storage: planes + logp + wdl floats, then the mask bytes. */
#define ZONE_ROW_BYTES \
    (((size_t)NN_TOTAL + PUCT_POLICY_SIZE + 3) * sizeof(float) \
     + PUCT_POLICY_SIZE)

/* The landing zone — exactly one per system, two banks (double buffer):
 * walkers stage rows into the active bank while the io thread evaluates
 * the other. Row ids are global across banks (bank = row / z_rows), so
 * delivery needs no row→game table. */
static struct {
    struct {
        float  *planes;   /* cap × NN_TOTAL, into z_planes or pinned */
        uint8_t *mask;    /* cap × PUCT_POLICY_SIZE */
        float  *logp;     /* cap × PUCT_POLICY_SIZE */
        float  *wdl;      /* cap × 3 */
        int     used;     /* rows staged; frozen once the bank leaves FREE */
        int     state;    /* BANK_FREE .. BANK_READY, guarded by Z.mu */
        bool    flush;    /* pull this bank now, not after the peer's push
                             (starved waker) */
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
    uint64_t   rng;        /* per-game sampling stream, re-keyed at every
                              fresh think */
    uint32_t   oom;
    uint32_t   compactions;
    PuctWalker *walkers;    /* parked idx fields are ground truth;
                               compaction remaps them */
    int        n_walkers;
    bool       think_done; /* completed (awaiting the driver's play) or
                              never-thought; see puct_walk */
    bool       suspended;  /* skip in walks: the game waits on something
                              external (e.g. the opponent's UCI reply) */
    uint8_t    step_state; /* WALK_*: this step's last pass result */
    uint64_t   done_key;   /* position key at completion; 0 = fresh game */
    int        sims_done;
    uint32_t   parks, terminal_hits;
    uint32_t   pending_parks;  /* parks that co-slept on a pending row */
};

/* Remap walkers whose pre-copy idx (snap) was old_idx to new_idx; the
 * snapshot keeps each walker a single hop — matching the live fields
 * can double-remap mid-copy. */
static void sleep_remap(PuctGame *g, const uint32_t *snap,
                        uint32_t old_idx, uint32_t new_idx) {
    for (int i = 0; i < g->n_walkers; i++)
        if (snap[i] == old_idx)
            g->walkers[i].idx = new_idx;
}

#ifdef __EMSCRIPTEN__
/* wasm has no mmap: the block is plain heap — same contract (a fresh,
 * writable, freeable region). */
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

/* Pack the root-reachable subtree into a fresh block (BFS, using the new
 * block as the queue), remapping parent/first_child and walker idx
 * fields; the old block is munmap'd. */
static bool compact(PuctGame *g) {
    PuctNode *nb = map_block(g->block_bytes);
    if (!nb)
        return false;
    const PuctNode *ob = g->nodes;
    /* pre-copy snapshot (see sleep_remap); never malloc(0) */
    uint32_t *snap = malloc((size_t)(g->n_walkers ? g->n_walkers : 1)
                            * sizeof *snap);
    if (!snap) {
        unmap_block(nb, g->block_bytes);
        return false;
    }
    for (int i = 0; i < g->n_walkers; i++)
        snap[i] = g->walkers[i].idx;
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
                sleep_remap(g, snap, old_fc + i, (uint32_t)tl + i);
                nb[tl + i].parent = (uint32_t)hd;
            }
            tl += n;
        }
        hd++;
    }
    free(snap);
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
    assert(!z_mem);
    if (max_rows < 1)
        max_rows = 1;
    size_t rows = (size_t)max_rows;
    z_mem = map_block((size_t)PUCT_BANKS * rows * ZONE_ROW_BYTES);
    assert(z_mem);
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
        Z.eval = dummy_eval;
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
/* Work the io thread can act on: a queued or pulled bank, or an
 * in-flight one flagged flush. */
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

/* Advance every drivable bank one phase (push → pull → publish); mutex
 * HELD, released around each evaluator call. Results are delivered by
 * the waker (zone_pump), never here. */
static void io_drive(void) {
    /* 1. push (async) */
    for (int b = 0; b < PUCT_BANKS; b++) {
        if (Z.banks[b].state != BANK_QUEUED)
            continue;
        Z.banks[b].state = BANK_IN_QUEUE;
        int n = Z.banks[b].used;
        ZONE_UNLOCK();
        bank_eval(PUCT_EVAL_PUSH, b, n);
        ZONE_LOCK();
        pthread_cond_broadcast(&Z.walker_cv);
    }
    /* 2. pull — after the peer is pushed too, or flush */
    for (int b = 0; b < PUCT_BANKS; b++) {
        if (Z.banks[b].state != BANK_IN_QUEUE)
            continue;
        if (!Z.banks[b].flush && Z.banks[b ^ 1].state == BANK_FREE)
            continue;
        Z.banks[b].state = BANK_PULLED;
        Z.banks[b].flush = false;
        int n = Z.banks[b].used;
        ZONE_UNLOCK();
        bank_eval(PUCT_EVAL_PULL, b, n);
        ZONE_LOCK();
    }
    /* 3. publish: block on the D2H, in order */
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

/* io thread (native only): wait for work, then io_drive it. */
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

/* Submit a bank for evaluation; its rows stay staged for delivery. */
static void zone_submit(int b) {
    ZONE_LOCK();
    assert(Z.banks[b].used > 0 && Z.banks[b].state == BANK_FREE);
    Z.banks[b].state = BANK_QUEUED;
    pthread_cond_signal(&Z.io_cv);
    ZONE_UNLOCK();
}

/* Block until bank b is READY; a starved waker flags flush to pull it
 * without the peer's push. */
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

/* Deliver bank b's rows into their games; the bank becomes the active
 * one. */
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
    Z.active = b;
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

/* Replant the game at `fen` in place (see game_fresh); call with no
 * walker in flight. 0 = ok. */
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

/* SEARCH (Grill et al. 2020): the descending move is sampled from
 * pi-bar — no deterministic PUCT path, no root noise. */
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
    return n->first_child + (uint32_t)sample_index_stream(&g->rng, w, m,
                                                          1.0);
}

/* Expand the childless leaf *idx with one child per legal move and
 * uniform placeholder priors; *idx must alias the parked walker's field
 * so a pack inside keeps it current. */
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

/* Fill the parked leaf's real priors: softmax over the row's logp at
 * each child's policy slot. */
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

/* Backup: W and one visit land on every path node; nothing is counted
 * during the descent. */
static void backup(PuctGame *g, uint32_t leaf, float v) {
    double vs = -(double)v;
    for (uint32_t i = leaf; i != 0; i = g->nodes[i].parent) {
        g->nodes[i].value_sum += vs;
        g->nodes[i].visits++;
        vs = -vs;
    }
}

/* One descent attempt: pi-bar sampling from the root to a terminal
 * backup or a park (leaf expanded on the spot, eval row staged).
 * Returns false only when the active bank is full. */
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
        /* expanded but unevaluated: co-sleep on its in-flight row */
        w->idx = idx;
        w->row = -1;
        for (int i = 0; i < g->n_walkers; i++)
            if (g->walkers[i].row >= 0 && g->walkers[i].idx == idx) {
                w->row = g->walkers[i].row;
                break;
            }
        assert(w->row >= 0);
        g->parks++;
        g->pending_parks++;
        return true;
    }
    if (n->n_children > 0) {
        /* ply wall at an evaluated node: no value will ever arrive */
        w->dead = true;
        return true;
    }
    int term;
    if (n->flags & PUCT_TERMINAL ||
        (term = state_termination(&w->st)) != TERM_PLAYING) {
        if (!(n->flags & PUCT_TERMINAL))
            set_terminal(n, term);
        backup(g, idx, n->wdl[0] - n->wdl[2]);
        g->parks++;
        g->terminal_hits++;
        g->sims_done++;
        return true;
    }
    /* park: reserve a row, expand here, stage into it. */
    ZONE_LOCK();
    int b = Z.active;
    if (Z.banks[b].used >= z_rows) {
        ZONE_UNLOCK();
        return false;
    }
    int off = Z.banks[b].used++;
    ZONE_UNLOCK();
    w->idx = idx;
    MoveList ml;
    gen_legal(&w->st, &ml);
    if (!expand_uniform(g, &w->idx, &ml)) {
        w->dead = true;
        return true;
    }
    w->row = b * z_rows + off;
    nn_encode(&w->st, &w->hist, Z.banks[b].planes + (size_t)off * NN_TOTAL);
    uint8_t *m = Z.banks[b].mask + (size_t)off * PUCT_POLICY_SIZE;
    memset(m, 0, PUCT_POLICY_SIZE);
    for (int i = 0; i < ml.count; i++)
        m[chess_policy_slot(&w->st, ml.moves[i])] = 1;
    g->parks++;
    return true;
}

/* Resume a parked walker: fill priors, back the value up; the root's
 * row fills priors only. */
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
    w->row = -1;
}

/* Why a walk pass stopped: DONE (think complete), STUCK (all walkers
 * parked — only a delivery unblocks), FULL (active bank out of rows). */
enum { WALK_STUCK = 0, WALK_DONE = 1, WALK_FULL = 2 };

static int puct_walk(PuctGame *g, int sims) {
    if (g->suspended)
        return WALK_DONE;
    if (g->think_done) {
        /* done-guard: a completed think stays complete until the driver
         * plays a move (puct_play clears done_key) */
        if (g->done_key == state_poskey(&g->state))
            return WALK_DONE;
        g->think_done = false;
        g->sims_done = 0;
        g->parks = g->terminal_hits = g->pending_parks = 0;
        g->rng = rng_next();
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
        puct_set_walkers(g, WALKERS_DEFAULT);
    assert(z_mem);
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
            return WALK_STUCK;
        for (int i = 0; i < g->n_walkers; i++) {
            PuctWalker *w = &g->walkers[i];
            if (w->dead || w->row >= 0)
                continue;
            if (!walker_step(g, w))
                return WALK_FULL;
        }
    }
}

/* Ship the active bank (FREE on entry), or wait out the in-flight peer
 * and deliver its rows. */
static int zone_pump(PuctGame **games, int n_games) {
    if (Z.banks[Z.active].used > 0) {
        int ship = Z.active;
        zone_submit(ship);
        ZONE_LOCK();
        Z.active = ship ^ 1;
        ZONE_UNLOCK();
        if (bank_busy(Z.active)) {
            zone_wait(Z.active);
            return deliver_bank(Z.active, games, n_games);
        }
        return 0;
    }
    assert(bank_busy(Z.active ^ 1));
    zone_wait(Z.active ^ 1);
    return deliver_bank(Z.active ^ 1, games, n_games);
}

bool puct_think_done(const PuctGame *g) { return g->think_done; }

/* True when the position is decided (mate/draw): drivers stop asking. */
bool puct_game_over(const PuctGame *g) {
    return state_termination((State *)&g->state) != TERM_PLAYING;
}

void puct_think(PuctGame *g, int sims) {
    uint64_t key0 = state_poskey(&g->state);
    g->think_done = true;
    g->done_key = 0;
    while (puct_walk(g, sims) != WALK_DONE) {
        PuctGame *gg = g;
        zone_pump(&gg, 1);
    }
    assert(state_poskey(&g->state) == key0);
}

/* Play m (must be legal): a root child keeps its subtree; any other legal
 * move gets a fresh root. */
int puct_play(PuctGame *g, Move m) {
    PuctNode *r = &g->nodes[g->root];
    uint32_t nr = 0;
    for (uint32_t i = 0; i < r->n_children; i++)
        if (g->nodes[r->first_child + i].move == m) {
            nr = r->first_child + i;
            break;
        }
    if (nr == 0)
        nr = alloc_nodes(g, 1);
    if (nr == 0)
        return -1;
    make_move(&g->state, m);
    nn_history_record(&g->hist, &g->state);
    g->root = nr;
    g->nodes[nr].parent = 0;
    g->done_key = 0;
    return 0;
}

Move puct_best_move(const PuctGame *g) {
    const PuctNode *r = &g->nodes[g->root];
    uint32_t best = r->first_child, bv = 0;
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

/* Per-think leaf-arrival telemetry (reset by the next fresh think). */
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
 * ONE mux per process, like the zone. mux_games[0..mux_games_max) is
 * allocated up front (games_max IS the memory budget) and handed to the
 * caller once; idle slots are idled via puct_set_suspended. puct_mux_step
 * runs in rounds: give every eligible game one puct_walk pass, pump,
 * repeat until every live think is complete (ending fully delivered).
 *
 * Locking: only Z.mu and the pool's handshake mutex; the walk path is
 * lock-free — games are claimed off an atomic scan cursor (each game one
 * pass per round ⇒ one writer of its step_state byte), ordered against
 * the driver's resets by the round barrier. Walk tasks are opt-in
 * (PUCT_TASKS; none by default — the driver pulls inline). */

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

/* The last round this task ran: sleep until P.epoch moves past it. */
static _Thread_local int t_epoch;

/* One task's round (and, with no tasks, the driver's): claim scan
 * offsets until dry, giving each eligible game one puct_walk pass. */
static void mux_pull(void) {
    for (;;) {
        int off = atomic_fetch_add_explicit(&mux_scan, 1,
                                            memory_order_relaxed);
        if (off >= mux_games_max ||
            atomic_load_explicit(&mux_full, memory_order_relaxed))
            return;
        PuctGame *g = mux_games[off];
        if (g->suspended || g->step_state == WALK_DONE)
            continue;
        if ((g->step_state = puct_walk(g, P.sims)) == WALK_FULL) {
            atomic_store_explicit(&mux_full, true,
                                  memory_order_relaxed);
            return;
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

/* The game array is handed to the caller once; size the zone first —
 * the mux does not. 0 = ok, -1 = a game failed to allocate. */
int puct_mux_new(int games_max, size_t block_bytes, PuctGame **games) {
    if (games_max < 1)
        games_max = 1;
    mux_games = calloc((size_t)games_max, sizeof *mux_games);
    assert(mux_games);           /* startup allocation */
    mux_games_max = games_max;
    int n = 0;
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
    for (int i = 0; i < games_max; i++) {
        mux_games[i] = puct_game_new(block_bytes);
        if (!mux_games[i]) {
            puct_mux_free();
            return -1;
        }
        games[i] = mux_games[i];
    }
    for (int i = 0; i < n; i++)
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
    P.epoch = 0;    /* fresh tasks (t_epoch 0) must sleep until a step */    pthread_mutex_destroy(&P.mu);
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
    assert(z_mem);
    for (int i = 0; i < mux_games_max; i++)
        mux_games[i]->step_state = WALK_STUCK;
    P.sims = sims;
    for (;;) {
        atomic_store_explicit(&mux_scan, 0, memory_order_relaxed);
        atomic_store_explicit(&mux_full, false, memory_order_relaxed);
        if (P.n_tasks > 0) {
            pthread_mutex_lock(&P.mu);
            P.active = P.n_tasks;
            P.epoch++;
            pthread_cond_broadcast(&P.cv);
            while (P.active > 0)
                pthread_cond_wait(&P.cv, &P.mu);
            pthread_mutex_unlock(&P.mu);
        } else {
            mux_pull();
        }
        bool all_done = true;
        for (int i = 0; i < mux_games_max; i++)
            if (mux_games[i]->step_state != WALK_DONE &&
                !mux_games[i]->suspended) {
                all_done = false;
                break;
            }
        if (all_done)
            break;
        zone_pump(mux_games, mux_games_max);
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
            /* pad q from a real child: 0.0 could NaN via 0*inf */
            q[i] = q[n - 1];
            lp[i] = 0.0;
        }
    }
    /* safe start: sum pi <= 1 already at qmax + lam */
    double a = qmax + lam;
    for (int it = 0; it < 64; it++) {
        double f = -1.0, df = 0.0;
        /* Newton on sum lp/(a-q) = 1 */
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
 * Walk passes run until the think completes (returns 1) or a bank is
 * QUEUED with its rows staged in *out (returns 0); JS runs the net, then
 * puct_zone_publish delivers the rows. */
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
