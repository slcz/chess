/* puct.c — Phase 1/2 PUCT MCTS + Phase 3a multi-game batching (chess_puct.h). */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include "chess_puct.h"
#include "chess_rng.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
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
#define BANK_FREE 0
#define BANK_QUEUED 1
#define BANK_IN_QUEUE 2
#define BANK_PULLED 3
#define BANK_READY 4

/* Permanent row storage: ONE mapping, partitioned into the four arrays ×
 * two banks, sized by puct_zone_init and grown only when a later init
 * asks for more rows (pages materialize on touch, so a small cap costs
 * nothing). The nn seam (puct_zone_set_buffers) can instead point a
 * bank's pointers at pinned tensors — this storage is then simply
 * bypassed, never freed. */
static void *z_mem;
static size_t z_bytes;
static int    z_rows;      /* allocated rows per bank */

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
    int     cap;          /* rows per bank, <= PUCT_MAX_ROWS */
    int     active;       /* bank the walkers stage into (waker only) */
    PuctEvalFn eval;
    void   *eval_ctx;
    pthread_t       io_thread;
    bool            io_up;
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
    uint32_t   oom;
    uint32_t   compactions;
    PuctWalker walkers[PUCT_MAX_WALKERS];   /* parked idx fields are the
                                               ground truth; PUCT_SLEEPING
                                               gates the compaction scan */
    int        n_walkers;
    bool       think_done; /* completed (awaiting the driver's play) or
                              never-thought; see puct_walk */
    bool       suspended;  /* skip in walks: the game waits on something
                              external (e.g. the opponent's UCI reply) */
    uint64_t   done_key;   /* position key at completion; 0 = fresh game */
    int        sims_done;
    uint32_t   parks, terminal_hits;
    uint32_t   pending_parks;  /* parks that co-slept on a pending row */
    double     noise_eps, noise_alpha;   /* root Dirichlet noise; 0 = off */
    bool       root_noised;    /* current root's children already mixed */
    float      root_noisy[MAX_MOVES];  /* mixed priors; c->prior stays clean */
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

static void *map_block(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return p == MAP_FAILED ? NULL : p;
}

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
    munmap(g->nodes, g->block_bytes);
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

void puct_zone_init(int max_rows) {
    if (max_rows < 1)
        max_rows = 1;
    if (Z.cap != max_rows)
        for (int b = 0; b < PUCT_BANKS; b++)
            assert(Z.banks[b].state == BANK_FREE &&
                   Z.banks[b].used == 0);   /* resize only between rounds:
                                               row ids encode bank*cap + off */
    if (max_rows > z_rows) {
        size_t rows = (size_t)max_rows;
        size_t planes_n = (size_t)PUCT_BANKS * rows * NN_TOTAL;
        size_t logp_n = (size_t)PUCT_BANKS * rows * PUCT_POLICY_SIZE;
        size_t wdl_n = (size_t)PUCT_BANKS * rows * 3;
        size_t bytes = (planes_n + logp_n + wdl_n) * sizeof(float)
                       + (size_t)PUCT_BANKS * rows * PUCT_POLICY_SIZE;
        void *mem = map_block(bytes);
        assert(mem);   /* row storage is the pipeline's life support */
        float *planes = mem;
        float *logp = planes + planes_n;
        float *wdl = logp + logp_n;
        uint8_t *mask = (uint8_t *)(wdl + wdl_n);
        for (int b = 0; b < PUCT_BANKS; b++)
            puct_zone_set_buffers(b,
                                  planes + (size_t)b * rows * NN_TOTAL,
                                  mask + (size_t)b * rows * PUCT_POLICY_SIZE,
                                  logp + (size_t)b * rows * PUCT_POLICY_SIZE,
                                  wdl + (size_t)b * rows * 3);
        if (z_mem)
            munmap(z_mem, z_bytes);
        z_mem = mem;
        z_bytes = bytes;
        z_rows = max_rows;
    }
    if (!Z.io_up) {
        pthread_mutex_init(&Z.mu, NULL);
        pthread_cond_init(&Z.io_cv, NULL);
        pthread_cond_init(&Z.walker_cv, NULL);
        assert(pthread_create(&Z.io_thread, NULL, io_main, &Z) == 0);
        Z.io_up = true;
    }
    Z.cap = max_rows;
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

/* Dummy evaluator: the whole "forward" is filling uniform logp + draw. */
static void dummy_eval(PuctEvalOp op, int b, int n) {
    if (op != PUCT_EVAL_PUSH)
        return;
    for (int r = 0; r < n; r++) {
        memset(Z.banks[b].logp + (size_t)r * PUCT_POLICY_SIZE, 0,
               PUCT_POLICY_SIZE * sizeof(float));
        float *w = Z.banks[b].wdl + (size_t)r * 3;
        w[0] = 0; w[1] = 1; w[2] = 0;
    }
}

static void bank_eval(PuctEvalOp op, int b, int n) {
    if (Z.eval)
        Z.eval(Z.eval_ctx, op, b, n, Z.banks[b].planes, Z.banks[b].mask,
               Z.banks[b].logp, Z.banks[b].wdl);
    else
        dummy_eval(op, b, n);
}

/* io thread: pick up queued banks one at a time, run the evaluator
 * (NULL = dummy: uniform logp, draw), mark the bank ready. Strict
 * hand-off — it never touches the trees or the walkers; delivering the
 * results is the waker's job (zone_pump). */
static void *io_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&Z.mu);
    for (;;) {
        while (!Z.io_quit && !io_pending())
            pthread_cond_wait(&Z.io_cv, &Z.mu);
        if (Z.io_quit)
            break;
        /* 1. push: H2D + launch compute for every queued bank (async) */
        for (int b = 0; b < PUCT_BANKS; b++) {
            if (Z.banks[b].state != BANK_QUEUED)
                continue;
            Z.banks[b].state = BANK_IN_QUEUE;
            int n = Z.banks[b].used;   /* frozen since submit: only the
                                          active (FREE) bank is staged */
            pthread_mutex_unlock(&Z.mu);
            bank_eval(PUCT_EVAL_PUSH, b, n);
            pthread_mutex_lock(&Z.mu);
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
            pthread_mutex_unlock(&Z.mu);
            bank_eval(PUCT_EVAL_PULL, b, n);
            pthread_mutex_lock(&Z.mu);
        }
        /* 3. publish: block on each pulled bank's D2H, in order */
        for (int b = 0; b < PUCT_BANKS; b++) {
            if (Z.banks[b].state != BANK_PULLED)
                continue;
            int n = Z.banks[b].used;
            pthread_mutex_unlock(&Z.mu);
            bank_eval(PUCT_EVAL_WAIT, b, n);
            pthread_mutex_lock(&Z.mu);
            Z.banks[b].state = BANK_READY;
            pthread_cond_broadcast(&Z.walker_cv);
        }
    }
    pthread_mutex_unlock(&Z.mu);
    return NULL;
}

static bool bank_busy(int b) {
    pthread_mutex_lock(&Z.mu);
    bool busy = Z.banks[b].state != BANK_FREE;
    pthread_mutex_unlock(&Z.mu);
    return busy;
}

/* Submit a bank to the io thread; its rows stay staged for delivery. */
static void zone_submit(int b) {
    pthread_mutex_lock(&Z.mu);
    assert(Z.banks[b].used > 0 && Z.banks[b].state == BANK_FREE);
    Z.banks[b].state = BANK_QUEUED;
    pthread_cond_signal(&Z.io_cv);
    pthread_mutex_unlock(&Z.mu);
}

/* Block until bank b's results have arrived. A pushed-but-unpulled bank
 * means the waker is starved (its games all park on it) — flag it flush so
 * the io thread pulls it instead of waiting for the peer's push. */
static void zone_wait(int b) {
    pthread_mutex_lock(&Z.mu);
    while (Z.banks[b].state != BANK_READY) {
        if (Z.banks[b].state == BANK_IN_QUEUE && !Z.banks[b].flush) {
            Z.banks[b].flush = true;
            pthread_cond_signal(&Z.io_cv);
        }
        pthread_cond_wait(&Z.walker_cv, &Z.mu);
    }
    pthread_mutex_unlock(&Z.mu);
}

/* Deliver bank b's rows into their games and free the bank (it becomes
 * the active one). bank_busy must have been true — the caller waited. */
static int deliver_bank(int b, PuctGame **games, int n_games) {
    for (int i = 0; i < n_games; i++) {
        PuctGame *g = games[i];
        for (int k = 0; k < g->n_walkers; k++) {
            PuctWalker *w = &g->walkers[k];
            if (w->row >= 0 && w->row / Z.cap == b)
                walker_deliver(g, w);
        }
    }
    pthread_mutex_lock(&Z.mu);
    assert(Z.banks[b].state == BANK_READY);
    int rows = Z.banks[b].used;
    Z.banks[b].state = BANK_FREE;
    Z.banks[b].used = 0;
    pthread_mutex_unlock(&Z.mu);
    Z.active = b;
    return rows;
}

/* Single-game users may think without ever calling puct_zone_init. */
static void ensure_zone(void) {
    if (Z.cap == 0)
        puct_zone_init(PUCT_MAX_WALKERS);
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
    g->root_noised = false;
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
    g->n_walkers = PUCT_MAX_WALKERS;
    state_init_startpos(&g->state);
    if (g->cap < 3) {
        munmap(g->nodes, g->block_bytes);
        free(g);
        return NULL;
    }
    game_fresh(g);
    return g;
}

void puct_game_free(PuctGame *g) {
    if (g) {
        munmap(g->nodes, g->block_bytes);
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
    if (n > PUCT_MAX_WALKERS)
        n = PUCT_MAX_WALKERS;
    g->n_walkers = n;
}

void puct_set_suspended(PuctGame *g, bool on) { g->suspended = on; }

void puct_set_noise(PuctGame *g, double eps, double alpha) {
    g->noise_eps = eps > 0.0 ? eps : 0.0;
    g->noise_alpha = alpha > 0.0 ? alpha : 0.3;
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

static uint32_t select_child(const PuctGame *g, const PuctNode *n) {
    /* sqrt(1+N), not sqrt(N): a node's children can be selected while the
     * node itself has 0 visits (its eval row just delivered, no backup has
     * climbed to it yet — the root's first think is the common case). With
     * sqrt(0) the whole PUCT score degenerates to Q=0 for every child and
     * the strict > tie-break sends every walker of the batch onto the
     * FIRST child in movegen order — a 15/16-visit stampede onto an
     * arbitrary move that then wins argmax-visits (the kg1_stampede
     * fixture: a mate-in-1 at prior 0.058 starved by a 0.010-prior king
     * shuffle). +1 keeps the U term alive from the first selection. */
    double sqrt_n = sqrt(1.0 + (double)n->visits);
    const float *pri = (n == &g->nodes[g->root] && g->root_noised)
                           ? g->root_noisy : NULL;
    uint32_t best = n->first_child;
    double best_score = -INFINITY;
    for (uint32_t i = 0; i < n->n_children; i++) {
        const PuctNode *c = &g->nodes[n->first_child + i];
        double q = c->visits ? c->value_sum / c->visits : 0.0;
        double score = q + (double)g->cpuct * (pri ? pri[i] : c->prior)
                               * sqrt_n / (1.0 + (double)c->visits);
        if (score > best_score) {
            best_score = score;
            best = n->first_child + i;
        }
    }
    return best;
}

/* Expand the childless leaf *idx on the spot: one child per legal move,
 * uniform placeholder priors (the real softmax arrives with the eval row).
 * Uniform — not zero — keeps the U term meaningful for walkers that descend
 * through the node while its row is in flight. The leaf must be flagged
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

/* AGZ root exploration noise: P = (1-eps)*p + eps*eta over the ROOT's
 * children, eta ~ Dir(alpha) (chess_puct.h). Runs at the single moment
 * the root's real priors exist — their delivery, or a fresh think's
 * start when the subtree was inherited — and only once per position
 * (root_noised, cleared whenever the root moves). Disabled or forced
 * roots (< 2 children) consume no RNG stream, so noise-off drivers keep
 * their exact random sequences. */
static void root_noise(PuctGame *g) {
    const PuctNode *r = &g->nodes[g->root];
    if (g->noise_eps <= 0.0 || g->root_noised || r->n_children < 2 ||
        (r->flags & PUCT_PENDING))
        return;
    double eta[MAX_MOVES];
    rng_dirichlet(eta, (int)r->n_children, g->noise_alpha);
    for (uint32_t i = 0; i < r->n_children; i++)
        g->root_noisy[i] = (float)((1.0 - g->noise_eps)
                                   * g->nodes[r->first_child + i].prior
                                   + g->noise_eps * eta[i]);
    g->root_noised = true;
}

/* Backup at delivery or terminal arrival: every node on the path already
 * took its visit virtually during the descent, so this adds W only (plus
 * the root's own visit — the descent never increments the root). */
static void backup(PuctGame *g, uint32_t leaf, float v) {
    double vs = -(double)v;
    for (uint32_t i = leaf; i != 0; i = g->nodes[i].parent) {
        g->nodes[i].value_sum += vs;
        vs = -vs;
    }
    g->nodes[g->root].visits++;
}

/* Retract a descent's virtual visits when no value will ever arrive for
 * its path (ply-wall death, failed expansion). */
static void undo_path(PuctGame *g, uint32_t idx) {
    for (uint32_t i = idx; i != g->root; i = g->nodes[i].parent)
        g->nodes[i].visits--;
}

/* One descent attempt: fresh State from the root, down by PUCT, ending in
 * a terminal backup, or a park — the leaf is expanded the moment we find
 * it (uniform placeholder priors) and its eval row staged; real priors and
 * the value are filled in at delivery. The leaf index is parked (flagged
 * and aliased to w->idx) BEFORE the expansion so a pack inside the
 * allocation keeps it current. */
static void walker_step(PuctGame *g, PuctWalker *w) {
    w->st = g->state;
    w->hist = g->hist;
    uint32_t idx = g->root;
    while (g->nodes[idx].n_children > 0 &&
           !(g->nodes[idx].flags & PUCT_PENDING) &&
           w->st.ply + 2 < MAX_PLY) {
        uint32_t c = select_child(g, &g->nodes[idx]);
        make_move(&w->st, g->nodes[c].move);
        nn_history_record(&w->hist, &w->st);
        g->nodes[c].visits++;   /* virtual visit on every path node, taken
                                   at descent so later walkers in the same
                                   phase see — and steer around — this path */
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
        return;
    }
    if (n->n_children > 0) {
        /* ply wall at an evaluated interior node (the descent loop left
         * with children present but no row in flight): no value will ever
         * arrive for this path — retract its virtual visits and give the
         * walker up, exactly like the OOM path. */
        undo_path(g, idx);
        w->dead = true;
        return;
    }
    if (n->flags & PUCT_TERMINAL) {
        if (getenv("PUCT_TRACE")) {
            char fen[96], ru[8] = "0000";
            uint32_t t = idx;
            while (g->nodes[t].parent != g->root)
                t = g->nodes[t].parent;
            move_to_uci(g->nodes[t].move, ru);
            fen_serialize(&w->st, fen, sizeof fen);
            fprintf(stderr, "TERM sim=%d ply=%d root=%s v=%.4f fen=%s\n",
                    g->sims_done + 1, (int)w->st.ply, ru,
                    n->wdl[0] - n->wdl[2], fen);
        }
        backup(g, idx, n->wdl[0] - n->wdl[2]);
        g->parks++;
        g->terminal_hits++;
        g->sims_done++;
        return;
    }
    int term = state_termination(&w->st);      /* fresh leaf: probe once */
    if (term != TERM_PLAYING) {
        set_terminal(n, term);
        backup(g, idx, n->wdl[0] - n->wdl[2]);
        g->parks++;
        g->terminal_hits++;
        g->sims_done++;
        return;
    }
    /* park: expand here, then stage the row. From now on the leaf has
     * children, so later walkers descend through it and park on fresh
     * leaves — a park is always the sole owner of its row. puct_walk has
     * verified the active bank has room. */
    w->idx = idx;
    g->nodes[w->idx].flags |= PUCT_SLEEPING;
    MoveList ml;
    gen_legal(&w->st, &ml);
    if (!expand_uniform(g, &w->idx, &ml)) {    /* may pack; w->idx follows */
        undo_path(g, w->idx);   /* expand failed: no value will arrive */
        g->nodes[w->idx].flags &= ~(uint8_t)PUCT_SLEEPING;
        w->dead = true;
        return;
    }
    int b = Z.active;
    int off = Z.banks[b].used++;
    w->row = b * Z.cap + off;                  /* global row id */
    nn_encode(&w->st, &w->hist, Z.banks[b].planes + (size_t)off * NN_TOTAL);
    uint8_t *m = Z.banks[b].mask + (size_t)off * PUCT_POLICY_SIZE;
    memset(m, 0, PUCT_POLICY_SIZE);
    for (int i = 0; i < ml.count; i++)
        m[chess_policy_slot(&w->st, ml.moves[i])] = 1;
    g->parks++;
}

/* Resume a parked walker with its row's result: fill the leaf's real
 * priors (its children were created at park) and back the value up along
 * the path. The root's row fills priors only — the root has no parent. */
static void walker_deliver(PuctGame *g, PuctWalker *w) {
    int b = w->row / Z.cap;
    int off = w->row - b * Z.cap;
    assert(g->nodes[w->idx].n_children > 0);
    /* fill only on the row's first delivery: co-sleepers share the row
     * and would otherwise clobber post-delivery edits (root noise) with
     * a recomputed clean softmax */
    if (g->nodes[w->idx].flags & PUCT_PENDING)
        fill_priors(g, w->idx,
                    Z.banks[b].logp + (size_t)off * PUCT_POLICY_SIZE, &w->st);
    if (w->idx == g->root)
        root_noise(g);   /* the fresh root's priors just became real */
    if (w->idx != g->root) {
        const float *wd = Z.banks[b].wdl + (size_t)off * 3;
        if (getenv("PUCT_TRACE")) {   /* leaf-value post-mortem dump
                                         (stderr; one line per delivered
                                         row: sim order, depth, the ROOT
                                         child the line descends through,
                                         mover-view value, FEN) */
            char fen[96], ru[8] = "0000";
            uint32_t n = w->idx;
            while (g->nodes[n].parent != g->root)
                n = g->nodes[n].parent;
            move_to_uci(g->nodes[n].move, ru);
            fen_serialize(&w->st, fen, sizeof fen);
            fprintf(stderr, "LEAF sim=%d ply=%d root=%s v=%.4f fen=%s\n",
                    g->sims_done + 1, (int)w->st.ply, ru, wd[0] - wd[2],
                    fen);
        }
        backup(g, w->idx, wd[0] - wd[2]);
        g->sims_done++;
    }
    sleep_clear(g, w);
    w->row = -1;
}

/* puct_think decomposed for the mux driver (both live in this file): one
 * walking pass — descend/park until all walkers are parked, the sims
 * target is met, or the active bank is full; true = think complete. */
static bool puct_walk(PuctGame *g, int sims) {
    if (g->suspended)
        return true;      /* waiting on the outside: not done, not parked */
    if (g->think_done) {
        /* A completed think stays complete until the driver plays a move
         * — puct_play clears done_key, because a repetition cycle
         * returns the same poskey while still deserving a fresh think.
         * The mux re-walks every game on every pass, and re-searching an
         * un-played root would corrupt the per-think stats.
         * done_key == 0 marks a fresh, never-thought game — its first
         * walk starts the fresh think below. */
        if (g->done_key == state_poskey(&g->state))
            return true;
        g->think_done = false;              /* fresh think: per-move reset */
        g->sims_done = 0;
        g->parks = g->terminal_hits = g->pending_parks = 0;
        for (int i = 0; i < g->n_walkers; i++) {
            g->walkers[i].dead = false;
            g->walkers[i].row = -1;
        }
        root_noise(g);   /* inherited subtree: the new root's clean
                            delivered priors get their one mix now */
    }
    if (g->nodes[g->root].flags & PUCT_TERMINAL) {
        g->think_done = true;
        g->done_key = state_poskey(&g->state);
        return true;
    }
    ensure_zone();
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
            return true;
        }
        if (all_parked || g->sims_done >= sims)
            return false;        /* batch staged (or needs its flush) */
        for (int i = 0; i < g->n_walkers; i++) {
            PuctWalker *w = &g->walkers[i];
            if (w->dead || w->row >= 0)
                continue;
            if (Z.banks[Z.active].used >= Z.cap)
                return false;    /* active bank full: caller ships it */
            walker_step(g, w);
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
        zone_submit(Z.active);
        Z.active ^= 1;
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

void puct_think(PuctGame *g, int sims) {
    uint64_t key0 = state_poskey(&g->state);
    ensure_zone();
    /* puct_think always runs a fresh sims budget on the current position
     * (back-to-back calls keep searching the same tree); the done-guard of
     * puct_walk is the mux driver's contract, not this one. */
    g->think_done = true;
    g->done_key = 0;
    while (!puct_walk(g, sims)) {
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
    g->root_noised = false;
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

/* ---- multi-game multiplexer ---------------------------------------------- */

struct PuctMux {
    PuctGame **games;
    int        games_max, cursor;   /* live == games_max, always */
    size_t     block_bytes;
};

/* Allocate ALL games_max games up front and hand the fixed game array to
 * the caller once (games_max IS the memory budget — one mmap'd node
 * block per game — and the slot set never changes, so there are no
 * per-slot accessors: the caller keeps the handles). Idle slots are
 * skipped via puct_set_suspended. NULL if any game fails (games is then
 * untouched). */
PuctMux *puct_mux_new(int games_max, int rows_max, size_t block_bytes,
                      PuctGame **games) {
    if (games_max < 1)
        games_max = 1;
    puct_zone_init(rows_max);
    PuctMux *m = calloc(1, sizeof *m);
    if (!m)
        return NULL;
    m->games = calloc((size_t)games_max, sizeof *m->games);
    if (!m->games) {
        free(m);
        return NULL;
    }
    m->games_max = games_max;
    m->block_bytes = block_bytes;
    for (int i = 0; i < games_max; i++) {
        m->games[i] = puct_game_new(block_bytes);
        if (!m->games[i]) {
            puct_mux_free(m);
            return NULL;
        }
        games[i] = m->games[i];
    }
    return m;
}

void puct_mux_free(PuctMux *m) {
    if (!m)
        return;
    for (int i = 0; i < m->games_max; i++)
        puct_game_free(m->games[i]);
    free(m->games);
    free(m);
}

/* One round: walk every non-suspended game with the same sims budget
 * (their walkers park into the active bank; suspended games are idle and
 * count as done). Full or stuck banks are shipped to the io thread
 * asynchronously — walking continues in the other bank while the
 * evaluator runs, batched across every live game. The step ends fully
 * delivered (no parked walkers, both banks empty), so the driver can
 * play and recycle between steps. The walk start rotates so a game late
 * in the order cannot starve. */
void puct_mux_step(PuctMux *m, int sims) {
    for (;;) {
        bool parked = false;
        for (int k = 0; k < m->games_max; k++) {
            int i = (m->cursor + k) % m->games_max;
            if (!puct_walk(m->games[i], sims))
                parked = true;
        }
        m->cursor = (m->cursor + 1) % m->games_max;
        if (!parked)
            break;
        zone_pump(m->games, m->games_max);
    }
    for (int b = 0; b < PUCT_BANKS; b++)
        assert(Z.banks[b].used == 0 && !bank_busy(b));
}

float puct_cpuct(const PuctGame *g) { return g->cpuct; }

/* Grill et al. (ICML 2020) regularized policy improvement: the exact
 * solution of argmax_y [q^T y - lambda KL[pi_theta, y]], i.e.
 *     pi(a) = lambda*pi_theta(a) / (alpha - q(a)),
 * with lambda = cpuct*sqrt(N)/(|A|+N) and alpha (a root of the monotone
 * constraint sum pi = 1) found by Newton from the right.  Unvisited
 * children take the root's value.  Writes every child (dense), returns
 * the count. */
int puct_learn_pi(const PuctGame *g, uint16_t *moves, float *pi, int cap) {
    const PuctNode *r = &g->nodes[g->root];
    int n = (int)r->n_children;
    if (n <= 0 || cap <= 0)
        return 0;
    if (n > cap)
        n = cap;
    double N = (double)r->visits;
    double q[MAX_MOVES], lp[MAX_MOVES], qmax = -INFINITY, sum = 0.0;
    double lam = N > 0.0 ? (double)g->cpuct * sqrt(N)
                               / (N + (double)r->n_children) : 0.0;
    /* root W is parent-view; children's q are mover-view (select_child),
     * so the unvisited fallback must be negated to match them */
    double v_root = N > 0.0 ? -(r->value_sum / N) : 0.0;
    for (int i = 0; i < n; i++) {
        const PuctNode *c = &g->nodes[r->first_child + i];
        q[i] = c->visits ? c->value_sum / (double)c->visits : v_root;
        lp[i] = lam * (double)c->prior;
        if (q[i] > qmax)
            qmax = q[i];
    }
    double a = qmax + 1.0;                     /* f(a) < 0: Newton is safe */
    for (int it = 0; it < 64; it++) {
        double f = -1.0, df = 0.0;
        for (int i = 0; i < n; i++) {
            double d = a - q[i];
            f += lp[i] / d;
            df -= lp[i] / (d * d);
        }
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
