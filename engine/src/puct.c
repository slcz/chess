/* puct.c — Phase 1/2 PUCT MCTS + Phase 3a multi-game batching (chess_puct.h). */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include "chess_puct.h"

#include <assert.h>
#include <math.h>
#include <pthread.h>
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

/* The landing zone — exactly one per system. The SoA batch of eval rows
 * every game stages into, the evaluator, and its single io thread (strict
 * ping-pong, unchanged from the single-game design; the batch it evals is
 * now cross-game). A walker claims a row when it parks and remembers its
 * row id, so delivery needs no row→game table. */
static struct {
    float  *planes;   /* [cap][NN_TOTAL] */
    uint8_t *mask;    /* [cap][PUCT_POLICY_SIZE] */
    float  *logp;     /* [cap][PUCT_POLICY_SIZE] */
    float  *wdl;      /* [cap][3] */
    bool    owned;    /* buffers malloc'd here; nn_eval swaps in pinned ones */
    int     cap, used;
    PuctEvalFn eval;
    void   *eval_ctx;
    pthread_t       io_thread;
    bool            io_up;
    pthread_mutex_t mu;
    pthread_cond_t  io_cv, walker_cv;
    int             batch_n;   /* rows staged for the io thread, 0 = idle */
    bool            done;      /* io finished the current round */
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
    bool       think_done; /* next walk starts a fresh think */
    int        sims_done;
    uint32_t   parks, terminal_hits;
};

/* Parked walkers point at nodes by plain index; compaction rewrites the
 * matching fields when it moves a flagged node. A stale idx on an awake
 * walker is harmless — it is overwritten at the next park. */
static void sleep_remap(PuctGame *g, uint32_t old_idx, uint32_t new_idx) {
    for (int i = 0; i < g->n_walkers; i++)
        if (g->walkers[i].idx == old_idx)
            g->walkers[i].idx = new_idx;
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
    uint32_t old_root = g->root;
    nb[1] = ob[old_root];
    nb[1].parent = 0;
    if (nb[1].flags & PUCT_SLEEPING)
        sleep_remap(g, old_root, 1);
    size_t hd = 1, tl = 2;
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

void puct_zone_init(int max_rows) {
    if (max_rows < 1)
        max_rows = 1;
    if (!Z.io_up) {
        pthread_mutex_init(&Z.mu, NULL);
        pthread_cond_init(&Z.io_cv, NULL);
        pthread_cond_init(&Z.walker_cv, NULL);
        assert(pthread_create(&Z.io_thread, NULL, io_main, &Z) == 0);
        Z.io_up = true;
    }
    if (Z.cap == max_rows)
        return;
    assert(Z.used == 0);      /* resize only between rounds: no staged rows,
                                 no parked walkers holding row ids */
    if (Z.owned) {
        free(Z.planes);
        free(Z.mask);
        free(Z.logp);
        free(Z.wdl);
    }
    Z.planes = malloc((size_t)max_rows * NN_TOTAL * sizeof(float));
    Z.mask = malloc((size_t)max_rows * PUCT_POLICY_SIZE);
    Z.logp = malloc((size_t)max_rows * PUCT_POLICY_SIZE * sizeof(float));
    Z.wdl = malloc((size_t)max_rows * 3 * sizeof(float));
    assert(Z.planes && Z.mask && Z.logp && Z.wdl);
    Z.cap = max_rows;
    Z.owned = true;
}

void puct_zone_set_buffers(float *planes, uint8_t *mask, float *logp,
                           float *wdl) {
    if (Z.owned) {
        free(Z.planes);
        free(Z.mask);
        free(Z.logp);
        free(Z.wdl);
        Z.owned = false;
    }
    Z.planes = planes;
    Z.mask = mask;
    Z.logp = logp;
    Z.wdl = wdl;
}

void puct_set_eval(PuctEvalFn fn, void *ctx) {
    Z.eval = fn;
    Z.eval_ctx = ctx;
}

/* io thread: grab the staged batch, run the evaluator (NULL = dummy:
 * uniform logp, draw), release the waker. Strict ping-pong — it never
 * touches the trees or the walkers; consuming the results is the caller's
 * job, one puct_deliver pass per game right after zone_eval. */
static void *io_main(void *arg) {
    (void)arg;
    pthread_mutex_lock(&Z.mu);
    for (;;) {
        while (Z.batch_n == 0 && !Z.io_quit)
            pthread_cond_wait(&Z.io_cv, &Z.mu);
        if (Z.io_quit)
            break;
        int n = Z.batch_n;
        Z.batch_n = 0;
        pthread_mutex_unlock(&Z.mu);
        if (Z.eval) {
            Z.eval(Z.eval_ctx, n, Z.planes, Z.mask, Z.logp, Z.wdl);
        } else {
            for (int r = 0; r < n; r++) {
                memset(Z.logp + (size_t)r * PUCT_POLICY_SIZE, 0,
                       PUCT_POLICY_SIZE * sizeof(float));
                float *w = Z.wdl + (size_t)r * 3;
                w[0] = 0; w[1] = 1; w[2] = 0;
            }
        }
        pthread_mutex_lock(&Z.mu);
        Z.done = true;
        pthread_cond_signal(&Z.walker_cv);
    }
    pthread_mutex_unlock(&Z.mu);
    return NULL;
}

/* Fire one eval over the staged prefix; rows stay staged for delivery. */
static void zone_eval(void) {
    assert(Z.used > 0);
    pthread_mutex_lock(&Z.mu);
    Z.batch_n = Z.used;
    Z.done = false;
    pthread_cond_signal(&Z.io_cv);
    while (!Z.done)
        pthread_cond_wait(&Z.walker_cv, &Z.mu);
    pthread_mutex_unlock(&Z.mu);
}

/* Single-game users may think without ever calling puct_zone_init. */
static void ensure_zone(void) {
    if (Z.cap == 0)
        puct_zone_init(PUCT_MAX_WALKERS);
}

/* ---- game ---------------------------------------------------------------- */

PuctGame *puct_game_new(const char *fen, size_t block_bytes) {
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
    g->think_done = true;
    if (fen && *fen) {
        if (fen_parse(&g->state, fen) != 0)
            goto fail;
    } else {
        state_init_startpos(&g->state);
    }
    nn_history_reset(&g->hist);
    nn_history_record(&g->hist, &g->state);
    if (g->cap < 3)
        goto fail;
    memset(g->nodes, 0, 2 * sizeof(PuctNode));   /* slot 0 sentinel, slot 1 root */
    g->used = 2;
    g->root = 1;
    return g;
fail:
    munmap(g->nodes, g->block_bytes);
    free(g);
    return NULL;
}

void puct_game_free(PuctGame *g) {
    if (g) {
        munmap(g->nodes, g->block_bytes);
        free(g);
    }
}

/* Fresh startpos game in place: trees, history and walkers reset; the
 * mmap'd block and the lifetime stats are kept. */
void puct_game_reset(PuctGame *g) {
    state_init_startpos(&g->state);
    nn_history_reset(&g->hist);
    nn_history_record(&g->hist, &g->state);
    memset(g->nodes, 0, 2 * sizeof(PuctNode));
    g->used = 2;
    g->root = 1;
    g->think_done = true;
}

void puct_set_walkers(PuctGame *g, int n) {
    if (n < 1)
        n = 1;
    if (n > PUCT_MAX_WALKERS)
        n = PUCT_MAX_WALKERS;
    g->n_walkers = n;
}

void puct_set_cpuct(PuctGame *g, float c) { g->cpuct = c; }

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

void puct_request_planes(PuctGame *g, float out[NN_TOTAL]) {
    nn_encode(&g->state, &g->hist, out);
}

void puct_request_mask(PuctGame *g, uint8_t out[PUCT_POLICY_SIZE]) {
    MoveList ml;
    gen_legal(&g->state, &ml);
    memset(out, 0, PUCT_POLICY_SIZE);
    for (int i = 0; i < ml.count; i++)
        out[chess_policy_slot(&g->state, ml.moves[i])] = 1;
}

static uint32_t select_child(const PuctGame *g, const PuctNode *n) {
    double sqrt_n = sqrt((double)n->visits);
    uint32_t best = n->first_child;
    double best_score = -INFINITY;
    for (uint32_t i = 0; i < n->n_children; i++) {
        const PuctNode *c = &g->nodes[n->first_child + i];
        double q = c->visits ? c->value_sum / c->visits : 0.0;
        double score = q + (double)g->cpuct * c->prior * sqrt_n
                           / (1.0 + (double)c->visits);
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
    while (g->nodes[idx].n_children > 0 && w->st.ply + 2 < MAX_PLY) {
        uint32_t c = select_child(g, &g->nodes[idx]);
        make_move(&w->st, g->nodes[c].move);
        nn_history_record(&w->hist, &w->st);
        g->nodes[c].visits++;   /* virtual visit on every path node, taken
                                   at descent so later walkers in the same
                                   phase see — and steer around — this path */
        idx = c;
    }
    PuctNode *n = &g->nodes[idx];
    if (n->n_children > 0) {
        undo_path(g, idx);              /* ply budget wall: give up */
        w->dead = true;
        return;
    }
    if (n->flags & PUCT_TERMINAL) {
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
     * leaves — a park is always the sole owner of its row. */
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
    int row = Z.used++;
    assert(row < Z.cap);
    nn_encode(&w->st, &w->hist, Z.planes + (size_t)row * NN_TOTAL);
    uint8_t *m = Z.mask + (size_t)row * PUCT_POLICY_SIZE;
    memset(m, 0, PUCT_POLICY_SIZE);
    for (int i = 0; i < ml.count; i++)
        m[chess_policy_slot(&w->st, ml.moves[i])] = 1;
    w->row = row;
    g->parks++;
}

/* Resume a parked walker with its row's result: fill the leaf's real
 * priors (its children were created at park) and back the value up along
 * the path. The root's row fills priors only — the root has no parent. */
static void walker_deliver(PuctGame *g, PuctWalker *w) {
    assert(g->nodes[w->idx].n_children > 0);
    fill_priors(g, w->idx, Z.logp + (size_t)w->row * PUCT_POLICY_SIZE, &w->st);
    if (w->idx != g->root) {
        const float *wd = Z.wdl + (size_t)w->row * 3;
        backup(g, w->idx, wd[0] - wd[2]);
        g->sims_done++;
    }
    g->nodes[w->idx].flags &= ~(uint8_t)PUCT_SLEEPING;
    w->row = -1;
}

bool puct_walk(PuctGame *g, int sims) {
    if (g->think_done) {                    /* fresh think: per-move reset */
        g->think_done = false;
        g->sims_done = 0;
        g->parks = g->terminal_hits = 0;
        for (int i = 0; i < g->n_walkers; i++) {
            g->walkers[i].dead = false;
            g->walkers[i].row = -1;
        }
    }
    if (g->nodes[g->root].flags & PUCT_TERMINAL) {
        g->think_done = true;
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
            return true;
        }
        if (all_parked || g->sims_done >= sims)
            return false;        /* batch staged (or needs its flush) */
        for (int i = 0; i < g->n_walkers; i++) {
            PuctWalker *w = &g->walkers[i];
            if (w->dead || w->row >= 0)
                continue;
            if (Z.used >= Z.cap)
                return false;    /* zone full: eval what is staged */
            walker_step(g, w);
        }
    }
}

void puct_deliver(PuctGame *g) {
    for (int i = 0; i < g->n_walkers; i++)
        if (g->walkers[i].row >= 0)
            walker_deliver(g, &g->walkers[i]);
}

bool puct_think_done(const PuctGame *g) { return g->think_done; }

void puct_think(PuctGame *g, int sims) {
    uint64_t key0 = state_poskey(&g->state);
    ensure_zone();
    while (!puct_walk(g, sims)) {
        zone_eval();
        puct_deliver(g);
        Z.used = 0;
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
 * arrivals; terminal_hits is the subset backed up from a terminal. */
void puct_park_stats(const PuctGame *g, uint32_t *parks,
                     uint32_t *terminal_hits) {
    if (parks)
        *parks = g->parks;
    if (terminal_hits)
        *terminal_hits = g->terminal_hits;
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
    int        games_max, live, cursor;
    size_t     block_bytes;
    uint64_t   rows;      /* cumulative eval rows */
};

PuctMux *puct_mux_new(int games_max, int rows_max, size_t block_bytes) {
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
    return m;
}

void puct_mux_free(PuctMux *m) {
    if (!m)
        return;
    for (int i = 0; i < m->live; i++)
        puct_game_free(m->games[i]);
    free(m->games);
    free(m);
}

static bool mux_spawn(PuctMux *m) {
    PuctGame *g = puct_game_new(NULL, m->block_bytes);
    if (!g)
        return false;
    m->games[m->live++] = g;
    return true;
}

/* One round: walk every live game (their walkers park into the shared
 * batch), starting one more game whenever all of them are sleeping and
 * both presets still allow it; then fire ONE eval and deliver every game.
 * The walk start rotates so a game late in the order cannot starve when
 * the zone fills early. After the step each game is either mid-think or
 * complete (puct_think_done) — the driver plays and recycles between
 * steps. */
void puct_mux_step(PuctMux *m, int sims) {
    if (m->live == 0 && !mux_spawn(m))
        return;
    for (;;) {
        bool parked = false;
        for (int k = 0; k < m->live; k++) {
            int i = (m->cursor + k) % m->live;
            if (!puct_walk(m->games[i], sims))
                parked = true;
        }
        m->cursor = (m->cursor + 1) % m->live;
        if (!parked || m->live >= m->games_max || Z.used >= Z.cap)
            break;
        if (!mux_spawn(m))
            break;
    }
    if (Z.used == 0)
        return;               /* all thinks complete: driver plays moves */
    zone_eval();
    for (int i = 0; i < m->live; i++)
        puct_deliver(m->games[i]);
    m->rows += (uint64_t)Z.used;
    Z.used = 0;
}

int puct_mux_games(const PuctMux *m) { return m->live; }

PuctGame *puct_mux_game(const PuctMux *m, int i) { return m->games[i]; }

uint64_t puct_mux_rows(const PuctMux *m) { return m->rows; }
