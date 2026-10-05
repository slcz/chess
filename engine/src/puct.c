/* puct.c — Phase 1/2 PUCT MCTS (see chess_puct.h). */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include "chess_puct.h"

#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

/* A walker = one concurrent descent. It owns a scratch State copied fresh
 * from the game root at every sim (no rewind: the copy IS the reset), and
 * parks — mid-descent, at the leaf — by hanging its index field on the
 * leaf's sleep chain until the IO thread delivers its row. */
typedef struct PuctWalker {
    State      st;
    NnHistory  hist;
    uint32_t   idx;      /* current leaf; sleep-chain remapped */
    int        row;      /* staging row, -1 when not parked */
    bool       asleep, has_result, dead;
    MoveList   ml;       /* legal moves cached at the leaf */
    int        slots[MAX_MOVES];
} PuctWalker;

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
    /* staging buffers, SoA, one row per walker */
    float     *planes;   /* [rows][NN_TOTAL] */
    uint8_t   *mask;     /* [rows][PUCT_POLICY_SIZE] */
    float     *logp;     /* [rows][PUCT_POLICY_SIZE] */
    float     *wdl;      /* [rows][3] */
    int        rows_used;
    int        sims_done;
    /* IO thread: strict ping-pong, the waker owns everything between rounds */
    pthread_t       io_thread;
    pthread_mutex_t mu;
    pthread_cond_t  io_cv, walker_cv;
    int             batch_n;   /* rows staged for the IO thread, 0 = idle */
    bool            done;      /* IO finished the current round */
    bool            io_quit;
    PuctEvalFn      eval;
    void           *eval_ctx;
    bool            staging_borrowed;   /* NN attached: staging is the
                                           worker's pinned tensors */
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
 * (co-sleepers may still need remapping by later compactions this round). */
static void sleep_clear(PuctGame *g, PuctWalker *self) {
    for (int i = 0; i < g->n_walkers; i++) {
        PuctWalker *w = &g->walkers[i];
        if (w != self && w->asleep && w->idx == self->idx)
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
    g->planes = malloc((size_t)PUCT_MAX_WALKERS * NN_TOTAL * sizeof(float));
    g->mask = malloc((size_t)PUCT_MAX_WALKERS * PUCT_POLICY_SIZE);
    g->logp = malloc((size_t)PUCT_MAX_WALKERS * PUCT_POLICY_SIZE * sizeof(float));
    g->wdl = malloc((size_t)PUCT_MAX_WALKERS * 3 * sizeof(float));
    if (!g->planes || !g->mask || !g->logp || !g->wdl)
        goto fail;
    pthread_mutex_init(&g->mu, NULL);
    pthread_cond_init(&g->io_cv, NULL);
    pthread_cond_init(&g->walker_cv, NULL);
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
    free(g->planes);
    free(g->mask);
    free(g->logp);
    free(g->wdl);
    munmap(g->nodes, g->block_bytes);
    free(g);
    return NULL;
}

void puct_game_free(PuctGame *g) {
    if (g) {
        munmap(g->nodes, g->block_bytes);
        if (!g->staging_borrowed) {
            free(g->planes);
            free(g->mask);
            free(g->logp);
            free(g->wdl);
        }
        pthread_mutex_destroy(&g->mu);
        pthread_cond_destroy(&g->io_cv);
        pthread_cond_destroy(&g->walker_cv);
        free(g);
    }
}

void puct_set_walkers(PuctGame *g, int n) {
    if (n < 1)
        n = 1;
    if (n > PUCT_MAX_WALKERS)
        n = PUCT_MAX_WALKERS;
    g->n_walkers = n;
}

void puct_set_eval(PuctGame *g, PuctEvalFn fn, void *ctx) {
    g->eval = fn;
    g->eval_ctx = ctx;
}

void puct_set_cpuct(PuctGame *g, float c) { g->cpuct = c; }

void puct_set_staging(PuctGame *g, float *planes, uint8_t *mask,
                      float *logp, float *wdl) {
    g->planes = planes;
    g->mask = mask;
    g->logp = logp;
    g->wdl = wdl;
    g->staging_borrowed = true;
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

/* Expand the childless leaf *idx into its children (the move list and
 * policy slots cached at enqueue time — the game State sits at the root
 * during delivery) with softmax priors over the row's logp. The leaf must
 * be registered on its sleep chain with pidx == idx: compaction inside the
 * allocation keeps *idx current through the swap. */
static bool expand(PuctGame *g, uint32_t *idx, const float *logp,
                   const MoveList *ml, const int *slots) {
    assert(ml->count > 0);
    uint32_t kids = alloc_nodes(g, ml->count);
    if (kids == 0)
        return false;

    double max = -INFINITY, sum = 0.0;
    for (int i = 0; i < ml->count; i++)
        if (logp[slots[i]] > max)
            max = logp[slots[i]];
    for (int i = 0; i < ml->count; i++)
        sum += exp((double)logp[slots[i]] - max);
    for (int i = 0; i < ml->count; i++) {
        PuctNode *c = &g->nodes[kids + i];
        c->parent = *idx;
        c->move = ml->moves[i];
        c->prior = (float)(exp((double)logp[slots[i]] - max) / sum);
    }
    g->nodes[*idx].first_child = kids;
    g->nodes[*idx].n_children = (uint32_t)ml->count;
    return true;
}

/* Delivery-time backup: the leaf's visit was already counted virtually at
 * enqueue, so W accumulates everywhere but N only above the leaf. */
static void backup(PuctGame *g, uint32_t leaf, float v) {
    double vs = -(double)v;
    g->nodes[leaf].value_sum += vs;
    vs = -vs;
    for (uint32_t i = g->nodes[leaf].parent; i != 0; i = g->nodes[i].parent) {
        g->nodes[i].value_sum += vs;
        g->nodes[i].visits++;
        vs = -vs;
    }
}

static void backup_full(PuctGame *g, uint32_t leaf, float v) {   /* terminals */
    double vs = -(double)v;
    for (uint32_t i = leaf; i != 0; i = g->nodes[i].parent) {
        g->nodes[i].value_sum += vs;
        g->nodes[i].visits++;
        vs = -vs;
    }
}

/* One descent attempt: fresh State from the root, down by PUCT, ending in
 * a terminal backup, a parking (eval request staged, walker asleep), or
 * death (ply wall). No pointers into the node block are held across any
 * allocation — the only allocation happens in walker_deliver. */
static void walker_step(PuctGame *g, PuctWalker *w) {
    w->st = g->state;
    w->hist = g->hist;
    uint32_t idx = g->root;
    while (g->nodes[idx].n_children > 0 && w->st.ply + 2 < MAX_PLY) {
        uint32_t c = select_child(g, &g->nodes[idx]);
        make_move(&w->st, g->nodes[c].move);
        nn_history_record(&w->hist, &w->st);
        idx = c;
    }
    PuctNode *n = &g->nodes[idx];
    if (n->n_children > 0) {
        w->dead = true;                    /* ply budget wall: give up */
        return;
    }
    if (n->flags & PUCT_TERMINAL) {
        backup_full(g, idx, n->wdl[0] - n->wdl[2]);
        g->sims_done++;
        return;
    }
    if (!(n->flags & PUCT_SLEEPING)) {     /* fresh leaf: probe termination */
        int term = state_termination(&w->st);
        if (term != TERM_PLAYING) {
            set_terminal(n, term);
            backup_full(g, idx, n->wdl[0] - n->wdl[2]);
            g->sims_done++;
            return;
        }
    }
    /* park: stage the eval request, hang our index on the leaf's chain,
     * add the virtual visit that steers later descents to siblings */
    int row = g->rows_used++;
    assert(row < g->n_walkers);
    nn_encode(&w->st, &w->hist, g->planes + (size_t)row * NN_TOTAL);
    gen_legal(&w->st, &w->ml);
    uint8_t *m = g->mask + (size_t)row * PUCT_POLICY_SIZE;
    memset(m, 0, PUCT_POLICY_SIZE);
    for (int i = 0; i < w->ml.count; i++) {
        w->slots[i] = chess_policy_slot(&w->st, w->ml.moves[i]);
        m[w->slots[i]] = 1;
    }
    w->idx = idx;
    g->nodes[idx].flags |= PUCT_SLEEPING;
    if (idx != g->root)
        g->nodes[idx].visits++;
    w->row = row;
    w->asleep = true;
}

/* Resume a parked walker with its row's result: first delivery expands the
 * leaf (softmax priors over the row's logp), co-sleepers find the leaf
 * already live and only back their own value up. The root's expansion row
 * backs nothing up — the root has no parent. */
static void walker_deliver(PuctGame *g, PuctWalker *w) {
    /* Every access goes through w->idx — that is the field compaction
     * updates, so it stays current through any pack inside expand. */
    if (g->nodes[w->idx].n_children == 0)
        w->dead = !expand(g, &w->idx,
                          g->logp + (size_t)w->row * PUCT_POLICY_SIZE,
                          &w->ml, w->slots);
    if (!w->dead && w->idx != g->root) {
        const float *wd = g->wdl + (size_t)w->row * 3;
        backup(g, w->idx, wd[0] - wd[2]);
        g->sims_done++;
    }
    sleep_clear(g, w);
    w->asleep = w->has_result = false;
    w->row = -1;
}

/* IO thread: grab the staged batch, run the evaluator (NULL = dummy:
 * uniform logp, draw), flag every parked walker, release the waker.
 * Strict ping-pong — it never touches the tree, and it only reads walker
 * fields while the waker is suspended on sem_done. */
static void *io_main(void *arg) {
    PuctGame *g = arg;
    pthread_mutex_lock(&g->mu);
    for (;;) {
        while (g->batch_n == 0 && !g->io_quit)
            pthread_cond_wait(&g->io_cv, &g->mu);
        if (g->io_quit)
            break;
        int n = g->batch_n;
        g->batch_n = 0;
        pthread_mutex_unlock(&g->mu);
        if (g->eval) {
            g->eval(g->eval_ctx, n, g->planes, g->mask, g->logp, g->wdl);
        } else {
            for (int r = 0; r < n; r++) {
                memset(g->logp + (size_t)r * PUCT_POLICY_SIZE, 0,
                       PUCT_POLICY_SIZE * sizeof(float));
                float *w = g->wdl + (size_t)r * 3;
                w[0] = 0; w[1] = 1; w[2] = 0;
            }
        }
        for (int i = 0; i < g->n_walkers; i++)
            if (g->walkers[i].asleep)
                g->walkers[i].has_result = true;
        pthread_mutex_lock(&g->mu);
        g->done = true;
        pthread_cond_signal(&g->walker_cv);
    }
    pthread_mutex_unlock(&g->mu);
    return NULL;
}

static void io_round(PuctGame *g) {
    pthread_mutex_lock(&g->mu);
    g->batch_n = g->rows_used;
    g->done = false;
    pthread_cond_signal(&g->io_cv);
    while (!g->done)
        pthread_cond_wait(&g->walker_cv, &g->mu);
    pthread_mutex_unlock(&g->mu);
    g->rows_used = 0;
}

void puct_think(PuctGame *g, int sims) {
    uint64_t key0 = state_poskey(&g->state);
    if (g->nodes[g->root].flags & PUCT_TERMINAL)
        return;
    g->sims_done = 0;
    g->rows_used = 0;
    for (int i = 0; i < g->n_walkers; i++) {
        PuctWalker *w = &g->walkers[i];
        w->asleep = w->has_result = w->dead = false;
        w->row = -1;
    }
    g->io_quit = false;
    assert(pthread_create(&g->io_thread, NULL, io_main, g) == 0);

    for (;;) {
        for (int i = 0; i < g->n_walkers; i++)
            if (g->walkers[i].has_result)
                walker_deliver(g, &g->walkers[i]);

        bool any_asleep = false, all_asleep = true, any_live = false;
        for (int i = 0; i < g->n_walkers; i++) {
            PuctWalker *w = &g->walkers[i];
            if (w->dead)
                continue;
            any_live = true;
            if (w->asleep) any_asleep = true; else all_asleep = false;
        }
        if (!any_live || (g->sims_done >= sims && !any_asleep))
            break;
        if (all_asleep || g->sims_done >= sims) {
            io_round(g);
            continue;
        }
        for (int i = 0; i < g->n_walkers; i++)
            if (!g->walkers[i].asleep && !g->walkers[i].dead)
                walker_step(g, &g->walkers[i]);
    }

    pthread_mutex_lock(&g->mu);
    g->io_quit = true;
    pthread_cond_signal(&g->io_cv);
    pthread_mutex_unlock(&g->mu);
    pthread_join(g->io_thread, NULL);
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
