/* puct.c — Phase 1 single-thread PUCT MCTS (see chess_puct.h). */
#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#include "chess_puct.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

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
    PuctEvalFn eval;
    void      *eval_ctx;
};

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
 * If `track` is a live node index (0 = none), its remapped location is
 * written to *track_out for callers holding an index across the swap. */
static bool compact(PuctGame *g, uint32_t track, uint32_t *track_out) {
    PuctNode *nb = map_block(g->block_bytes);
    if (!nb)
        return false;
    const PuctNode *ob = g->nodes;
    if (track == g->root)
        *track_out = 1;
    nb[1] = ob[g->root];
    nb[1].parent = 0;
    size_t hd = 1, tl = 2;
    while (hd < tl) {
        uint32_t old_fc = nb[hd].first_child;
        uint32_t n = nb[hd].n_children;
        if (n > 0) {
            if (track >= old_fc && track < old_fc + n)
                *track_out = (uint32_t)tl + (track - old_fc);
            memcpy(&nb[tl], &ob[old_fc], (size_t)n * sizeof(PuctNode));
            nb[hd].first_child = (uint32_t)tl;
            for (uint32_t i = 0; i < n; i++)
                nb[tl + i].parent = (uint32_t)hd;
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

static uint32_t alloc_nodes(PuctGame *g, int n, uint32_t track,
                            uint32_t *track_out) {
    uint32_t tnew = track;
    if ((size_t)g->used + (size_t)n > g->cap &&
        (!compact(g, track, &tnew) || (size_t)g->used + (size_t)n > g->cap)) {
        g->oom++;
        return 0;
    }
    *track_out = tnew;
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
    if (fen && *fen) {
        if (fen_parse(&g->state, fen) != 0) {
            munmap(g->nodes, block_bytes);
            free(g);
            return NULL;
        }
    } else {
        state_init_startpos(&g->state);
    }
    nn_history_reset(&g->hist);
    nn_history_record(&g->hist, &g->state);
    if (g->cap < 3) {
        munmap(g->nodes, block_bytes);
        free(g);
        return NULL;
    }
    memset(g->nodes, 0, 2 * sizeof(PuctNode));   /* slot 0 sentinel, slot 1 root */
    g->used = 2;
    g->root = 1;
    return g;
}

void puct_game_free(PuctGame *g) {
    if (g) {
        munmap(g->nodes, g->block_bytes);
        free(g);
    }
}

void puct_set_eval(PuctGame *g, PuctEvalFn fn, void *ctx) {
    g->eval = fn;
    g->eval_ctx = ctx;
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

/* *idx is the leaf to expand; compaction during the allocation may move it,
 * so the remapped index comes back in *idx and callers keep using it. */
static bool expand(PuctGame *g, uint32_t *idx, float *v_out) {
    MoveList ml;
    gen_legal(&g->state, &ml);
    assert(ml.count > 0);
    uint32_t kids = alloc_nodes(g, ml.count, *idx, idx);
    if (kids == 0)
        return false;

    float v = 0.0f;
    if (g->eval) {
        float planes[NN_TOTAL];
        float logp[PUCT_POLICY_SIZE];
        float wdl[3];
        uint8_t mask[PUCT_POLICY_SIZE];
        int slots[MAX_MOVES];
        nn_encode(&g->state, &g->hist, planes);
        memset(mask, 0, PUCT_POLICY_SIZE);
        for (int i = 0; i < ml.count; i++) {
            slots[i] = chess_policy_slot(&g->state, ml.moves[i]);
            mask[slots[i]] = 1;
        }
        g->eval(g->eval_ctx, 1, planes, mask, logp, wdl);
        double max = -INFINITY, sum = 0.0;
        for (int i = 0; i < ml.count; i++)
            if (logp[slots[i]] > max)
                max = logp[slots[i]];
        for (int i = 0; i < ml.count; i++)
            sum += exp((double)logp[slots[i]] - max);
        for (int i = 0; i < ml.count; i++)
            g->nodes[kids + i].prior =
                (float)(exp((double)logp[slots[i]] - max) / sum);
        v = wdl[0] - wdl[2];
    } else {
        for (int i = 0; i < ml.count; i++)
            g->nodes[kids + i].prior = 1.0f / (float)ml.count;
    }
    for (int i = 0; i < ml.count; i++) {
        g->nodes[kids + i].parent = *idx;
        g->nodes[kids + i].move = ml.moves[i];
    }
    g->nodes[*idx].first_child = kids;
    g->nodes[*idx].n_children = (uint32_t)ml.count;
    *v_out = v;
    return true;
}

static bool simulate(PuctGame *g) {
    uint32_t idx = g->root;
    PuctNode *n = &g->nodes[idx];
    /* Dive as deep as the tree goes; the only wall is the State's fixed
     * MAX_PLY undo stack (make_move asserts ply < MAX_PLY). The +2 keeps
     * state_termination's one-move lookahead probes inside the budget. */
    while (n->n_children > 0 && g->state.ply + 2 < MAX_PLY) {
        uint32_t c = select_child(g, n);
        make_move(&g->state, g->nodes[c].move);
        nn_history_record(&g->hist, &g->state);
        idx = c;
        n = &g->nodes[c];
    }

    float v = 0.0f;
    bool ok = true;
    if (n->n_children > 0) {
        ok = false;   /* ply budget exhausted at an interior node: skip */
    } else if (n->flags & PUCT_TERMINAL) {
        v = n->wdl[0] - n->wdl[2];
    } else {
        int term = state_termination(&g->state);
        if (term != TERM_PLAYING) {
            set_terminal(n, term);
            v = n->wdl[0] - n->wdl[2];
        } else if (!expand(g, &idx, &v)) {
            ok = false;   /* block exhausted: oom counter already bumped */
        }
    }

    if (ok) {
        double vs = -(double)v;   /* leaf's W is its parent-mover's view */
        for (uint32_t i = idx; i != 0; i = g->nodes[i].parent) {
            g->nodes[i].value_sum += vs;
            g->nodes[i].visits++;
            vs = -vs;
        }
    }
    while (idx != g->root) {
        unmake_move(&g->state, g->nodes[idx].move);
        nn_history_drop(&g->hist);
        idx = g->nodes[idx].parent;
    }
    return ok;
}

void puct_think(PuctGame *g, int sims) {
    uint64_t key0 = state_poskey(&g->state);
    PuctNode *r = &g->nodes[g->root];
    if (!(r->flags & PUCT_TERMINAL) && r->n_children == 0) {
        int term = state_termination(&g->state);
        if (term != TERM_PLAYING) {
            set_terminal(r, term);
        } else {
            float v = 0.0f;
            uint32_t ridx = g->root;
            expand(g, &ridx, &v);   /* no backup: the root has no parent */
            g->root = ridx;         /* compaction may have moved the root */
        }
    }
    for (int i = 0; i < sims && simulate(g); i++)
        ;
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
        nr = alloc_nodes(g, 1, 0, &nr);   /* may pack; play replaces the root anyway */
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
