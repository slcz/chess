/*
 * search.c — the Gumbel self-play search core (chess_search.h).
 *
 * A line-for-line port of the sequential reference
 * chess_ai/selfplay/gumbel.py, reshaped into the resumable per-game state
 * machine PARALLEL_DESIGN.md §4.3 specifies for the parallel engine: the
 * only park point is the fresh expansion leaf (and the fresh root), and
 * sp_game_step() / sp_deliver() are exactly the worker loop's game_step()
 * and result delivery. The descent order inside a round is the reference's
 * pinned candidate-major order (`for a in A: plays × descend(a)`), which is
 * what makes the L2 exact diff against gumbel.py bit-exact.
 *
 * Parity obligations honored here (each is load-bearing for L2):
 *   - RNG consumption: one Gumbel per legal move in movegen enumeration
 *     order at candidate time; one uniform() per sampled move, resolved by
 *     inverse CDF over the candidates in enumeration order; nothing else.
 *   - All tree arithmetic in IEEE double, expressions parenthesized in the
 *     same order as the reference (fp addition/multiplication is not
 *     associative; the L2 gate needs the same rounding at every step).
 *   - Ties everywhere break by movegen enumeration order (strict `>`
 *     comparisons scanning in enumeration order; sorts use the enumeration
 *     index as the secondary key).
 *   - π′ is computed over the ORIGINAL candidate set A₀ (z-descending) in
 *     that order — softmax sums in the same order as the reference.
 *   - The handoff carries the played child's whole subtree (stats included)
 *     into the next move's search — the F4 property, "every root after
 *     move 1 costs zero evaluations".
 *
 * Board discipline (F7): nodes store no position. The game's working State
 * is walked with make/unmake; `pushed[]` is the move log (mirror of the FFI
 * handle's), so rewinding to the root ply unmakes with the right moves, and
 * the NN history ring is kept in step on every push/pop — a plane encoding
 * taken while parked therefore sees the correct 8-ply window.
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "chess_ffi.h"
#include "chess_nn.h"
#include "chess_rng.h"
#include "chess_search.h"

/* ------------------------------------------------------------------ */
/* Node — arena-allocated, variable size                               */
/* ------------------------------------------------------------------ */

/* Node states (gumbel.py's NEW / EVALUATED / TERMINAL). */
enum { SP_NEW = 0, SP_EVALUATED = 1, SP_TERMINAL = 2 };

#define SP_NO_CHILD 0xFFFFFFFFu
#define SP_NO_NODE  SP_NO_CHILD

/*
 * Fixed header, then a variable tail sized by the legal-move count k,
 * laid out so the doubles stay 8-aligned:
 *
 *   [header] [double logp[k]] [u32 child[k]] [Move moves[k]] [u16 slots[k]]
 *
 * child[i] is the arena index of the child for legal move i (SP_NO_CHILD
 * until a descent creates it) — the O(1) equivalent of the reference's
 * children dict keyed by enumeration index. moves[]/slots[] record the
 * movegen enumeration: moves[i] is the move, slots[i] its policy slot
 * (nn_policy_slot, features.py mirror). logp[i] is the net's policy_logp
 * for slots[i]. All per-node arrays are fixed at creation (the legal-move
 * list of a position never changes).
 */
typedef struct SpNode {
    uint32_t visits;      /* backed-up descents through this node          */
    uint16_t n_legal;     /* k: legal moves in movegen order               */
    Move     move;        /* the move from the parent leading here (root 0)*/
    uint8_t  state;       /* SP_NEW / SP_EVALUATED / SP_TERMINAL           */
    uint8_t  pad_;
    double   value_sum;   /* Σ sign·v — the node's OWN mover's view        */
    double   wdl[3];      /* net (pW,pD,pL) own-mover view; terminal one-hot */
} SpNode;

#define SP_NODE_HDR (((sizeof(SpNode)) + 7u) & ~(size_t)7u)

static double *sp_logp(const SpNode *n) {
    return (double *)((const char *)n + SP_NODE_HDR);
}
static uint32_t *sp_child(const SpNode *n) {
    return (uint32_t *)((const char *)n + SP_NODE_HDR + (size_t)n->n_legal * 8u);
}
static Move *sp_moves(const SpNode *n) {
    return (Move *)((const char *)n + SP_NODE_HDR + (size_t)n->n_legal * 12u);
}
static uint16_t *sp_slots(const SpNode *n) {
    return (uint16_t *)((const char *)n + SP_NODE_HDR + (size_t)n->n_legal * 14u);
}

/* The node's own net value v = pW − pL (own mover's view). */
static double node_v(const SpNode *n) { return n->wdl[0] - n->wdl[2]; }

/* Quantize the f32 feature tensor into a request's u8 planes, exactly:
 * dims 0–115 of each token are 0/1 one-hots; dim 116 stores the RAW
 * halfmove clock (the encoder emits clock/100.0 — a plain u8 cast would
 * truncate it to 0 or 1). The un-quantizer computes (float)clock/100.0f,
 * bit-identical to nn_encode's f32 output. */
static void quantize_planes(const float *f32, uint8_t *u8) {
    for (int t = 0; t < 64; t++) {
        const float *row = f32 + (size_t)t * 117;
        uint8_t *ou = u8 + (size_t)t * 117;
        for (int i = 0; i < 116; i++)
            ou[i] = (uint8_t)(row[i] + 0.5f);
        ou[116] = (uint8_t)(row[116] * 100.0f + 0.5f);
    }
}

/* ------------------------------------------------------------------ */
/* Descent — one root→leaf walk (parkable)                             */
/* ------------------------------------------------------------------ */

/* One descent's state: the path of arena node indices (root first, the
 * current position last) and the SH schedule's forced first step. The
 * sequential mode keeps one descent in fixed storage inside the game; the
 * wave mode keeps one per candidate in the arena (growable paths — many
 * flights are parked concurrently and the arena already reset per move). */
typedef struct {
    uint32_t *path;       /* node indices root→leaf                        */
    int       len, cap;
    int       forced;     /* forced first step (enum idx), −1 = Eq. 14     */
    int       growable;   /* grow `path` in the arena (wave-mode flights)  */
} SpDescent;

/* Wave-mode per-candidate descent states within one round's wave. */
enum { SP_D_RUNNABLE = 0, SP_D_RUNNING = 1, SP_D_PARKED = 2, SP_D_DONE = 3 };

/* ------------------------------------------------------------------ */
/* Arena — per-game chunked bump allocator                             */
/* ------------------------------------------------------------------ */

/*
 * The tree arena is a FIXED bump region inside the game (no allocation —
 * deviation 9). Two regions ping-pong per move: at finalize the handoff
 * subtree is compacted into the other region, so the copy never overlaps
 * its source and the dead tree is reclaimed by a pointer reset. A full
 * region or node table asserts — the caps (SP_GAME_ARENA_CAP /
 * SP_GAME_MAX_NODES) bound the worst search (eval at n = 200) with
 * headroom; self-play peaks at ~55 KB.
 */
typedef struct {
    char    *base;            /* the game's fixed region                   */
    size_t   used;            /* bump offset                               */
    SpNode  *nodes[SP_GAME_MAX_NODES];   /* arena index → node pointer    */
    uint32_t n_nodes;
} SpArena;

static void arena_init(SpArena *a, char *base) {
    a->base = base;
    a->used = 0;
    a->n_nodes = 0;
}

/* Bump-allocate `bytes` (8-aligned); NULL when the region is full. */
/* The game's CURRENT bump arena (region ping-pongs per move). A macro so
 * const getters can reach it without a const-cast dance. */
#define ARENA(g) ((SpArena *)&(g)->arenas[(g)->cur_arena])

static void *arena_alloc(SpArena *a, size_t bytes) {
    bytes = (bytes + 7u) & ~(size_t)7u;
    if (a->used + bytes > SP_GAME_ARENA_CAP)
        return NULL;
    void *p = a->base + a->used;
    a->used += bytes;
    return p;
}

static uint32_t arena_add_node(SpArena *a, SpNode *n) {
    assert(a->n_nodes < SP_GAME_MAX_NODES);
    a->nodes[a->n_nodes] = n;
    return a->n_nodes++;
}

static SpNode *arena_node(const SpArena *a, uint32_t idx) {
    assert(idx != SP_NO_NODE && idx < a->n_nodes);
    return a->nodes[idx];
}

/* ------------------------------------------------------------------ */
/* Records — gumbel.py's MoveRecord, field for field                   */
/* ------------------------------------------------------------------ */

/* One recorded move — 108 bytes, fixed. No FEN string: the position is
 * reconstructible by replaying the game's moves from the start position
 * (sp_rec_fen_before does it on demand); π′ is stored f32 (net-native —
 * the L2 gate's 1e-6 tolerance is untouched by the rounding). Bounded by
 * MAX_PLY: the rules engine cannot play a longer game. */
typedef struct {
    uint16_t pi_slot[SP_REC_CANDIDATES];  /* A₀, z-descending              */
    float    pi_prob[SP_REC_CANDIDATES];  /* π′, aligned with pi_slot      */
    uint16_t played_slot;    /* 0xFFFF on a resign-terminated move         */
    uint16_t played_move;    /* Move bits (0 on resign)                    */
    float    outlook;        /* max completedQ over A₀                     */
    uint16_t ply;
    uint8_t  mover;          /* 0 white / 1 black                          */
    uint8_t  n_pi;           /* |A₀| — the π′ support                      */
} SpRec;

/* ------------------------------------------------------------------ */
/* The game                                                            */
/* ------------------------------------------------------------------ */

/* State-machine phases of one move. The step loop advances through them
 * without returning except to park (NEED_EVAL) or finish a move (MOVED). */
enum { SP_PH_START_MOVE = 0, SP_PH_CANDIDATES = 1, SP_PH_ROUND = 2,
       SP_PH_FINAL = 3 };

struct SpGame {
    SpConfig  cfg;
    State     st;                /* the working position (F7)             */
    NnHistory nn;                /* encoder history ring, in step with st */
    Move      pushed[MAX_PLY];   /* move log: pushed[i] led to ply i+1    */
    Rng256    rng;               /* the game's private stream (R6)        */
    uint64_t  seed;
    uint32_t  game_index;
    char      start_fen[CHESS_FEN_MAX];
    State     start;             /* the position at game start — FEN
                                  * replay for records reads from here   */

    /* Tree arenas: two fixed bump regions, ping-ponged at the per-move
     * reset (see SpArena above). */
    SpArena   arenas[2];
    int       cur_arena;
    char      regions[2][SP_GAME_ARENA_CAP];

    int       phase;
    uint32_t  root;              /* current search's root (arena idx)     */
    uint32_t  handoff;           /* next move's root (arena idx), or NO   */
    int       root_ply;          /* st.ply when this move's search began  */

    /* Candidate schedule (root-level search state, per move). */
    double    z[MAX_MOVES];      /* Gumbel scores by enumeration index    */
    uint16_t  cand[SP_MAX_CANDIDATES]; /* survivors A, z-desc; A[0]=survivor */
    uint16_t  a0[SP_MAX_CANDIDATES];   /* original candidates A₀, z-desc   */
    int       n_cand;
    int       m0;                /* ORIGINAL candidate count |A₀| — π′ and
                                  * the sampled move live over this set, not
                                  * the halved survivor set                 */
    int       rounds;            /* ceil(log2 m), fixed from original m   */
    int       plays;             /* descents per candidate this round     */
    int       cur_cand, cur_play;/* candidate-major scheduler cursor      */

    /* Active descent (sequential mode). path[0] is the root; the board
     * stands on path[len-1]. len == 0 between descents. */
    uint32_t  seq_path[SP_MAX_PATH];
    SpDescent seq;

    /* Wave mode (parallel engine): per-candidate flights of the current
     * wave. Many descents are parked concurrently (one per candidate,
     * disjoint subtrees); wave w+1 starts only when every wave-w descent
     * is backed up (the barrier — SP_STEP_IDLE reports the stall). */
    int       wave;              /* current wave index within the round    */
    int       wave_pending;      /* parked, undelivered descents this wave */
    uint8_t   dstate[SP_MAX_CANDIDATES];     /* SP_D_* per candidate       */
    SpDescent *flights[SP_MAX_CANDIDATES]; /* per-candidate descent          */
    int       wave_mode;         /* wave-concurrent scheduling enabled     */
    int       root_parked;       /* wave mode: root eval outstanding       */
    int       owner;             /* engine bookkeeping (result routing)    */

    /* Park state. */
    int       parked;            /* 1 while an eval request is pending    */
    int       park_root;         /* the pending request is a root eval    */
    uint32_t  parked_node;       /* the node awaiting its evaluation      */
    float     planes[NN_TOTAL];  /* encoding of the parked position       */

    int       max_plies;         /* adjudication bound (< 0 = unbounded)  */

    /* Records + result. */
    SpRec     recs[MAX_PLY];     /* one per ply (the rules-engine bound)  */
    int       n_recs;
    int       ended;             /* SP_END_*                              */
    int       term_code;         /* state_termination code when ended==NATURAL */
    int       resigned;          /* −1 / 0 / 1                            */
};

void sp_config_init(SpConfig *cfg) {
    cfg->n = 8;
    cfg->m_cap = 16;
    cfg->c_visit = 50.0;
    cfg->c_scale = 1.0;
    cfg->c_fpu = 0.0;
    cfg->sample_plies = -1;      /* None: sample ∝ π′ every ply (training) */
    cfg->resign_value = -0.90;
}

static int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* Python's (m-1).bit_length() — the number of halving rounds, from the
 * ORIGINAL m. m ≤ 1 gives 0 (no halving loop). */
static int bit_length_m1(int m) {
    if (m <= 1) return 0;
    return 64 - __builtin_clzll((unsigned long long)(m - 1));
}

/* ------------------------------------------------------------------ */
/* Board helpers — make/unmake with the log + encoder ring in step     */
/* ------------------------------------------------------------------ */

static void push_move(SpGame *g, Move m) {
    make_move(&g->st, m);
    g->pushed[g->st.ply - 1] = m;
    nn_history_record(&g->nn, &g->st);
}

static void rewind_to(SpGame *g, int ply) {
    while ((int)g->st.ply > ply) {
        unmake_move(&g->st, g->pushed[g->st.ply - 1]);
        nn_history_drop(&g->nn);
    }
}

/* ------------------------------------------------------------------ */
/* Node construction                                                   */
/* ------------------------------------------------------------------ */

/* Create a node for the position the State currently stands on. Board must
 * be PLAYING (termination is checked by the caller — terminal nodes are
 * built only through make_child). Legal moves are enumerated once here:
 * this list, its policy slots and the array order ARE the enumeration the
 * RNG stream and every tie-break are specified over (R6). Returns the arena
 * index, or SP_NO_NODE on OOM. */
static uint32_t new_node_here(SpGame *g, Move incoming) {
    MoveList ml;
    gen_legal(&g->st, &ml);
    assert(ml.count >= 1 && ml.count <= MAX_MOVES);
    int k = ml.count;

    SpNode *n = arena_alloc(ARENA(g), SP_NODE_HDR + (size_t)k * 16u);
    if (!n) return SP_NO_NODE;
    memset(n, 0, SP_NODE_HDR + (size_t)k * 16u);
    n->state = SP_NEW;
    n->n_legal = (uint16_t)k;
    n->move = incoming;
    Move *mv = sp_moves(n);
    uint16_t *sl = sp_slots(n);
    uint32_t *ch = sp_child(n);
    for (int i = 0; i < k; i++) {
        mv[i] = ml.moves[i];
        int slot = nn_policy_slot(ml.moves[i], (int)g->st.side);
        assert(slot >= 0 && slot < NN_POLICY_SIZE);
        sl[i] = (uint16_t)slot;
        ch[i] = SP_NO_CHILD;
    }
    return arena_add_node(ARENA(g), n);
}

/* make_child (reference _make_child): the board has already been pushed to
 * the child's position. A terminal position builds a one-hot node (mate
 * from the side-to-move's view = loss; every other termination = draw —
 * state_termination implements the claim-draw semantics). */
static uint32_t make_child(SpGame *g, const SpNode *parent, int slot) {
    Move incoming = sp_moves(parent)[slot];
    int t = state_termination(&g->st);
    SpNode *c;
    if (t != TERM_PLAYING) {
        c = arena_alloc(ARENA(g), SP_NODE_HDR);
        if (!c) return SP_NO_NODE;
        memset(c, 0, SP_NODE_HDR);
        c->state = SP_TERMINAL;
        c->n_legal = 0;
        c->move = incoming;
        if (t == TERM_CHECKMATE) {
            c->wdl[0] = 0.0; c->wdl[1] = 0.0; c->wdl[2] = 1.0;
        } else {
            c->wdl[0] = 0.0; c->wdl[1] = 1.0; c->wdl[2] = 0.0;
        }
        return arena_add_node(ARENA(g), c);
    }
    return new_node_here(g, incoming);
}

/* Deep-copy a subtree into a fresh arena (the handoff: the played child and
 * everything under it survives the per-move arena reset, stats included).
 * Children are copied first so the parent's child entries can be remapped
 * on the way out — u32 indices make the remap total. */
static uint32_t copy_subtree(SpArena *dst, const SpArena *src, uint32_t idx) {
    const SpNode *n = arena_node(src, idx);
    SpNode *c = arena_alloc(dst, SP_NODE_HDR + (size_t)n->n_legal * 16u);
    assert(c);
    memcpy(c, n, SP_NODE_HDR + (size_t)n->n_legal * 16u);
    uint32_t cidx = arena_add_node(dst, c);
    uint32_t *ch = sp_child(c);
    for (int i = 0; i < n->n_legal; i++)
        if (ch[i] != SP_NO_CHILD)
            ch[i] = copy_subtree(dst, src, ch[i]);
    return cidx;
}

/* ------------------------------------------------------------------ */
/* Selection, backup, scheduler                                        */
/* ------------------------------------------------------------------ */

/* Full Gumbel Eq. 14 at an interior node — PLAN math verbatim (reference
 * _select_eq14). Scans legal moves in enumeration order with a strict `>`:
 * the first maximum wins, i.e. ties break by movegen order. */
static int select_eq14(const SpGame *g, const SpNode *node) {
    const uint32_t *child = sp_child(node);
    uint32_t max_n = 0, total_n = 0;
    for (int i = 0; i < node->n_legal; i++) {
        if (child[i] == SP_NO_CHILD) continue;
        const SpNode *c = arena_node(ARENA(g), child[i]);
        if (c->visits > max_n) max_n = c->visits;
        total_n += c->visits;
    }
    double base = g->cfg.c_visit + (double)max_n;

    int best = -1;
    double best_score = -INFINITY;
    for (int i = 0; i < node->n_legal; i++) {
        double q;
        double n_a;
        const SpNode *c = child[i] != SP_NO_CHILD
                              ? arena_node(ARENA(g), child[i]) : NULL;
        if (c != NULL && c->visits > 0) {
            q = -(c->value_sum / (double)c->visits);  /* parent's view */
            n_a = (double)c->visits;
        } else {
            q = node_v(node) - g->cfg.c_fpu;          /* Eq. 10 */
            n_a = (c != NULL) ? (double)c->visits : 0.0;
        }
        double score = (double)sp_logp(node)[i]
                       + base * g->cfg.c_scale * q
                       - n_a / (1.0 + (double)total_n);
        if (score > best_score) {
            best = i;
            best_score = score;
        }
    }
    assert(best >= 0);
    return best;
}

/* ------------------------------------------------------------------ */
/* Descent helpers (shared by both scheduling modes)                   */
/* ------------------------------------------------------------------ */

static void descent_push(SpGame *g, SpDescent *d, uint32_t idx) {
    if (d->growable) {
        if (d->len == d->cap) {
            int ncap = d->cap ? d->cap * 2 : 16;
            uint32_t *np = arena_alloc(ARENA(g), (size_t)ncap * 4u);
            assert(np);
            if (d->len) memcpy(np, d->path, (size_t)d->len * 4u);
            d->path = np;
            d->cap = ncap;
        }
    } else {
        assert(d->len < d->cap);
    }
    d->path[d->len++] = idx;
}

/* Back a leaf value up along a descent path, sign-flipping every ply
 * (reference _backup). Stats are each node's OWN mover's view; the leaf
 * itself is visited. Pure accumulation — the scheduler tick is per-mode
 * (the sequential cursor here; the wave's dstate in sp_deliver_node). */
static void backup_path(SpGame *g, const uint32_t *path, int len, double v) {
    double sign = 1.0;
    for (int i = len - 1; i >= 0; i--) {
        SpNode *n = arena_node(ARENA(g), path[i]);
        n->visits += 1;
        n->value_sum += sign * v;
        sign = -sign;
    }
}

/* Sequential-mode scheduler tick: each completed descent advances the
 * candidate-major cursor to the next play (or candidate) of the round. */
static void seq_tick(SpGame *g) {
    g->seq.len = 0;              /* the descent slot is free again */
    g->cur_play++;
    if (g->cur_play >= g->plays) {
        g->cur_play = 0;
        g->cur_cand++;
    }
}

/* One halving step (reference _halve): keep the top ⌈|A|/2⌉ candidates by
 * z(a) + σ(q(a)), ties by enumeration order. Recomputes the round's plays
 * from the SURVIVOR count; `rounds` stays fixed (from the original m). */
static void halve(SpGame *g) {
    SpNode *root = arena_node(ARENA(g), g->root);
    const uint32_t *child = sp_child(root);

    uint32_t max_n = 0;
    for (int j = 0; j < g->n_cand; j++) {
        const SpNode *c = arena_node(ARENA(g), child[g->cand[j]]);
        assert(c->visits > 0);   /* m ≤ n ⇒ round-1 coverage */
        if (c->visits > max_n) max_n = c->visits;
    }
    double base = g->cfg.c_visit + (double)max_n;

    double score[SP_MAX_CANDIDATES];
    for (int j = 0; j < g->n_cand; j++) {
        const SpNode *c = arena_node(ARENA(g), child[g->cand[j]]);
        score[j] = g->z[g->cand[j]]
                   + base * g->cfg.c_scale * (-(c->value_sum / (double)c->visits));
    }

    /* Insertion sort, score descending, enumeration order ascending on
     * ties — the reference's sorted(key=(-score, rank)) with rank = the
     * enumeration index. n ≤ 64: the simple sort is the deterministic one. */
    int order[SP_MAX_CANDIDATES];
    for (int j = 0; j < g->n_cand; j++) order[j] = j;
    for (int i = 1; i < g->n_cand; i++) {
        int x = order[i];
        int y = i - 1;
        while (y >= 0) {
            int o = order[y];
            if (score[o] > score[x]) break;
            if (score[o] == score[x] && g->cand[o] <= g->cand[x]) break;
            order[y + 1] = order[y];
            y--;
        }
        order[y + 1] = x;
    }

    int keep = (g->n_cand + 1) / 2;
    uint16_t next[SP_MAX_CANDIDATES];
    for (int j = 0; j < keep; j++)
        next[j] = g->cand[order[j]];
    memcpy(g->cand, next, (size_t)keep * sizeof next[0]);
    g->n_cand = keep;

    g->plays = ceil_div(g->cfg.n, g->rounds * g->n_cand);
    g->cur_cand = 0;
    g->cur_play = 0;
}

/* ------------------------------------------------------------------ */
/* Move finalization (reference _finalize + _play_one_move tail)       */
/* ------------------------------------------------------------------ */

/* Softmax in the reference's exact evaluation order (_softmax). */
static void softmax_in_order(const double *raw, int n, double *out) {
    double m = raw[0];
    for (int i = 1; i < n; i++)
        if (raw[i] > m) m = raw[i];
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(raw[i] - m);
        total += out[i];
    }
    for (int i = 0; i < n; i++)
        out[i] /= total;
}

static void append_record(SpGame *g, const SpRec *r) {
    assert(g->n_recs < MAX_PLY);   /* a game cannot exceed MAX_PLY plies */
    g->recs[g->n_recs++] = *r;
}

/* Sample ∝ π′ (reference _sample_pi): one uniform() draw, resolved by
 * inverse CDF over the candidate set A₀ in ENUMERATION order. `pi` is
 * aligned with a0 (z-descending); the order of summation is rank order. */
static int sample_pi(SpGame *g, const double *pi) {
    int order[SP_MAX_CANDIDATES];
    for (int j = 0; j < g->m0; j++) order[j] = j;
    for (int i = 1; i < g->m0; i++) {
        int x = order[i];
        int y = i - 1;
        while (y >= 0 && g->a0[order[y]] > g->a0[x]) {
            order[y + 1] = order[y];
            y--;
        }
        order[y + 1] = x;
    }

    double u = rng_uniform(&g->rng);
    double cum = 0.0;
    for (int j = 0; j < g->m0; j++) {
        cum += pi[order[j]];
        if (u < cum)
            return g->a0[order[j]];
    }
    return g->a0[order[g->m0 - 1]];   /* fp guard: last candidate */
}

/* finalize_move: π′ over A₀, resignation check, move choice, record, push,
 * handoff (arena swap with subtree copy). */
static void finalize_move(SpGame *g) {
    SpNode *root = arena_node(ARENA(g), g->root);
    const uint32_t *child = sp_child(root);
    int mover = (int)g->st.side;

    /* completedQ per A₀ candidate (reference _finalize): the visited
     * child's q negated, or the Eq. 10 fallback v̂(root) − c_fpu — which is
     * the whole story for a forced position (m = 1, nothing visited).
     * π′, the record's support and the outlook all live over A₀ (the
     * ORIGINAL candidate set), not the halved survivor set. */
    double qs[SP_MAX_CANDIDATES];
    uint32_t max_n = 0;
    for (int j = 0; j < g->m0; j++) {
        int a = g->a0[j];
        if (child[a] != SP_NO_CHILD) {
            const SpNode *c = arena_node(ARENA(g), child[a]);
            if (c->visits > 0) {
                qs[j] = -(c->value_sum / (double)c->visits);
                if (c->visits > max_n) max_n = c->visits;
                continue;
            }
        }
        qs[j] = node_v(root) - g->cfg.c_fpu;
    }
    double base = g->cfg.c_visit + (double)max_n;

    double raw[SP_MAX_CANDIDATES], pi[SP_MAX_CANDIDATES];
    double outlook = -INFINITY;
    for (int j = 0; j < g->m0; j++) {
        raw[j] = (double)sp_logp(root)[g->a0[j]] + base * g->cfg.c_scale * qs[j];
        if (qs[j] > outlook) outlook = qs[j];
    }
    softmax_in_order(raw, g->m0, pi);

    /* Resignation (R4): the mover's best post-search value is hopeless.
     * The record still carries π′ and the outlook; no move is pushed. */
    if (outlook <= g->cfg.resign_value) {
        SpRec r;
        memset(&r, 0, sizeof r);
        r.ply = (uint16_t)g->st.ply;
        r.mover = (uint8_t)mover;
        r.played_slot = 0xFFFF;          /* resign: no move played */
        r.n_pi = (uint8_t)g->m0;
        for (int j = 0; j < g->m0; j++) {
            r.pi_slot[j] = sp_slots(root)[g->a0[j]];
            r.pi_prob[j] = (float)pi[j];
        }
        r.outlook = (float)outlook;
        append_record(g, &r);
        g->resigned = mover;
        g->phase = SP_PH_START_MOVE;
        return;
    }

    /* Move choice: sample ∝ π′ (training default: every ply), or the
     * halving survivor — the eval-measurement strategy (R5). */
    int played;
    if (g->cfg.sample_plies < 0 || (int)g->st.ply < g->cfg.sample_plies)
        played = sample_pi(g, pi);
    else
        played = g->cand[0];

    SpRec r;
    memset(&r, 0, sizeof r);
    r.ply = (uint16_t)g->st.ply;
    r.mover = (uint8_t)mover;
    r.played_slot = sp_slots(root)[played];
    r.played_move = (uint16_t)sp_moves(root)[played];
    r.n_pi = (uint8_t)g->m0;
    for (int j = 0; j < g->m0; j++) {
        r.pi_slot[j] = sp_slots(root)[g->a0[j]];
        r.pi_prob[j] = (float)pi[j];
    }
    r.outlook = (float)outlook;
    append_record(g, &r);

    push_move(g, sp_moves(root)[played]);

    /* Handoff (F4): the played child — always evaluated, subtree included —
     * is the next search's root. A forced move's child was never created,
     * so the chain ends naturally. The subtree is compacted into the OTHER
     * fixed region (ping-pong: the copy can never overlap its source) and
     * the dead tree is reclaimed by a pointer reset — no frees. */
    uint32_t hidx = child[played];
    SpArena *old = ARENA(g);
    g->cur_arena ^= 1;
    SpArena *fresh = ARENA(g);
    fresh->used = 0;
    fresh->n_nodes = 0;
    g->handoff = (hidx != SP_NO_CHILD) ? copy_subtree(fresh, old, hidx)
                                       : SP_NO_NODE;
    g->root = SP_NO_NODE;
    g->phase = SP_PH_START_MOVE;
}

/* ------------------------------------------------------------------ */
/* Candidate setup (the root: Gumbel top-m)                            */
/* ------------------------------------------------------------------ */

static void start_candidates(SpGame *g) {
    SpNode *root = arena_node(ARENA(g), g->root);
    int k = root->n_legal;
    assert(k >= 1 && k <= MAX_MOVES);

    /* One Gumbel per legal action, enumeration order — the RNG contract's
     * first consumption of the move (a forced position draws exactly one:
     * the general path has no k == 1 special case, round 6). */
    const double *lp = sp_logp(root);
    for (int i = 0; i < k; i++)
        g->z[i] = lp[i] + rng_gumbel(&g->rng);

    int m = g->cfg.n < g->cfg.m_cap ? g->cfg.n : g->cfg.m_cap;
    if (m > k) m = k;
    assert(m >= 1 && m <= SP_MAX_CANDIDATES);

    /* Top-m by (z desc, enumeration order asc) — partial selection with a
     * strict `>` keeps the earliest index on ties, matching the
     * reference's stable sort on key (-z, rank). */
    uint8_t taken[MAX_MOVES] = {0};
    for (int j = 0; j < m; j++) {
        int best = -1;
        for (int i = 0; i < k; i++) {
            if (taken[i]) continue;
            if (best < 0 || g->z[i] > g->z[best]) best = i;
        }
        taken[best] = 1;
        g->cand[j] = (uint16_t)best;
        g->a0[j] = (uint16_t)best;
    }
    g->n_cand = m;
    g->m0 = m;
    g->rounds = bit_length_m1(m);

    if (m == 1) {
        /* Forced (or n == 1): the halving loop never runs; finalize handles
         * the unvisited-candidate Eq. 10 fallback. */
        g->seq.len = 0;
        g->phase = SP_PH_FINAL;
    } else {
        g->plays = ceil_div(g->cfg.n, g->rounds * m);
        g->cur_cand = 0;
        g->cur_play = 0;
        g->seq.len = 0;
        g->phase = SP_PH_ROUND;
    }
}

/* ------------------------------------------------------------------ */
/* Descent (reference _descend, as a resumable loop)                   */
/* ------------------------------------------------------------------ */

static void fill_req(SpGame *g, SpReq *req, uint32_t node);  /* wave park */

/* Run the active descent until it parks on a fresh leaf (returns 1) or
 * completes at a terminal (returns 0 — the backup already ran). Invariant:
 * the board stands on the path's last node; path[0] is the root.
 *
 * The park differs by mode: sequential (req == NULL) parks the whole game
 * on the leaf (request served through the game's accessors, f32 planes);
 * wave mode (req != NULL) fills the engine-owned request — planes copied
 * and quantized, slots in enumeration order — because sibling descents of
 * the wave park concurrently and each request must own its payload (the
 * game's scratch is reused by the next walk). */
static int descend_run(SpGame *g, SpDescent *d, SpReq *req) {
    for (;;) {
        SpNode *node = arena_node(ARENA(g), d->path[d->len - 1]);
        if (node->state == SP_TERMINAL) {
            backup_path(g, d->path, d->len, node_v(node));
            if (!req)
                seq_tick(g);
            return 0;
        }
        assert(node->state == SP_EVALUATED);

        int slot;
        if (d->forced >= 0) {
            slot = d->forced;       /* the SH schedule's forced first step */
            d->forced = -1;
        } else {
            slot = select_eq14(g, node);
        }

        uint32_t cidx = sp_child(node)[slot];
        if (cidx != SP_NO_CHILD) {
            push_move(g, sp_moves(node)[slot]);
            descent_push(g, d, cidx);
            continue;               /* walk into the existing child */
        }

        /* Fresh child: the board is pushed, the node created; a NEW child
         * parks right here (the one park point of the whole engine — the
         * batch request of the parallel design), a fresh TERMINAL child is
         * backed up at the loop top. */
        push_move(g, sp_moves(node)[slot]);
        uint32_t nidx = make_child(g, node, slot);
        assert(nidx != SP_NO_NODE);
        sp_child(node)[slot] = nidx;
        descent_push(g, d, nidx);

        SpNode *child = arena_node(ARENA(g), nidx);
        if (child->state != SP_NEW)
            continue;               /* fresh TERMINAL: loop top backs it up */

        if (req) {
            /* Wave-mode park: the request owns its payload (planes, mover,
             * legal slots in enumeration order). */
            fill_req(g, req, nidx);
            req->is_root = 0;
        } else {
            /* Sequential park: the request is served through the game. */
            nn_encode(&g->st, &g->nn, g->planes);
            g->parked_node = nidx;
            g->park_root = 0;
            g->parked = 1;
        }
        return 1;
    }
}

/* ------------------------------------------------------------------ */
/* The state machine                                                   */
/* ------------------------------------------------------------------ */

int sp_game_step(SpGame *g) {
    assert(!g->parked && "deliver the pending evaluation before stepping");
    for (;;) {
        switch (g->phase) {
        case SP_PH_START_MOVE: {
            if (g->resigned >= 0) {
                g->ended = SP_END_RESIGN;
                return SP_EVENT_DONE;
            }
            int t = state_termination(&g->st);
            if (t != TERM_PLAYING) {
                g->ended = SP_END_NATURAL;
                g->term_code = t;
                return SP_EVENT_DONE;
            }
            if (g->max_plies >= 0 && (int)g->st.ply >= g->max_plies) {
                g->ended = SP_END_ADJUDICATED;
                return SP_EVENT_DONE;
            }

            g->root_ply = (int)g->st.ply;
            if (g->handoff != SP_NO_NODE) {
                /* F4: the played child of the previous move IS this root —
                 * evaluated, children and stats included, zero evals. */
                g->root = g->handoff;
                g->handoff = SP_NO_NODE;
                assert(arena_node(ARENA(g), g->root)->state == SP_EVALUATED);
                g->phase = SP_PH_CANDIDATES;
                continue;
            }
            uint32_t ridx = new_node_here(g, 0);
            assert(ridx != SP_NO_NODE);
            g->root = ridx;
            nn_encode(&g->st, &g->nn, g->planes);
            g->parked_node = ridx;
            g->park_root = 1;
            g->parked = 1;
            return SP_EVENT_NEED_EVAL;
        }

        case SP_PH_CANDIDATES:
            start_candidates(g);
            continue;

        case SP_PH_ROUND:
            for (;;) {
                if (g->seq.len == 0) {
                    if (g->cur_cand >= g->n_cand) {
                        halve(g);
                        if (g->n_cand == 1) {
                            g->phase = SP_PH_FINAL;
                            break;
                        }
                    }
                    /* One descent for (cur_cand, cur_play), forced first
                     * step = the candidate (the SH schedule). */
                    rewind_to(g, g->root_ply);
                    g->seq.len = 0;
                    descent_push(g, &g->seq, g->root);
                    g->seq.forced = g->cand[g->cur_cand];
                }
                if (descend_run(g, &g->seq, NULL))
                    return SP_EVENT_NEED_EVAL;   /* parked on a fresh leaf */
                /* completed: seq_tick advanced the scheduler; loop head
                 * starts the next descent or ends the round. */
            }
            continue;

        case SP_PH_FINAL:
            /* The last descent left the board on its leaf; the reference
             * rewinds to the root ply before the driver's push ("the board
             * must stand on the root for the driver's push") — finalize's
             * record fen_before and the played move's push both need it. */
            rewind_to(g, g->root_ply);
            finalize_move(g);
            return SP_EVENT_MOVED;

        default:
            assert(0 && "unreachable phase");
            return SP_EVENT_DONE;
        }
    }
}

void sp_deliver(SpGame *g, const double *logp, const double *wdl) {
    assert(g->parked);
    SpNode *n = arena_node(ARENA(g), g->parked_node);
    assert(n->state == SP_NEW);
    double *dst = sp_logp(n);
    for (int i = 0; i < n->n_legal; i++)
        dst[i] = logp[i];
    n->wdl[0] = wdl[0];
    n->wdl[1] = wdl[1];
    n->wdl[2] = wdl[2];
    n->state = SP_EVALUATED;
    g->parked = 0;
    if (g->park_root) {
        g->park_root = 0;
        g->phase = SP_PH_CANDIDATES;
    } else {
        /* Fresh leaf: back the value up right here (path still ends at the
         * leaf); the scheduler tick inside seq_tick sets up the next
         * descent. */
        backup_path(g, g->seq.path, g->seq.len, node_v(n));
        seq_tick(g);
    }
}

/* ------------------------------------------------------------------ */
/* Lifecycle + request accessors + sequential driver                   */
/* ------------------------------------------------------------------ */

SpGame *sp_game_init(void *storage, const SpConfig *cfg, const char *fen,
                     uint64_t seed, uint32_t game_index) {
    /* Every field a game carries is (re)initialized here — pool slots run
     * many games over their lifetime, and no calloc zeroes them. */
    assert(storage != NULL);
    assert(((uintptr_t)storage & 63u) == 0);      /* aligned slot */
    assert(cfg->m_cap <= SP_REC_CANDIDATES);      /* record layout bound */
    SpGame *g = storage;
    g->cfg = *cfg;
    g->seed = seed;
    g->game_index = game_index;
    if (!fen || !*fen) {
        state_init_startpos(&g->st);
    } else if (fen_parse(&g->st, fen) != 0) {
        return NULL;
    }
    g->start = g->st;               /* replay base for sp_rec_fen_before */
    fen_serialize(&g->st, g->start_fen, CHESS_FEN_MAX);
    nn_history_reset(&g->nn);
    nn_history_record(&g->nn, &g->st);
    rng_init(&g->rng, rng_game_seed(seed, (uint64_t)game_index));
    for (int r = 0; r < 2; r++)
        arena_init(&g->arenas[r], g->regions[r]);
    g->cur_arena = 0;
    g->phase = SP_PH_START_MOVE;
    g->root = SP_NO_NODE;
    g->handoff = SP_NO_NODE;
    g->seq.path = g->seq_path;
    g->seq.cap = SP_MAX_PATH;
    g->seq.growable = 0;
    g->seq.forced = -1;
    g->seq.len = 0;
    g->wave = 0;
    g->wave_pending = 0;
    g->root_parked = 0;
    g->wave_mode = 0;
    for (int j = 0; j < SP_MAX_CANDIDATES; j++) {
        g->dstate[j] = SP_D_RUNNABLE;
        g->flights[j] = NULL;
    }
    g->parked = 0;
    g->park_root = 0;
    g->parked_node = SP_NO_NODE;
    g->n_recs = 0;
    g->max_plies = -1;
    g->resigned = -1;
    g->ended = SP_END_PLAYING;
    g->term_code = 0;
    return g;
}

void sp_set_max_plies(SpGame *g, int max_plies) { g->max_plies = max_plies; }

const float *sp_request_planes(const SpGame *g) {
    assert(g->parked);
    return g->planes;
}

const uint16_t *sp_request_slots(const SpGame *g) {
    assert(g->parked);
    return sp_slots(arena_node(ARENA(g), g->parked_node));
}

int sp_request_count(const SpGame *g) {
    assert(g->parked);
    return arena_node(ARENA(g), g->parked_node)->n_legal;
}

int sp_request_mover(const SpGame *g) {
    assert(g->parked);
    return (int)g->st.side;
}

int sp_request_is_root(const SpGame *g) {
    assert(g->parked);
    return g->park_root;
}

void sp_play(SpGame *g, SpEvalFn fn, void *ctx, int max_plies) {
    g->max_plies = max_plies;
    double logp[MAX_MOVES], wdl[3];
    for (;;) {
        int ev = sp_game_step(g);
        if (ev == SP_EVENT_DONE) return;
        if (ev == SP_EVENT_NEED_EVAL) {
            fn(ctx, g, logp, wdl);
            sp_deliver(g, logp, wdl);
        }
    }
}

/* ------------------------------------------------------------------ */
/* Records + result                                                    */
/* ------------------------------------------------------------------ */

int sp_record_count(const SpGame *g) { return g->n_recs; }

int sp_rec_ply(const SpGame *g, int i) {
    return (i >= 0 && i < g->n_recs) ? g->recs[i].ply : -1;
}

int sp_rec_mover(const SpGame *g, int i) {
    return (i >= 0 && i < g->n_recs) ? g->recs[i].mover : -1;
}

/* FEN before record i, reconstructed by replaying the game's first i
 * moves from the stored start position (records carry no FEN string —
 * deviation 9). Identical bytes to the live fen_serialize the search
 * performed, since it is the same serializer on a replayed State. */
void sp_rec_fen_before(const SpGame *g, int i, char *out) {
    if (i < 0 || i > g->n_recs) {
        out[0] = '\0';
        return;
    }
    State tmp = g->start;
    for (int k = 0; k < i; k++)
        make_move(&tmp, (Move)g->recs[k].played_move);
    fen_serialize(&tmp, out, CHESS_FEN_MAX);
}

int sp_rec_played_slot(const SpGame *g, int i) {
    if (i < 0 || i >= g->n_recs) return -1;
    return g->recs[i].played_slot == 0xFFFF ? -1 : g->recs[i].played_slot;
}

void sp_rec_played_uci(const SpGame *g, int i, char out[8]) {
    if (i >= 0 && i < g->n_recs && g->recs[i].played_slot != 0xFFFF)
        move_to_uci((Move)g->recs[i].played_move, out);
    else
        out[0] = '\0';
}

int sp_rec_n_pi(const SpGame *g, int i) {
    return (i >= 0 && i < g->n_recs) ? g->recs[i].n_pi : 0;
}

int sp_rec_pi_slot(const SpGame *g, int i, int j) {
    if (i < 0 || i >= g->n_recs || j < 0 || j >= g->recs[i].n_pi) return -1;
    return g->recs[i].pi_slot[j];
}

double sp_rec_pi_prob(const SpGame *g, int i, int j) {
    if (i < 0 || i >= g->n_recs || j < 0 || j >= g->recs[i].n_pi) return 0.0;
    return (double)g->recs[i].pi_prob[j];
}

double sp_rec_outlook(const SpGame *g, int i) {
    return (i >= 0 && i < g->n_recs) ? g->recs[i].outlook : 0.0;
}

int sp_result_kind(const SpGame *g) { return g->ended; }

int sp_result_resigner(const SpGame *g) { return g->resigned; }

double sp_result_z_white(const SpGame *g) {
    switch (g->ended) {
    case SP_END_RESIGN:
        return g->resigned == 0 ? -1.0 : 1.0;  /* the resigner loses */
    case SP_END_NATURAL:
        if (g->term_code == TERM_CHECKMATE)
            return g->st.side == BLACK ? 1.0 : -1.0;  /* to-move is mated */
        return 0.0;
    default:
        return 0.0;   /* draws and adjudication */
    }
}

void sp_result_start_fen(const SpGame *g, char *out) {
    memcpy(out, g->start_fen, CHESS_FEN_MAX);
}

int sp_result_game_index(const SpGame *g) { return (int)g->game_index; }

int sp_result_n_sims(const SpGame *g) { return g->cfg.n; }

/* ------------------------------------------------------------------ */
/* Wave mode (Phase 3.3b): the parallel engine's game machine          */
/* ------------------------------------------------------------------ */

/* Fill a wave-mode request for `node` (which must be NEW and standing on
 * the board): quantized planes + legal slots in enumeration order. */
static void fill_req(SpGame *g, SpReq *req, uint32_t node) {
    SpNode *n = arena_node(ARENA(g), node);
    assert(n->state == SP_NEW);
    nn_encode(&g->st, &g->nn, g->planes);
    quantize_planes(g->planes, req->planes_u8);
    req->node = node;
    req->mover = (uint8_t)g->st.side;
    req->n_legal = n->n_legal;
    memcpy(req->slots, sp_slots(n), (size_t)n->n_legal * sizeof req->slots[0]);
}

void sp_game_set_wave_mode(SpGame *g) {
    assert(g->phase == SP_PH_START_MOVE && !g->parked && !g->root_parked);
    g->wave_mode = 1;
    g->wave = 0;
    g->wave_pending = 0;
    g->root_parked = 0;
}

void sp_game_set_owner(SpGame *g, int owner) { g->owner = owner; }
int  sp_game_owner(const SpGame *g)          { return g->owner; }

int sp_game_pending(const SpGame *g) {
    return g->root_parked || g->wave_pending > 0;
}

/* One wave-mode step (see chess_search.h for the event contract). The
 * scheduler starts the wave's descents in candidate order, one per call;
 * the wave barrier (all started descents backed up) gates the next wave. */
int sp_game_step_wave(SpGame *g, SpReq *req) {
    assert(g->wave_mode && req && !g->parked);
    for (;;) {
        switch (g->phase) {
        case SP_PH_START_MOVE: {
            if (g->resigned >= 0) {
                g->ended = SP_END_RESIGN;
                return SP_STEP_DONE;
            }
            int t = state_termination(&g->st);
            if (t != TERM_PLAYING) {
                g->ended = SP_END_NATURAL;
                g->term_code = t;
                return SP_STEP_DONE;
            }
            if (g->max_plies >= 0 && (int)g->st.ply >= g->max_plies) {
                g->ended = SP_END_ADJUDICATED;
                return SP_STEP_DONE;
            }
            if (g->root_parked)
                return SP_STEP_IDLE;   /* root eval outstanding */

            g->root_ply = (int)g->st.ply;
            if (g->handoff != SP_NO_NODE) {
                /* F4: the played child of the previous move IS this root —
                 * evaluated, children and stats included, zero evals. */
                g->root = g->handoff;
                g->handoff = SP_NO_NODE;
                assert(arena_node(ARENA(g), g->root)->state == SP_EVALUATED);
                g->phase = SP_PH_CANDIDATES;
                continue;
            }
            uint32_t ridx = new_node_here(g, 0);
            assert(ridx != SP_NO_NODE);
            g->root = ridx;
            fill_req(g, req, ridx);
            req->is_root = 1;
            g->root_parked = 1;
            return SP_STEP_PARKED;
        }

        case SP_PH_CANDIDATES:
            start_candidates(g);
            if (g->phase == SP_PH_FINAL)
                continue;              /* forced position: no waves */
            /* Round schedule: waves of |A| descents, candidate order. */
            g->wave = 0;
            g->wave_pending = 0;
            for (int j = 0; j < g->n_cand; j++)
                g->dstate[j] = SP_D_RUNNABLE;
            g->phase = SP_PH_ROUND;
            continue;

        case SP_PH_ROUND: {
            /* Start the lowest-index runnable descent of the current
             * wave (candidate order — the wave discipline's order). */
            for (int j = 0; j < g->n_cand; j++) {
                if (g->dstate[j] != SP_D_RUNNABLE)
                    continue;
                SpDescent *d = arena_alloc(ARENA(g), sizeof *d);
                assert(d);
                d->path = NULL;
                d->len = d->cap = 0;
                d->growable = 1;
                rewind_to(g, g->root_ply);
                descent_push(g, d, g->root);
                d->forced = g->cand[j];
                g->flights[j] = d;
                g->dstate[j] = SP_D_RUNNING;
                int parked = descend_run(g, d, req);
                if (parked) {
                    g->dstate[j] = SP_D_PARKED;
                    g->wave_pending++;
                    return SP_STEP_PARKED;
                }
                /* reached a terminal inline: backed up, wave slot done */
                g->dstate[j] = SP_D_DONE;
                return SP_STEP_PROGRESS;
            }

            /* Nothing runnable: the barrier (parked descents await their
             * evaluations) or the wave is complete. */
            if (g->wave_pending > 0)
                return SP_STEP_IDLE;

            g->wave++;
            if (g->wave >= g->plays) {
                halve(g);
                if (g->n_cand == 1) {
                    g->phase = SP_PH_FINAL;
                    continue;
                }
            }
            /* next wave (or the new round's first wave after halving) */
            for (int j = 0; j < g->n_cand; j++)
                g->dstate[j] = SP_D_RUNNABLE;
            g->wave_pending = 0;
            continue;
        }

        case SP_PH_FINAL:
            /* The wave barrier guarantees every descent backed up; the
             * board stands wherever the last walk left it — rewind to the
             * root for the driver's push (as in the sequential mode). */
            rewind_to(g, g->root_ply);
            finalize_move(g);
            return SP_STEP_PROGRESS;

        default:
            assert(0 && "unreachable phase");
            return SP_STEP_DONE;
        }
    }
}

void sp_deliver_node(SpGame *g, uint32_t node, const double *logp,
                     const double *wdl) {
    assert(g->wave_mode);

    if (g->root_parked && node == g->root) {
        SpNode *n = arena_node(ARENA(g), g->root);
        assert(n->state == SP_NEW);
        double *dst = sp_logp(n);
        for (int i = 0; i < n->n_legal; i++)
            dst[i] = logp[i];
        n->wdl[0] = wdl[0];
        n->wdl[1] = wdl[1];
        n->wdl[2] = wdl[2];
        n->state = SP_EVALUATED;
        g->root_parked = 0;
        g->phase = SP_PH_CANDIDATES;
        return;
    }

    /* Find the parked flight whose leaf is `node` — unique among the
     * wave's parked descents (each created its own fresh leaf; encountering
     * anything else is a protocol bug and must fail loudly, F3). */
    int j = -1;
    for (int i = 0; i < g->n_cand; i++) {
        if (g->dstate[i] != SP_D_PARKED)
            continue;
        const SpDescent *d = g->flights[i];
        if (d->len > 0 && d->path[d->len - 1] == node) {
            j = i;
            break;
        }
    }
    assert(j >= 0 && "delivery for a node no parked descent owns");

    SpDescent *d = g->flights[j];
    SpNode *leaf = arena_node(ARENA(g), node);
    assert(leaf->state == SP_NEW);
    double *dst = sp_logp(leaf);
    for (int i = 0; i < leaf->n_legal; i++)
        dst[i] = logp[i];
    leaf->wdl[0] = wdl[0];
    leaf->wdl[1] = wdl[1];
    leaf->wdl[2] = wdl[2];
    leaf->state = SP_EVALUATED;
    backup_path(g, d->path, d->len, node_v(leaf));
    g->dstate[j] = SP_D_DONE;
    g->wave_pending--;
}

/* The static game must fit the published slot size (deviation 9). */
_Static_assert(sizeof(SpGame) <= SP_GAME_SLOT_SIZE,
               "SpGame outgrew SP_GAME_SLOT_SIZE — raise the constant");

size_t sp_game_slot_size(void) { return SP_GAME_SLOT_SIZE; }
