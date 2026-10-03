/*
 * az.c — the AZ-PUCT evaluation search core (chess_az.h).
 *
 * A faithful port of scripts/az_eval.py's search(): the walk, the backup
 * and every tie-break are the Python reference's, reshaped into the same
 * park/deliver state machine the Gumbel search uses (chess_search.h), so
 * the Python driver half (a pull loop fulfilling az_request_* via any
 * evaluator — including the ladder's GPU-batching service) is shared
 * machinery, not new design. Read chess_az.h's header first: it documents
 * the contract, the Q-sign flag, and the deliberate deviations.
 *
 * Parity obligations honored here (each is load-bearing for the fixture
 * equivalence test, tests/test_az_c.py):
 *   - selection scans the legal moves in movegen enumeration order with
 *     strict > (first maximum wins — Python's dict insertion order IS this
 *     order: Board.legal_moves comes from this generator, and az_eval.py
 *     inserts children in that order);
 *   - all arithmetic in IEEE double with the reference's evaluation order
 *     (q + c_puct*prior*sqrt(total)/(1+visits), left-associative; the
 *     build is -ffp-contract=off, so no FMA collapses a rounding);
 *   - priors: exp(logp[i]) summed in enumeration order, each prior divided
 *     by that total (a re-computed exp is bit-identical to a cached one);
 *   - backup: v sign-flipped per ply from the leaf, so every node's
 *     value_sum is its OWN mover's view; total in selection is the child
 *     visit sum over ALL children (lazily created children simply do not
 *     exist yet — their q is 0.0 and visits 0, exactly Python's unvisited
 *     Node);
 *   - exactly `sims` simulations; one evaluation request per fresh leaf
 *     (the root's own expansion is simulation 1); the tree is discarded
 *     and rebuilt from scratch on every az_think().
 *
 * Board discipline (same as search.c): nodes store no position. The walk
 * makes/unmakes moves on the game's working State, and the NN history ring
 * is kept in step on every push/pop, so the encoding taken while parked
 * sees the correct 8-ply window. A handle is not thread-safe; handles are
 * independent (no mutable globals).
 */
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "chess_ffi.h"
#include "chess_nn.h"
#include "chess_az.h"

/* ------------------------------------------------------------------ */
/* Node — arena-allocated, variable size                               */
/* ------------------------------------------------------------------ */

/* Node states (az_eval.py's terminal_v-is-set / children-exist lattice). */
enum { AZ_NEW = 0, AZ_EVALUATED = 1, AZ_TERMINAL = 2 };

#define AZ_NO_NODE  0xFFFFFFFFu
#define AZ_NO_CHILD 0xFFFFFFFFu

/*
 * Fixed header, then a variable tail sized by the legal-move count k,
 * laid out so the doubles stay 8-aligned (the same trick as search.c's
 * SpNode, with prior in logp's seat):
 *
 *   [header] [double prior[k]] [u32 child[k]] [Move moves[k]] [u16 slots[k]]
 *
 * prior[i] is P(s,a) — the net's policy_logp over the legal slots,
 * renormalized by the masked softmax at delivery time. child[i] is the
 * arena index of the child for legal move i, AZ_NO_CHILD until a descent
 * first arrives there (lazy materialization: an unvisited child is
 * exactly Python's visits-0 Node with q 0.0, and its prior already lives
 * in this node's tail). moves[i]/slots[i] fix the movegen enumeration:
 * the order of the logp delivery, of selection's tie-breaks, and of the
 * az_root_* getters.
 */
typedef struct {
    double   value_sum;   /* Σ sign·v — the node's OWN mover's view       */
    double   wdl[3];      /* delivered (pW,pD,pL) / terminal one-hot      */
    uint32_t visits;      /* backed-up simulations through this node      */
    uint16_t n_legal;     /* k: legal moves in movegen order              */
    Move     move;        /* the move from the parent leading here (root 0)*/
    uint8_t  state;       /* AZ_NEW / AZ_EVALUATED / AZ_TERMINAL          */
    uint8_t  pad_[7];
} AzNode;

#define AZ_NODE_HDR (((sizeof(AzNode)) + 7u) & ~(size_t)7u)   /* 48 */

static double *az_prior(const AzNode *n) {
    return (double *)((const char *)n + AZ_NODE_HDR);
}
static uint32_t *az_child(const AzNode *n) {
    return (uint32_t *)((const char *)n + AZ_NODE_HDR + (size_t)n->n_legal * 8u);
}
static Move *az_moves(const AzNode *n) {
    return (Move *)((const char *)n + AZ_NODE_HDR + (size_t)n->n_legal * 12u);
}
static uint16_t *az_slots(const AzNode *n) {
    return (uint16_t *)((const char *)n + AZ_NODE_HDR + (size_t)n->n_legal * 14u);
}

/* The node's own value v = pW − pL (own mover's view; terminal one-hot). */
static double node_v(const AzNode *n) { return n->wdl[0] - n->wdl[2]; }

/* ------------------------------------------------------------------ */
/* The game                                                            */
/* ------------------------------------------------------------------ */

struct AzGame {
    State     st;               /* the working position                   */
    NnHistory nn;               /* encoder history ring, in step with st  */
    Move      pushed[MAX_PLY];  /* move log: pushed[i] led to ply i+1     */

    /* Current search (set by az_think; root == AZ_NO_NODE before/none). */
    int       sims, sims_done;
    double    c_puct;
    int       q_parent;         /* 0 = child view (verbatim), 1 = paper   */
    uint32_t  root;
    int       root_ply;         /* st.ply when this search began          */
    int       parked;
    uint32_t  parked_node;      /* the node awaiting its evaluation       */
    int       done;
    uint32_t  path[MAX_PLY + 2];/* active descent, root first             */
    int       path_len;
    float     planes[NN_TOTAL]; /* encoding of the parked position        */

    /* Node arena: one bump region + an index→pointer table, reset every
     * az_think() (az_eval.py rebuilds the tree each move — no handoff).
     * Untouched pages of the generous region are never committed. */
    size_t    arena_used;
    uint32_t  n_nodes;
    AzNode   *nodes[AZ_MAX_NODES];
    _Alignas(8) char arena[AZ_ARENA_CAP];
};

/* ------------------------------------------------------------------ */
/* Arena                                                               */
/* ------------------------------------------------------------------ */

static void *arena_alloc(AzGame *g, size_t bytes) {
    bytes = (bytes + 7u) & ~(size_t)7u;
    assert(g->arena_used + bytes <= AZ_ARENA_CAP);
    if (g->arena_used + bytes > AZ_ARENA_CAP)
        return NULL;
    void *p = g->arena + g->arena_used;
    g->arena_used += bytes;
    return p;
}

static uint32_t arena_add_node(AzGame *g, AzNode *n) {
    assert(g->n_nodes < AZ_MAX_NODES && n != NULL);
    g->nodes[g->n_nodes] = n;
    return g->n_nodes++;
}

static AzNode *az_node(const AzGame *g, uint32_t idx) {
    assert(idx != AZ_NO_NODE && idx < g->n_nodes);
    return g->nodes[idx];
}

/* ------------------------------------------------------------------ */
/* Board helpers — make/unmake with the log + encoder ring in step      */
/* ------------------------------------------------------------------ */

static void push_move(AzGame *g, Move m) {
    assert((int)g->st.ply < MAX_PLY);
    make_move(&g->st, m);
    g->pushed[g->st.ply - 1] = m;
    nn_history_record(&g->nn, &g->st);
}

static void rewind_to(AzGame *g, int ply) {
    while ((int)g->st.ply > ply) {
        unmake_move(&g->st, g->pushed[g->st.ply - 1]);
        nn_history_drop(&g->nn);
    }
}

/* ------------------------------------------------------------------ */
/* Node construction                                                   */
/* ------------------------------------------------------------------ */

/* Create a node for the position the State stands on (must be PLAYING —
 * terminal positions go through make_child). Enumerates the legal moves
 * once: this list, its policy slots and the array order ARE the
 * enumeration the delivery alignment, every tie-break and the az_root_*
 * getters are specified over. The node stays AZ_NEW until delivery fills
 * its priors. */
static uint32_t new_node_here(AzGame *g, Move incoming) {
    MoveList ml;
    gen_legal(&g->st, &ml);
    assert(ml.count >= 1 && ml.count <= MAX_MOVES);
    int k = ml.count;

    AzNode *n = arena_alloc(g, AZ_NODE_HDR + (size_t)k * 16u);
    if (!n) return AZ_NO_NODE;
    memset(n, 0, AZ_NODE_HDR + (size_t)k * 16u);
    n->state = AZ_NEW;
    n->n_legal = (uint16_t)k;
    n->move = incoming;
    Move *mv = az_moves(n);
    uint16_t *sl = az_slots(n);
    uint32_t *ch = az_child(n);
    for (int i = 0; i < k; i++) {
        mv[i] = ml.moves[i];
        int slot = nn_policy_slot(ml.moves[i], (int)g->st.side);
        assert(slot >= 0 && slot < NN_POLICY_SIZE);
        sl[i] = (uint16_t)slot;
        ch[i] = AZ_NO_CHILD;
    }
    return arena_add_node(g, n);
}

/* The board has already been pushed to the child's position. A terminal
 * position builds a one-hot node (mate from the side-to-move's view =
 * loss; every other termination = draw — state_termination implements
 * the claim-draw semantics az_eval.py's board.termination reports); a
 * playing position builds a fresh AZ_NEW node. */
static uint32_t make_child(AzGame *g, Move incoming) {
    int t = state_termination(&g->st);
    if (t == TERM_PLAYING)
        return new_node_here(g, incoming);
    AzNode *c = arena_alloc(g, AZ_NODE_HDR);
    if (!c) return AZ_NO_NODE;
    memset(c, 0, AZ_NODE_HDR);
    c->state = AZ_TERMINAL;
    c->n_legal = 0;
    c->move = incoming;
    if (t == TERM_CHECKMATE) {
        c->wdl[0] = 0.0; c->wdl[1] = 0.0; c->wdl[2] = 1.0;
    } else {
        c->wdl[0] = 0.0; c->wdl[1] = 1.0; c->wdl[2] = 0.0;
    }
    return arena_add_node(g, c);
}

/* ------------------------------------------------------------------ */
/* Selection, backup                                                   */
/* ------------------------------------------------------------------ */

/* az_eval.py's selection, verbatim:
 *
 *     u_score = c.q + c_puct * c.prior * sqrt(total) / (1 + c.visits)
 *
 * total = the child-visit sum (missing children contribute 0); q is the
 * child's own-mover mean value (0.0 unvisited — Python's Node.q
 * property), negated first when q_parent_view selected the paper's
 * Q(s,a). Strict > scanning in enumeration order: the first maximum
 * wins, Python's dict-insertion tie-break. */
static int select_puct(const AzGame *g, const AzNode *node) {
    const uint32_t *child = az_child(node);
    const double *prior = az_prior(node);
    int k = node->n_legal;

    uint32_t total = 0;
    for (int i = 0; i < k; i++)
        if (child[i] != AZ_NO_CHILD)
            total += az_node(g, child[i])->visits;
    double root_total = sqrt((double)total);

    int best = -1;
    double best_score = -INFINITY;
    for (int i = 0; i < k; i++) {
        const AzNode *c = child[i] != AZ_NO_CHILD
                              ? az_node(g, child[i]) : NULL;
        uint32_t n_i = (c != NULL) ? c->visits : 0;
        double q = (c != NULL && n_i > 0)
                     ? c->value_sum / (double)n_i : 0.0;
        if (g->q_parent)
            q = -q;
        double score = q + g->c_puct * prior[i] * root_total
                             / (1.0 + (double)n_i);
        if (score > best_score) {
            best = i;
            best_score = score;
        }
    }
    assert(best >= 0);
    return best;
}

/* Backup with the sign flip per ply (leaf first, sign +1), then rewind
 * the working State to the search root for the next simulation. */
static void backup(AzGame *g, double v) {
    double sign = 1.0;
    for (int i = g->path_len - 1; i >= 0; i--) {
        AzNode *n = az_node(g, g->path[i]);
        n->visits += 1;
        n->value_sum += sign * v;
        sign = -sign;
    }
    rewind_to(g, g->root_ply);
}

/* ------------------------------------------------------------------ */
/* The state machine                                                   */
/* ------------------------------------------------------------------ */

/* Walk one simulation from the root to its fresh-or-terminal node,
 * making moves and materializing lazily-created children on the way. */
static void descend(AzGame *g) {
    assert((int)g->st.ply == g->root_ply);   /* between sims we sit at root */
    g->path_len = 1;
    g->path[0] = g->root;
    for (;;) {
        AzNode *node = az_node(g, g->path[g->path_len - 1]);
        if (node->state != AZ_EVALUATED)
            return;                     /* AZ_NEW → park; AZ_TERMINAL → rules */
        int i = select_puct(g, node);
        push_move(g, az_moves(node)[i]);
        uint32_t *link = &az_child(node)[i];
        if (*link == AZ_NO_CHILD)
            *link = make_child(g, az_moves(node)[i]);
        g->path[g->path_len++] = *link;
    }
}

void az_think(AzGame *g, int sims, double c_puct, int q_parent_view) {
    assert(sims >= 1 && sims <= AZ_MAX_SIMS);
    assert(!g->parked && g->done);      /* the previous search is finished */
    assert(state_termination(&g->st) == TERM_PLAYING);

    g->arena_used = 0;                  /* fresh tree, like az_eval.py     */
    g->n_nodes = 0;
    g->sims = sims;
    g->sims_done = 0;
    g->c_puct = c_puct;
    g->q_parent = q_parent_view;
    g->root_ply = (int)g->st.ply;
    g->parked = 0;
    g->done = 0;
    g->path_len = 0;
    g->root = new_node_here(g, 0);
    assert(g->root != AZ_NO_NODE);
}

int az_step(AzGame *g) {
    if (g->done)
        return AZ_EVENT_DONE;
    if (g->parked)
        return AZ_EVENT_NEED_EVAL;      /* deliver first */
    while (g->sims_done < g->sims) {    /* budget checked BEFORE descending:
                                         * the sim that exhausts it usually
                                         * ends in az_deliver's backup, not
                                         * here — a do-while would run one
                                         * unbudgeted sim per step forever */
        descend(g);
        AzNode *leaf = az_node(g, g->path[g->path_len - 1]);
        if (leaf->state == AZ_TERMINAL) {
            backup(g, node_v(leaf));
            g->sims_done++;
        } else {
            /* Park on the fresh leaf: the request IS this position. */
            nn_encode(&g->st, &g->nn, g->planes);
            g->parked = 1;
            g->parked_node = g->path[g->path_len - 1];
            return AZ_EVENT_NEED_EVAL;
        }
    }
    g->done = 1;
    return AZ_EVENT_DONE;
}

/* ------------------------------------------------------------------ */
/* The pending request / its fulfilment                                */
/* ------------------------------------------------------------------ */

const float *az_request_planes(const AzGame *g) { return g->planes; }
const uint16_t *az_request_slots(const AzGame *g) {
    return az_slots(az_node(g, g->parked_node));
}
int az_request_count(const AzGame *g) {
    return az_node(g, g->parked_node)->n_legal;
}
int az_request_mover(const AzGame *g) { return (int)g->st.side; }

void az_deliver(AzGame *g, const double *logp, const double *wdl) {
    assert(g->parked);
    AzNode *n = az_node(g, g->parked_node);
    int k = n->n_legal;

    /* Masked softmax over the legal slots: priors sum to 1 whatever the
     * net thinks of the illegal rest (az_eval.py's _priors_and_value). */
    double *prior = az_prior(n);
    double total = 0.0;
    for (int i = 0; i < k; i++) {
        prior[i] = exp(logp[i]);
        total += prior[i];
    }
    for (int i = 0; i < k; i++)
        prior[i] = prior[i] / total;

    n->wdl[0] = wdl[0];
    n->wdl[1] = wdl[1];
    n->wdl[2] = wdl[2];
    n->state = AZ_EVALUATED;
    g->parked = 0;
    backup(g, wdl[0] - wdl[2]);
    g->sims_done++;
}

void az_think_sync(AzGame *g, int sims, double c_puct, int q_parent_view,
                   AzEvalFn fn, void *ctx) {
    az_think(g, sims, c_puct, q_parent_view);
    double logp[MAX_MOVES], wdl[3];
    while (az_step(g) == AZ_EVENT_NEED_EVAL) {
        fn(ctx, g, logp, wdl);
        az_deliver(g, logp, wdl);
    }
}

/* ------------------------------------------------------------------ */
/* Root statistics                                                     */
/* ------------------------------------------------------------------ */

static const AzNode *root_node(const AzGame *g) {
    assert(g->done && g->root != AZ_NO_NODE);
    return az_node(g, g->root);
}

int az_root_count(const AzGame *g) { return root_node(g)->n_legal; }

void az_root_uci(const AzGame *g, int i, char out[CHESS_UCI_MAX]) {
    move_to_uci(az_moves(root_node(g))[i], out);
}

int az_root_visits(const AzGame *g, int i) {
    const AzNode *root = root_node(g);
    uint32_t c = az_child(root)[i];
    return c != AZ_NO_CHILD ? (int)az_node(g, c)->visits : 0;
}

double az_root_prior(const AzGame *g, int i) {
    return az_prior(root_node(g))[i];
}

double az_root_q(const AzGame *g, int i) {
    const AzNode *root = root_node(g);
    uint32_t c = az_child(root)[i];
    if (c == AZ_NO_CHILD)
        return 0.0;
    const AzNode *ch = az_node(g, c);
    return ch->visits ? ch->value_sum / (double)ch->visits : 0.0;
}

double az_root_value(const AzGame *g) {
    /* az_eval.py's effective search() second return: its root_v is dead
     * code (initialized None, never assigned), so the reference always
     * surfaces root.q — the visits-weighted backup value. Mirror that. */
    const AzNode *root = root_node(g);
    assert(root->visits > 0);
    return root->value_sum / (double)root->visits;
}

/* ------------------------------------------------------------------ */
/* Lifecycle and the game                                              */
/* ------------------------------------------------------------------ */

AzGame *az_game_new(const char *fen) {
    AzGame *g = malloc(sizeof *g);
    if (!g) return NULL;
    if (!fen || !*fen) {
        state_init_startpos(&g->st);
    } else if (fen_parse(&g->st, fen) != 0) {
        free(g);
        return NULL;
    }
    nn_history_reset(&g->nn);
    nn_history_record(&g->nn, &g->st);
    g->arena_used = 0;
    g->n_nodes = 0;
    g->sims = g->sims_done = 0;
    g->c_puct = 0.0;
    g->q_parent = 0;
    g->root = AZ_NO_NODE;
    g->root_ply = (int)g->st.ply;
    g->parked = 0;
    g->done = 1;               /* no search yet: az_step is a no-op DONE */
    g->path_len = 0;
    return g;
}

void az_game_free(AzGame *g) {
    free(g);
}

int az_play_uci(AzGame *g, const char *uci) {
    if (!uci || g->parked) return -1;
    size_t len = strlen(uci);
    if (len < 4 || len > 5) return -1;
    MoveList ml;
    gen_legal(&g->st, &ml);
    for (int i = 0; i < ml.count; i++) {
        char candidate[CHESS_UCI_MAX];
        move_to_uci(ml.moves[i], candidate);
        if (strcmp(candidate, uci) == 0) {
            make_move(&g->st, ml.moves[i]);
            g->pushed[g->st.ply - 1] = ml.moves[i];
            nn_history_record(&g->nn, &g->st);
            return 0;
        }
    }
    return -1;
}

/* state_termination / state_is_checkmate temporarily probe the State
 * (restored before returning), so these take a mutable handle — same
 * convention as ffi.c's one-line observation forwards. */
int az_termination(AzGame *g)            { return state_termination(&g->st); }
int az_winner(AzGame *g) {
    return state_termination(&g->st) == TERM_CHECKMATE
               ? 1 - (int)g->st.side : -1;
}
int  az_side(const AzGame *g)            { return (int)g->st.side; }
int  az_ply(const AzGame *g)             { return (int)g->st.ply; }
void az_fen(const AzGame *g, char *out)  { fen_serialize(&g->st, out,
                                                       CHESS_FEN_MAX); }
