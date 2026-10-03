/*
 * test_az.c — C unit tests for the AZ-PUCT evaluation search
 * (chess_az.h / az.c), in the hand-rolled CHECK style of test_units.c /
 * test_search.c.
 *
 * The deepest verification — the exact walk parity against the Python
 * reference (scripts/az_eval.py) under a deterministic fixture evaluator —
 * lives across the boundary in tests/test_az_c.py, where both engines run
 * in one process. What is pinned here is the C-only surface:
 *
 *   1. request/lifecycle shape: the root parks first, planes/slots/count/
 *      mover describe the parked position, the request count is aligned
 *      with the root's enumeration;
 *   2. priors: the masked-softmax renormalization (delivered logp →
 *      probabilities summing to 1, favorites keep their ranking);
 *   3. the Q-sign contract (chess_az.h's header): on a mate-in-1 with a
 *      value-free uniform net, the paper sign (q_parent_view=1) must pile
 *      visits on the mating move; the verbatim az_eval.py sign (=0) must
 *      avoid it — the empirical pin of the sign finding;
 *   4. terminal leaves never request an evaluation (az_eval.py evaluates
 *      and discards; the port skips the call outright);
 *   5. per-move arena reset + the game ops (play_uci/fen/ply/winner).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "chess_nn.h"
#include "chess_az.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (cond) {                                                        \
            g_pass++;                                                      \
        } else {                                                           \
            g_fail++;                                                      \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
        }                                                                  \
    } while (0)

/* Same board as test_search.c's MATE_FEN: white Kb6 + Rc1, black Ka8 —
 * Rc1-c8 is the UNIQUE mate in one (Ra1+ lets the king out via b8). */
#define MATE_FEN "k7/8/1K6/8/8/8/8/2R5 w - - 0 1"

/* ------------------------------------------------------------------ */
/* Fixture evaluators                                                  */
/* ------------------------------------------------------------------ */

typedef struct { int requests; } CountCtx;

/* Uniform net: flat logp over the legal slots, W=D=L — every non-terminal
 * evaluates to v = 0, so all steering visible in the tests comes from the
 * rules terminals (±1) and the PUCT arithmetic itself. */
static void uniform_eval(void *ctx, const AzGame *g, double *logp,
                         double *wdl) {
    CountCtx *c = ctx;
    c->requests++;
    int n = az_request_count(g);
    for (int i = 0; i < n; i++)
        logp[i] = 0.0;
    wdl[0] = 1.0 / 3.0; wdl[1] = 1.0 / 3.0; wdl[2] = 1.0 / 3.0;
}

/* ------------------------------------------------------------------ */
/* 1. request/lifecycle shape                                          */
/* ------------------------------------------------------------------ */

static void test_request_shape(void) {
    AzGame *g = az_game_new(NULL);

    az_think(g, 8, 1.25, 0);
    CHECK(az_step(g) == AZ_EVENT_NEED_EVAL);      /* the root parks first */
    CHECK(az_request_count(g) == 20);             /* startpos legal moves */
    CHECK(az_request_mover(g) == 0);
    int has_pawn = 0;
    const uint16_t *slots = az_request_slots(g);
    for (int i = 0; i < az_request_count(g); i++)
        if (slots[i] == 12 * 64 + 28) has_pawn = 1;   /* e2e4 */
    CHECK(has_pawn);
    /* planes are the same tensor chess_encode would produce here: spot
     * the piece one-hots (dims [0,96) per token, token-major stride 117)
     * — exactly 32 set squares at start, matching Board.encode(). */
    const float *pl = az_request_planes(g);
    int ones = 0;
    for (int t = 0; t < 64; t++)
        for (int d = 0; d < 96; d++)
            if (pl[t * 117 + d] == 1.0f) ones++;
    CHECK(ones == 32);

    /* Finish the search on the uniform net; done stays sticky. */
    double logp[MAX_MOVES], wdl[3];
    while (az_step(g) == AZ_EVENT_NEED_EVAL) {
        int n = az_request_count(g);
        for (int i = 0; i < n; i++) logp[i] = 0.0;
        wdl[0] = wdl[1] = wdl[2] = 1.0 / 3.0;
        az_deliver(g, logp, wdl);
    }
    CHECK(az_step(g) == AZ_EVENT_DONE);
    CHECK(az_root_count(g) == 20);
    CHECK(az_root_value(g) == 0.0);               /* uniform WDL ⇒ v = 0  */
    int total_child_visits = 0;
    for (int i = 0; i < az_root_count(g); i++)
        total_child_visits += az_root_visits(g, i);
    CHECK(total_child_visits == 7);   /* 8 sims − the root's own eval sim */

    az_game_free(g);
}

/* ------------------------------------------------------------------ */
/* 2. priors renormalization                                           */
/* ------------------------------------------------------------------ */

typedef struct { int requests; int fav; } FavCtx;

/* logp favors the FIRST legal slot by 4 nats; the net is otherwise flat
 * and slightly pessimistic (v = pW − pL = 0.2 − 0.5 = −0.3). */
static void fav_first_eval(void *ctx, const AzGame *g, double *logp,
                           double *wdl) {
    FavCtx *c = ctx;
    c->requests++;
    int n = az_request_count(g);
    for (int i = 0; i < n; i++)
        logp[i] = (i == c->fav) ? 0.0 : -4.0;
    wdl[0] = 0.2; wdl[1] = 0.3; wdl[2] = 0.5;
}

static void test_priors_normalization(void) {
    AzGame *g = az_game_new(NULL);
    FavCtx ctx = {0, 0};

    az_think_sync(g, 16, 1.25, 0, fav_first_eval, &ctx);

    double sum = 0.0, best = 0.0;
    int best_i = -1;
    for (int i = 0; i < az_root_count(g); i++) {
        double p = az_root_prior(g, i);
        sum += p;
        if (p > best) { best = p; best_i = i; }
    }
    CHECK(fabs(sum - 1.0) < 1e-12);
    CHECK(best_i == 0);
    CHECK(fabs(best - 1.0 / (1.0 + 19.0 * exp(-4.0))) < 1e-12);

    az_game_free(g);
}

/* ------------------------------------------------------------------ */
/* 3+4. the Q sign, and terminals never evaluate                       */
/* ------------------------------------------------------------------ */

/* Find the root child playing `uci`, -1 when absent. */
static int root_find(AzGame *g, const char *uci) {
    char buf[CHESS_UCI_MAX];
    for (int i = 0; i < az_root_count(g); i++) {
        az_root_uci(g, i, buf);
        if (strcmp(buf, uci) == 0) return i;
    }
    return -1;
}

static void test_mate_in_1_q_sign(void) {
    /* 20 legal moves (14 rook + 6 king), c1c8 the unique mate.
     *
     * Dynamics with a value-free uniform net (why the expectations look
     * like this): unvisited children carry NO Q estimate under either
     * sign, so the first ~20 sims rotate through the children in
     * enumeration order on the exploration bonus alone — the mate gets
     * exactly one rotation visit either way. From that moment its
     * terminal q (own view −1) enters selection:
     *   paper sign (+1 from the root's view) — the mate out-scores every
     *     quiet child forever after, so it takes ALL remaining sims;
     *   verbatim sign (−1 as scored) — the mate sits at the bottom and
     *     is never revisited, the quiet children share the rest. */
    for (int q_parent = 0; q_parent <= 1; q_parent++) {
        AzGame *g = az_game_new(MATE_FEN);
        CountCtx ctx = {0};
        az_think_sync(g, 60, 1.25, q_parent, uniform_eval, &ctx);

        int mate = root_find(g, "c1c8");
        CHECK(mate >= 0);
        int max_visits = 0, others_max = 0;
        for (int i = 0; i < az_root_count(g); i++) {
            int v = az_root_visits(g, i);
            if (v > max_visits) max_visits = v;
            if (i != mate && v > others_max) others_max = v;
        }
        int mate_visits = az_root_visits(g, mate);

        /* The mated child's q is its OWN mover's (black's) view: −1. */
        CHECK(fabs(az_root_q(g, mate) - (-1.0)) < 1e-15);

        if (q_parent) {
            /* AlphaGo-paper sign: the mate wins the visit race outright —
             * every quiet child keeps at most its single rotation visit. */
            CHECK(mate_visits >= 40);
            CHECK(mate_visits == max_visits);
            CHECK(others_max == 1);
            /* After the rotation reached the mate, no further evals. */
            CHECK(ctx.requests <= 20);
        } else {
            /* Verbatim az_eval.py sign: the mate is visited exactly once
             * (while still "free"), then its q keeps it at the bottom
             * forever — 39 remaining sims shared by the quiet children. */
            CHECK(mate_visits == 1);
            CHECK(others_max >= 3);
        }
        az_game_free(g);
    }

    /* Terminals never evaluate (verbatim sign, so the rotation covers
     * every child exactly once): 21 sims = root + 20 children, but only
     * 20 requests — the mate child parked nothing. */
    {
        AzGame *g = az_game_new(MATE_FEN);
        CountCtx ctx = {0};
        az_think_sync(g, 21, 1.25, 0, uniform_eval, &ctx);
        CHECK(ctx.requests == 20);
        int visited = 0;
        for (int i = 0; i < az_root_count(g); i++)
            visited += az_root_visits(g, i) > 0;
        CHECK(visited == 20);
        az_game_free(g);
    }
}

/* ------------------------------------------------------------------ */
/* 5. per-move reset + game ops                                        */
/* ------------------------------------------------------------------ */

static void test_reset_and_game_ops(void) {
    AzGame *g = az_game_new(NULL);
    char fen[CHESS_FEN_MAX];

    /* Two consecutive searches must not share tree state. */
    CountCtx c1 = {0}, c2 = {0};
    az_think_sync(g, 12, 1.25, 0, uniform_eval, &c1);
    CHECK(c1.requests >= 1 && c1.requests <= 12);
    CHECK(az_play_uci(g, "e2e4") == 0);
    CHECK(az_ply(g) == 1);
    CHECK(az_side(g) == 1);
    az_think_sync(g, 12, 1.25, 0, uniform_eval, &c2);
    CHECK(c2.requests >= 1 && c2.requests <= 12);
    CHECK(az_root_count(g) == 20);

    /* Illegal / malformed moves are rejected, the game unchanged. */
    CHECK(az_play_uci(g, "e2e4") == -1);
    CHECK(az_play_uci(g, "zz") == -1);
    CHECK(az_ply(g) == 1);

    az_fen(g, fen);
    CHECK(strstr(fen, "e7e8") == NULL);      /* no UCI garbage in the FEN */
    CHECK(fen[0] == 'r' || fen[0] == 'n' || fen[0] == 'b'
          || fen[0] == 'q' || fen[0] == 'k' || fen[0] == 'p'
          || fen[0] == 'P' || fen[0] == '1' || fen[0] == '8');

    /* FEN round-trip into a fresh handle. A FEN carries no move history:
     * the new game starts at ply 0 (the same semantics as the Python
     * Board — Board(fen).ply == 0), while its position matches ours. */
    AzGame *h = az_game_new(fen);
    char fen2[CHESS_FEN_MAX];
    az_fen(h, fen2);
    CHECK(strcmp(fen, fen2) == 0);
    CHECK(az_ply(h) == 0);
    CHECK(az_side(h) == 1);
    az_game_free(h);

    /* Malformed FEN rejected. */
    CHECK(az_game_new("not a fen") == NULL);

    az_game_free(g);
}

/* Argmax-visits move choice (the driver's deterministic phase), first
 * maximum in enumeration order. */
static int argmax_visits(AzGame *g) {
    int best = 0, best_v = -1;
    for (int i = 0; i < az_root_count(g); i++) {
        int v = az_root_visits(g, i);
        if (v > best_v) { best_v = v; best = i; }
    }
    return best;
}

static void test_game_smoke(void) {
    /* Uniform-net "self-play" through the whole game API: every move the
     * argmax-visits child, 240 plies of arena resets, natural or
     * adjudicated termination without any assertion/crash. */
    AzGame *g = az_game_new(NULL);
    char uci[CHESS_UCI_MAX];
    for (int ply = 0; ply < 240; ply++) {
        if (az_termination(g) != TERM_PLAYING)
            break;
        CountCtx ctx = {0};
        az_think_sync(g, 8, 1.25, 1, uniform_eval, &ctx);
        az_root_uci(g, argmax_visits(g), uci);
        CHECK(az_play_uci(g, uci) == 0);
    }
    CHECK(az_ply(g) > 0);
    if (az_termination(g) == TERM_CHECKMATE) {
        /* Winner is the side NOT to move; the API mirrors Board.winner. */
        CHECK(az_winner(g) == 1 - az_side(g));
    } else {
        CHECK(az_winner(g) == -1);
    }
    az_game_free(g);
}

int main(int argc, char **argv) {
    chess_init();
    setbuf(stdout, NULL);
    int which = (argc > 1) ? atoi(argv[1]) : 0;

    if (!which || which == 1) { fprintf(stderr, "test: request\n");  test_request_shape(); }
    if (!which || which == 2) { fprintf(stderr, "test: priors\n");   test_priors_normalization(); }
    if (!which || which == 3) { fprintf(stderr, "test: mate\n");     test_mate_in_1_q_sign(); }
    if (!which || which == 4) { fprintf(stderr, "test: reset\n");    test_reset_and_game_ops(); }
    if (!which || which == 5) { fprintf(stderr, "test: smoke\n");    test_game_smoke(); }

    printf("test_az: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
