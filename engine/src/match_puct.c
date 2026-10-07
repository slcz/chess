/*
 * match_puct.c — one game between two search configurations of the same
 * checkpoint, played through a single PuctGame (tree persists per side).
 *
 *   match_puct <ckpt> <white_sims> <black_sims> [sample_plys] [seed]
 *              [max_plys] [device]
 *
 * sims <= 1 selects the raw-policy player: think(1) evaluates the root
 * once and the move comes from the policy softmax alone (visit counts are
 * ignored — with one sim at most one child ever has a visit, and that
 * visit reflects where the walker descended, not the policy ranking).
 *
 * Randomness (self-play variety): for the first sample_plys each side
 * samples from its own move distribution — the search player proportional
 * to visit counts, the policy player proportional to the priors. After
 * that, argmax (visits for the search player, priors for the policy
 * player). Both sides get the same treatment so matchups stay symmetric.
 *
 * Output: one line per ply pair, then a final RESULT line:
 *   RESULT <white|black>_win|draw <reason> plies=<n>
 * A ply cap counts as a draw (reason ply_cap).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "chess_nn_eval.h"

static uint64_t rng_state;

static uint64_t rng_next(void) {          /* splitmix64 */
    rng_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double rng_uniform(void) {
    return (double)(rng_next() >> 11) / 9007199254740992.0;
}

typedef struct {
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    float    prior[MAX_MOVES];
    int      n;
} RootKids;

static RootKids root_kids(const PuctGame *g) {
    RootKids k;
    k.n = puct_root_children(g, k.moves, k.visits, NULL, k.prior);
    return k;
}

/* Sample proportional to w[i]; weights must not be all-zero. */
static int sample_index(const double *w, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++)
        total += w[i];
    double r = rng_uniform() * total;
    for (int i = 0; i < n; i++) {
        r -= w[i];
        if (r < 0.0)
            return i;
    }
    return n - 1;
}

static Move select_move(const RootKids *k, int sims, int ply, int sample_plys) {
    int search = sims >= 2;

    uint32_t total_visits = 0;
    for (int i = 0; i < k->n; i++)
        total_visits += k->visits[i];
    int by_visits = search && total_visits > 0;

    if (ply < sample_plys) {
        if (by_visits) {
            double w[MAX_MOVES];
            for (int i = 0; i < k->n; i++)
                w[i] = (double)k->visits[i];
            return k->moves[sample_index(w, k->n)];
        }
        double w[MAX_MOVES];
        for (int i = 0; i < k->n; i++)
            w[i] = k->prior[i];
        return k->moves[sample_index(w, k->n)];
    }

    int best = 0;
    for (int i = 1; i < k->n; i++) {
        double vi = by_visits ? (double)k->visits[i] : (double)k->prior[i];
        double vb = by_visits ? (double)k->visits[best] : (double)k->prior[best];
        if (vi > vb)
            best = i;
    }
    return k->moves[best];
}

static const char *term_reason(int t) {
    switch (t) {
    case TERM_CHECKMATE:              return "checkmate";
    case TERM_STALEMATE:              return "stalemate";
    case TERM_INSUFFICIENT_MATERIAL:  return "insufficient_material";
    case TERM_SEVENTYFIVE_MOVES:      return "seventyfive_moves";
    case TERM_FIVEFOLD_REPETITION:    return "fivefold_repetition";
    case TERM_FIFTY_MOVES:            return "fifty_moves";
    case TERM_THREEFOLD_REPETITION:   return "threefold_repetition";
    default:                          return "playing";
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <ckpt> <white_sims> <black_sims> "
                "[sample_plys=30] [seed=1] [max_plys=250] [device=mps]\n",
                argv[0]);
        return 2;
    }
    const char *ckpt = argv[1];
    int white_sims = atoi(argv[2]);
    int black_sims = atoi(argv[3]);
    int sample_plys = argc > 4 ? atoi(argv[4]) : 30;
    rng_state = argc > 5 ? (uint64_t)strtoull(argv[5], NULL, 10) : 1;
    int max_plys = argc > 6 ? atoi(argv[6]) : 250;
    const char *device = argc > 7 ? argv[7] : "mps";
    if (!rng_state)
        rng_state = 1;

    chess_init();
    PuctGame *g = puct_game_new(NULL, 0);
    const char *wenv = getenv("PUCT_WALKERS");
    if (wenv && *wenv)
        puct_set_walkers(g, atoi(wenv));
    if (puct_attach_nn(ckpt, device, PUCT_MAX_WALKERS) != 0) {
        fprintf(stderr, "puct_attach_nn failed\n");
        return 1;
    }

    int ply = 0;
    int term = TERM_PLAYING;
    uint32_t parks = 0, terminal_hits = 0;
    while ((term = state_termination((State *)puct_state(g))) == TERM_PLAYING) {
        if (ply >= max_plys) {
            term = -1;                         /* ply cap: adjudicated draw */
            break;
        }
        int sims = (puct_state(g)->side == WHITE) ? white_sims : black_sims;
        puct_think(g, sims);
        if (getenv("PUCT_DBG")) {
            uint64_t steps = 0, pend = 0;
            puct_pending_stats(g, &steps, &pend);
            fprintf(stderr, "DBG ply=%d sims=%d walk_steps=%llu "
                    "pending_steps=%llu (%.1f%%)\n",
                    ply, sims, (unsigned long long)steps,
                    (unsigned long long)pend,
                    steps ? 100.0 * pend / steps : 0.0);
        }
        uint32_t p, t;
        puct_park_stats(g, &p, &t);
        parks += p;
        terminal_hits += t;
        RootKids k = root_kids(g);
        if (k.n == 0) {                        /* no legal reply: terminal  */
            term = state_termination((State *)puct_state(g));
            break;
        }
        Move m = select_move(&k, sims, ply, sample_plys);
        char u[8];
        move_to_uci(m, u);
        printf("%s%s", ply ? " " : "", u);
        if (puct_play(g, m) != 0) {
            fprintf(stderr, "\npuct_play failed at ply %d (%s)\n", ply, u);
            return 1;
        }
        ply++;
    }
    printf("\n");

    const char *result;
    if (term == -1)
        result = "draw";
    else if (term == TERM_CHECKMATE)
        result = (puct_state(g)->side == BLACK) ? "white_win" : "black_win";
    else
        result = "draw";
    printf("RESULT %s %s plies=%d\n", result,
           term == -1 ? "ply_cap" : term_reason(term), ply);
    printf("STATS parks=%u terminal_hits=%u term_rate=%.4f\n",
           parks, terminal_hits,
           parks ? (double)terminal_hits / parks : 0.0);

    puct_game_free(g);
    return 0;
}
