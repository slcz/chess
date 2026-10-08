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
#include "chess_match.h"
#include "chess_nn_eval.h"
#include "chess_rng.h"

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

static Move select_move(const RootKids *k, int sims, int ply, int sample_plys) {
    /* the shared policy; match_puct's raw-policy player AND its
     * no-visits-yet fallback are both just prior_side */
    uint32_t total_visits = 0;
    for (int i = 0; i < k->n; i++)
        total_visits += k->visits[i];
    bool prior_side = sims < 2 || total_visits == 0;
    return match_pick_move(k->moves, k->visits, k->prior, k->n,
                           ply, sample_plys, prior_side, 1.0, 0);
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
    rng_seed(argc > 5 ? (uint64_t)strtoull(argv[5], NULL, 10) : 1);
    int max_plys = argc > 6 ? atoi(argv[6]) : 250;
    const char *device = argc > 7 ? argv[7] : "mps";

    chess_init();
    PuctGame *g = puct_game_new(0);
    const char *wenv = getenv("PUCT_WALKERS");
    if (wenv && *wenv)
        puct_set_walkers(g, atoi(wenv));
    if (puct_attach_nn(ckpt, device, PUCT_MAX_WALKERS) != 0) {
        fprintf(stderr, "puct_attach_nn failed\n");
        return 1;
    }

    int ply = 0;
    int term = TERM_PLAYING;
    uint32_t parks = 0, terminal_hits = 0, pending_parks = 0;
    while ((term = state_termination((State *)puct_state(g))) == TERM_PLAYING) {
        if (ply >= max_plys) {
            term = -1;                         /* ply cap: adjudicated draw */
            break;
        }
        int sims = (puct_state(g)->side == WHITE) ? white_sims : black_sims;
        puct_think(g, sims);
        uint32_t p, t, d;
        puct_park_stats(g, &p, &t, &d);
        parks += p;
        terminal_hits += t;
        pending_parks += d;
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
    printf("STATS parks=%u terminal_hits=%u term_rate=%.4f "
           "pending_parks=%u pend_rate=%.4f rows=%u\n",
           parks, terminal_hits,
           parks ? (double)terminal_hits / parks : 0.0,
           pending_parks,
           parks ? (double)pending_parks / parks : 0.0,
           parks - terminal_hits - pending_parks);

    puct_game_free(g);
    return 0;
}
