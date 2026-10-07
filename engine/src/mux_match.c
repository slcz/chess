/*
 * mux_match.c — N games between two search configurations of the same
 * checkpoint, ALL run concurrently through the landing-zone mux (batched
 * evals across every game and both sides).
 *
 *   mux_match <ckpt> <a_sims> <b_sims> <games> [sample_plys=30]
 *             [max_plys=250] [device=cuda] [rows=0] [seed=1]
 *             [openings=50] [opening_plies=8]
 *
 * Deterministic evaluation (sample_plys=0, the AGZ evaluator protocol:
 * argmax visits, strongest play) needs opening variety: `openings`
 * distinct lines of `opening_plies` prior-sampled plies are generated
 * from the net itself at startup, and each line is played TWICE — slot
 * i uses line i/2 with side A taking White on even i — so every line
 * contributes one game per color. games should be 2*openings.
 *
 * Slot i: side A takes White when i is even, Black when odd (50/50
 * colors). Each round walks every game with the side to move's own sims
 * budget (puct_mux_step_each), then the driver plays one move per
 * finished think — visits-sampled for the first sample_plys plies (both
 * sides, opening diversity), argmax after. Finished games tally and
 * recycle in place until `games` have been played.
 *
 * a_sims <= 1 selects the raw-policy player (think(1) + argmax prior —
 * the eval-round baseline). Output: one GAME line per game, then
 *   SUMMARY a=<sims> b=<sims> games=N w=.. d=.. l=.. score_a=0.123
 * (w/d/l/score from side A's perspective).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "chess.h"
#include "chess_nn_eval.h"
#include "chess_puct.h"

static uint64_t rng_state;

static uint64_t rng_next(void) {
    rng_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static double rng_uniform(void) {
    return (double)(rng_next() >> 11) / 9007199254740992.0;
}

static int sample_index(const double *w, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++)
        total += w[i];
    if (total <= 0.0)
        return n - 1;
    double r = rng_uniform() * total;
    for (int i = 0; i < n; i++) {
        r -= w[i];
        if (r < 0.0)
            return i;
    }
    return n - 1;
}

/* Search side: visits-sampled early (the eval-round diversity protocol),
 * argmax after. Raw-policy side (sims <= 1): the priors ARE the
 * distribution — visits+1 would be near-uniform noise, so sample/argmax
 * by prior exactly like match_puct's raw player. */
static Move select_move(const PuctGame *g, int ply, int sample_plys,
                        bool policy_player) {
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    float prior[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, NULL, prior);
    if (n <= 0)
        return 0;
    if (policy_player) {
        if (ply < sample_plys) {
            double w[MAX_MOVES];
            for (int i = 0; i < n; i++)
                w[i] = prior[i];
            return moves[sample_index(w, n)];
        }
        int best = 0;
        for (int i = 1; i < n; i++)
            if (prior[i] > prior[best])
                best = i;
        return moves[best];
    }
    if (ply < sample_plys) {
        double w[MAX_MOVES];
        for (int i = 0; i < n; i++)
            w[i] = visits[i];   /* strict: the eval-round protocol */
        return moves[sample_index(w, n)];
    }
    int best = 0;
    for (int i = 1; i < n; i++)
        if (visits[i] > visits[best])
            best = i;
    return moves[best];
}

typedef struct {
    bool a_white;     /* slot i: side A is White when i is even */
    int  ply;
    char moves[4096]; /* uci move log for the game (debug/comparison) */
} Slot;

/* The book: `n_open` lines of at most `open_plies` uci moves. */
static char (*book)[8 * 16];
static int *book_len;
static int n_open, open_plies;

/* Generate the book from the net's own priors: each line samples
 * open_plies moves from think(1)'s policy — a natural mini-book. */
static void gen_book(const char *ckpt, const char *device, uint64_t seed) {
    book = calloc((size_t)n_open, sizeof *book);
    book_len = calloc((size_t)n_open, sizeof *book_len);
    rng_state = seed ? seed : 1;
    PuctGame *g = puct_game_new(NULL, 0);
    for (int k = 0; k < n_open; k++) {
        puct_game_reset(g);
        for (int p = 0; p < open_plies; p++) {
            puct_think(g, 1);
            uint16_t moves[MAX_MOVES];
            float prior[MAX_MOVES];
            int n = puct_root_children(g, moves, NULL, NULL, prior);
            if (n <= 0)
                break;
            double w[MAX_MOVES], total = 0.0;
            for (int i = 0; i < n; i++) {
                w[i] = prior[i];
                total += prior[i];
            }
            double r = rng_uniform() * total;
            int pick = n - 1;
            for (int i = 0; i < n; i++) {
                r -= w[i];
                if (r < 0.0) {
                    pick = i;
                    break;
                }
            }
            char u[8];
            move_to_uci(moves[pick], u);
            strcpy(book[k] + p * 8, u);
            book_len[k] = p + 1;
            if (getenv("NN_BOOK_LOG"))
                fprintf(stderr, "book[%02d] ply%d %s\n", k, p + 1, u);
            puct_play(g, moves[pick]);
        }
    }
    puct_game_free(g);
    (void)ckpt; (void)device;
}

/* Replay slot i's line onto its (fresh) game, logging the moves; the
 * caller clears slot->moves first. Returns plies played. */
static int play_opening(PuctGame *g, Slot *s, int i) {
    int line = i / 2;                  /* each line once per color */
    for (int p = 0; p < book_len[line]; p++) {
        MoveList ml;
        gen_legal(puct_state(g), &ml);
        Move mv = 0;
        for (int k = 0; k < ml.count; k++) {
            char u[8];
            move_to_uci(ml.moves[k], u);
            if (strcmp(u, book[line] + p * 8) == 0) {
                mv = ml.moves[k];
                break;
            }
        }
        if (!mv) {
            fprintf(stderr, "book line %d replay failed at ply %d\n",
                    line, p);
            return p;
        }
        int len = strlen(s->moves);
        snprintf(s->moves + len, sizeof s->moves - len, "%.4s ",
                 book[line] + p * 8);
        puct_play(g, mv);
    }
    return book_len[line];
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr,
                "usage: %s <ckpt> <a_sims> <b_sims> <games> "
                "[sample_plys=30] [max_plys=250] [device=cuda] [rows=0] "
                "[seed=1]\n",
                argv[0]);
        return 2;
    }
    const char *ckpt = argv[1];
    int a_sims = atoi(argv[2]) < 1 ? 1 : atoi(argv[2]);
    int b_sims = atoi(argv[3]) < 1 ? 1 : atoi(argv[3]);
    int games_total = atoi(argv[4]);
    int sample_plys = argc > 5 ? atoi(argv[5]) : 30;
    int max_plys = argc > 6 ? atoi(argv[6]) : 250;
    const char *device = argc > 7 ? argv[7] : "cuda";
    int rows = argc > 8 ? atoi(argv[8]) : 0;
    rng_state = argc > 9 ? strtoull(argv[9], NULL, 10) : 1;
    if (!rng_state)
        rng_state = 1;
    n_open = argc > 10 ? atoi(argv[10]) : games_total / 2;
    open_plies = argc > 11 ? atoi(argv[11]) : 8;
    int pool = argc > 12 ? atoi(argv[12]) : 0;
    if (pool <= 0 || pool > games_total)
        pool = games_total;
    if (n_open < 1)
        n_open = 1;
    if (games_total < 1)
        games_total = 1;
    if (games_total < 2 * n_open)
        games_total = 2 * n_open;   /* every line once per color */

    chess_init();
    if (rows <= 0)
        rows = 8 * games_total;    /* batch scale comes from the pool */
    PuctMux *m = puct_mux_new(pool, rows, 0);   /* pool < games: slots
                                                   recycle, replaying their
                                                   line (same color) */
    if (!m) {
        fprintf(stderr, "mux_new failed\n");
        return 1;
    }
    if (puct_attach_nn(ckpt, device, rows) != 0) {
        fprintf(stderr, "puct_attach_nn failed\n");
        return 1;
    }
    gen_book(ckpt, device, rng_state);
    /* all slots exist before round 1: replay each line before the first
     * walk, so every think runs from the book-end position */
    puct_mux_reserve(m, games_total);

    Slot *slots = calloc((size_t)games_total, sizeof *slots);
    int *sims = calloc((size_t)games_total, sizeof *sims);
    int opened = 0;
    for (int i = 0; i < puct_mux_games(m); i++) {
        slots[i].a_white = (i % 2 == 0);
        slots[i].moves[0] = 0;
        slots[i].ply = play_opening(puct_mux_game(m, i), &slots[i], i);
    }
    uint32_t wins = 0, draws = 0, losses = 0;
    int played = 0;
    double t0 = now_s();

    while (played < games_total) {
        /* refresh per-game sims: the side to move thinks with its budget.
         * Covers ALL slots — unspawned ones are fresh startpos games
         * (White to move), and step_each spawns with sims[live]. */
        for (int i = 0; i < games_total; i++) {
            slots[i].a_white = (i % 2 == 0);
            if (i < puct_mux_games(m)) {
                bool a_to_move =
                    (puct_state(puct_mux_game(m, i))->side == WHITE) ==
                    slots[i].a_white;
                sims[i] = a_to_move ? a_sims : b_sims;
            } else {
                sims[i] = slots[i].a_white ? a_sims : b_sims;
            }
        }
        puct_mux_step_each(m, sims);

        /* play one move per finished think; tally + recycle finished */
        for (int i = 0; i < puct_mux_games(m); i++) {
            PuctGame *g = puct_mux_game(m, i);
            if (!puct_think_done(g))
                continue;
            int term = state_termination((State *)puct_state(g));
            if (played >= games_total)
                break;         /* one pass may finish many games at once */
            if (term != TERM_PLAYING || slots[i].ply >= max_plys) {
                double sc;
                if (term == TERM_CHECKMATE) {
                    bool a_won = (puct_state(g)->side == BLACK) ==
                                 slots[i].a_white;
                    sc = a_won ? 1.0 : 0.0;
                } else {
                    sc = 0.5;
                }
                if (sc == 1.0) wins++;
                else if (sc == 0.0) losses++;
                else draws++;
                played++;
                const char *res = sc == 1.0 ? (slots[i].a_white ? "white_win"
                                                    : "black_win")
                                  : sc == 0.0 ? (slots[i].a_white ? "black_win"
                                                      : "white_win")
                                              : "draw";
                printf("GAME a=%d b=%d slot=%d result=%s plies=%d\n"
                       "  moves: %s\n",
                       a_sims, b_sims, i, res, slots[i].ply, slots[i].moves);
                puct_game_reset(g);
                slots[i].moves[0] = 0;
                slots[i].ply = play_opening(g, &slots[i], i);
                continue;
            }
            bool a_to_move =
                (puct_state(g)->side == WHITE) == slots[i].a_white;
            int mover_sims = a_to_move ? a_sims : b_sims;
            Move mv = select_move(g, slots[i].ply, sample_plys,
                                  mover_sims <= 1);
            char u[8];
            move_to_uci(mv, u);
            int len = strlen(slots[i].moves);
            snprintf(slots[i].moves + len, sizeof slots[i].moves - len,
                     "%s ", u);
            if (!mv || puct_play(g, mv) != 0) {
                char fen[96];
                fen_serialize(puct_state(g), fen, sizeof fen);
                fprintf(stderr,
                        "bad move at slot %d: mv=%d children=%d term=%d "
                        "fen=%s ply=%d\n",
                        i, (int)mv,
                        puct_root_children(g, NULL, NULL, NULL, NULL),
                        state_termination((State *)puct_state(g)), fen,
                        slots[i].ply);
                return 1;
            }
            slots[i].ply++;
        }
    }

    printf("SUMMARY a=%d b=%d games=%d w=%u d=%u l=%u score_a=%.3f "
           "elapsed=%.1fs\n",
           a_sims, b_sims, played, wins, draws, losses,
           (wins + 0.5 * draws) / played, now_s() - t0);
    puct_mux_free(m);
    free(slots);
    free(sims);
    free(book);
    free(book_len);
    return 0;
}
