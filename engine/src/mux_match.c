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
 * from the net itself at startup, and each line is played TWICE —
 * game j uses line j with side A taking White on even j — so every
 * line contributes one game per color. games should be 2*openings.
 *
 * Each game is played by TWO ENGINE SESSIONS, each with its own tree:
 * session 2j is side A's engine for game j, session 2j+1 is side B's.
 * Only the session of the side to move ever searches (the other is
 * suspended); the driver relays every played move into the partner's
 * tree. This mirrors cutechess's two-engine-process model: a side never
 * inherits the opponent's search — with asymmetric budgets, subtree
 * sharing would let the smaller search leverage the larger one's work.
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
 *   PARKS parks=.. terminal=.. pending=.. pend_rate=.. eval_rows=..
 * (w/d/l/score from side A's perspective; the PARKS line is the
 * virtual-visit duplication telemetry: pend_rate = fraction of leaf
 * arrivals that co-slept on an in-flight row instead of staging a new
 * one, eval_rows = actual net evals staged).
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
    bool     is_a;        /* session 2j is A's engine, 2j+1 is B's */
    bool     a_white;     /* game j = sessions (2j, 2j+1): A is White
                             on even j */
    bool     mover;       /* this session's side is to move */
    int      ply;
    char     moves[4096]; /* uci move log for the game */
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

/* Replay game i/2's line onto session slot s (both sessions of a pair
 * get it), logging the moves. Returns plies played. */
static int play_opening(PuctGame *g, Slot *s, int game) {
    int line = game;
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
    for (int i = 0; i < games_total; i++) {
        slots[i].is_a = (i % 2 == 0);
        slots[i].a_white = ((i / 2) % 2 == 0);
        slots[i].moves[0] = 0;
    }
    /* all sessions exist before round 1: replay each line into BOTH
     * sessions of the pair before the first walk */
    puct_mux_reserve(m, games_total);
    for (int i = 0; i < puct_mux_games(m); i++) {
        slots[i].ply = play_opening(puct_mux_game(m, i), &slots[i], i / 2);
        puct_set_suspended(puct_mux_game(m, i), true);
    }
    uint32_t wins = 0, draws = 0, losses = 0;
    uint64_t parks = 0, terminal_hits = 0, pending_parks = 0;
    int played = 0;
    double t0 = now_s();

    while (played < games_total) {
        /* per session: the side to move thinks with its engine's budget;
         * the partner session is suspended (it shares the game position
         * but never searches on the opponent's turn) */
        for (int i = 0; i < puct_mux_games(m); i++) {
            bool a_to_move =
                (puct_state(puct_mux_game(m, i))->side == WHITE) ==
                slots[i].a_white;
            bool mover = a_to_move == slots[i].is_a;
            slots[i].mover = mover;
            puct_set_suspended(puct_mux_game(m, i), !mover);
            int budget = slots[i].is_a ? a_sims : b_sims;
            sims[i] = mover ? budget : 0;
        }
        puct_mux_step_each(m, sims);

        /* park telemetry: parks is per-think (reset at each fresh think),
         * so harvest it right after the round's thinks complete */
        for (int i = 0; i < puct_mux_games(m); i++) {
            if (!slots[i].mover)
                continue;
            uint32_t p, t, d;
            puct_park_stats(puct_mux_game(m, i), &p, &t, &d);
            parks += p;
            terminal_hits += t;
            pending_parks += d;
        }

        /* relay + play: each pair's mover completes its think in its OWN
         * tree; the move is played into the mover's tree and relayed into
         * the partner's (which keeps only the partner's own analysis
         * under it). The partner session then becomes the mover. */
        for (int i = 0; i < puct_mux_games(m); i++) {
            if (!slots[i].mover)
                continue;              /* partner session: suspended */
            PuctGame *g = puct_mux_game(m, i);
            int partner = i ^ 1;
            if (!puct_think_done(g))
                continue;              /* defensive */
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
                printf("GAME a=%d b=%d game=%d result=%s plies=%d\n"
                       "  moves: %s\n",
                       a_sims, b_sims, i / 2, res, slots[i].ply,
                       slots[i].moves);
                puct_game_reset(g);
                puct_game_reset(puct_mux_game(m, partner));
                slots[i].moves[0] = 0;
                slots[partner].moves[0] = 0;
                slots[i].ply = play_opening(g, &slots[i], i / 2);
                slots[partner].ply = play_opening(
                    puct_mux_game(m, partner), &slots[partner], i / 2);
                continue;
            }
            int mover_sims = slots[i].is_a ? a_sims : b_sims;
            Move mv = select_move(g, slots[i].ply, sample_plys,
                                  mover_sims <= 1);
            char u[8];
            move_to_uci(mv, u);
            int len = strlen(slots[i].moves);
            snprintf(slots[i].moves + len, sizeof slots[i].moves - len,
                     "%s ", u);
            if (!mv || puct_play(g, mv) != 0 ||
                puct_play(puct_mux_game(m, partner), mv) != 0) {
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
            slots[partner].ply++;
        }
    }

    double dt = now_s() - t0;
    printf("SUMMARY a=%d b=%d games=%d w=%u d=%u l=%u score_a=%.3f "
           "rows=%llu rate=%.0f rows/s elapsed=%.1fs\n"
           "PARKS parks=%llu terminal=%llu pending=%llu pend_rate=%.4f "
           "eval_rows=%llu\n",
           a_sims, b_sims, played, wins, draws, losses,
           played ? (wins + 0.5 * draws) / played : 0.0,
           (unsigned long long)puct_mux_rows(m),
           puct_mux_rows(m) / dt, dt,
           (unsigned long long)parks,
           (unsigned long long)terminal_hits,
           (unsigned long long)pending_parks,
           parks ? (double)pending_parks / parks : 0.0,
           (unsigned long long)(parks - terminal_hits - pending_parks));
    puct_mux_free(m);
    free(slots);
    free(sims);
    free(book);
    free(book_len);
    return 0;
}
