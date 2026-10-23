#define _POSIX_C_SOURCE 200809L
/*
 * sf_gauntlet.c — N concurrent games vs Stockfish children, our engine's
 * turns advanced through the landing-zone mux (batched evals).
 *
 *   sf_gauntlet <ckpt> <sims> <elo_csv> [games=100] [pool=games]
 *               [movetime=200] [sample_plys=10] [max_plys=300]
 *               [sf=stockfish] [device=cuda] [rows=0] [seed=1]
 *
 * One Stockfish process per live game, configured with
 * limited TWO ways (SF_LIMIT env): "elo" (default) sets UCI_LimitStrength
 * + UCI_Elo=<level>; "nodes" plays at full strength under a node budget
 * per move, i.e. the CSV holds node counts. One level per run; the CSV is
 * played level by level. Slot i: our engine takes White when i is even,
 * Black when odd. Games waiting for Stockfish's reply are suspended
 * (puct_set_suspended) so the mux never searches the opponent's turn; SF
 * children attach once at startup (one per pool slot). Finished games
 * are recycled in place — pool slots play many games per level.
 *
 * Output: one GAME line per finished game, one SUMMARY line per level:
 *   GAME elo=1350 slot=3 result=white_win plies=87      (SF_LIMIT=nodes
 *   SUMMARY elo=1350 games=100 w=12 d=61 l=27 score=0.425  labels it nodes=)
 *
 * SF_LOG_MOVES=<file> appends one line per finished game with its full UCI
 * move list. SP_RECORD=<dir> instead writes the games as SPSHARD1 shards
 * (plus DONE.json), i.e. the self-play unit format, so scripts/sp_replay.py
 * can step through them on a board with π and the root value per ply.
 * (w/d/l/score from our engine's perspective).
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "chess.h"
#include "chess_match.h"
#include "chess_nn_eval.h"
#include "chess_sp_dump.h"
#include "chess_puct.h"
#include "chess_rng.h"

/* ---- our move selection (sample by visits early, argmax later) ---------- */

static Move select_move(const PuctGame *g, int ply, int sample_plys) {
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, NULL, NULL);
    /* boost 1 keeps unvisited moves reachable in the sampling phase */
    return match_pick_move(moves, visits, NULL, n, ply, sample_plys,
                           false, 1.0, 1);
}

/* ---- Stockfish children -------------------------------------------------- */

typedef struct {
    pid_t pid;
    FILE *in;             /* child's stdin  (we write)   */
    FILE *out;            /* child's stdout (we read)    */
} SfChild;

static SfChild sf_spawn(const char *path) {
    int to_sf[2], from_sf[2];
    if (pipe(to_sf) != 0 || pipe(from_sf) != 0) {
        fprintf(stderr, "pipe failed\n");
        exit(1);
    }
    pid_t pid = fork();
    if (pid < 0) {
        fprintf(stderr, "fork failed\n");
        exit(1);
    }
    if (pid == 0) {
        dup2(to_sf[0], 0);
        dup2(from_sf[1], 1);
        close(to_sf[0]); close(to_sf[1]);
        close(from_sf[0]); close(from_sf[1]);
        execlp(path, path, (char *)NULL);
        fprintf(stderr, "exec %s failed\n", path);
        _exit(1);
    }
    close(to_sf[0]);
    close(from_sf[1]);
    SfChild sf = {pid, fdopen(to_sf[1], "w"), fdopen(from_sf[0], "r")};
    if (!sf.in || !sf.out) {
        fprintf(stderr, "fdopen failed\n");
        exit(1);
    }
    return sf;
}

static void sf_send(SfChild *sf, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(sf->in, fmt, ap);
    va_end(ap);
    fputc('\n', sf->in);
    fflush(sf->in);
}

/* Read lines until one starts with `needle`; returns the line. */
static const char *sf_read_until(SfChild *sf, const char *needle) {
    static char line[8192];
    while (fgets(line, sizeof line, sf->out)) {
        if (strncmp(line, needle, strlen(needle)) == 0)
            return line;
    }
    fprintf(stderr, "stockfish closed the pipe (wanted '%s')\n", needle);
    exit(1);
}

static int g_sims;           /* our search budget, for the shard's n_sims */
static int g_dump_fd = -1;   /* SP_RECORD shard; -1 = not recording    */
static uint32_t g_game_idx;

static int g_limit_mode;   /* 0 = UCI_Elo (default), 1 = nodes per move */
static int g_level_val;    /* the current level: Elo or node budget  */

static void sf_setup(SfChild *sf, int elo) {
    sf_send(sf, "uci");
    sf_read_until(sf, "uciok");
    sf_send(sf, "setoption name Threads value 1");
    sf_send(sf, "setoption name Hash value 16");
    if (!g_limit_mode) {   /* nodes mode plays full strength */
        sf_send(sf, "setoption name UCI_LimitStrength value true");
        sf_send(sf, "setoption name UCI_Elo value %d", elo);
    }
    sf_send(sf, "isready");
    sf_read_until(sf, "readyok");
}

static void sf_newgame(SfChild *sf) {
    sf_send(sf, "ucinewgame");
    sf_send(sf, "isready");
    sf_read_until(sf, "readyok");
}

/* bestmove after the caller already sent position+go */
static char sf_move_buf[8];

static const char *sf_read_bestmove(SfChild *sf) {
    const char *line = sf_read_until(sf, "bestmove");
    const char *mv = line + strlen("bestmove");
    while (*mv == ' ')
        mv++;
    int i = 0;
    while (mv[i] && mv[i] != ' ' && mv[i] != '\n' && i < 7) {
        sf_move_buf[i] = mv[i];
        i++;
    }
    sf_move_buf[i] = 0;
    return sf_move_buf;
}

/* ---- match slots ---------------------------------------------------------- */

typedef struct {
    SfChild *sf;
    bool     we_white;     /* slot i: our engine is White when i is even */
    bool     sf_turn;      /* suspended: waiting for stockfish's reply */
    char     moves[16384]; /* uci move list for "position startpos moves" */
    int      ply;
    SpPlyRec *rec;    /* recorded plies of the current game (SP_RECORD) */
    int      n_rec;
} Slot;

static int g_movetime;

static void sf_ask_move(Slot *s) {
    sf_send(s->sf, "position startpos moves%s", s->moves);
    if (g_limit_mode)
        sf_send(s->sf, "go nodes %d", g_level_val);
    else
        sf_send(s->sf, "go movetime %d", g_movetime);
}

static void moves_append(Slot *s, Move mv) {
    char u[8];
    move_to_uci(mv, u);
    int len = strlen(s->moves);
    snprintf(s->moves + len, sizeof s->moves - len, " %s", u);
}

/* Begin (or re-begin) on a fresh startpos game. reset_tree also rewinds
 * the PuctGame — use it between levels; mid-level attaches must NOT reset
 * (the mux already finished the slot's first think). */
static void slot_start(Slot *s, PuctGame *g, int i, bool reset_tree) {
    if (reset_tree)
        puct_game_reset(g);
    s->moves[0] = 0;
    s->ply = 0;
    s->we_white = (i % 2 == 0);
    s->sf_turn = !s->we_white;
    puct_set_suspended(g, s->sf_turn);
    sf_newgame(s->sf);
    if (s->sf_turn)
        sf_ask_move(s);        /* stockfish moves first; read later */
}

/* Our engine's score for the game at g (1 win / 0.5 draw / 0 loss).
 * In a CHECKMATE position the side to move is mated; a ply cap and every
 * non-mate termination is a draw. */
static double our_score(const PuctGame *g, bool we_white) {
    if (state_termination((State *)puct_state(g)) != TERM_CHECKMATE)
        return 0.5;
    bool winner_white = puct_state(g)->side == BLACK;
    return winner_white == we_white ? 1.0 : 0.0;
}

/* "elo=1350" or "nodes=10000", whichever mode is active */
static const char *lim_label(char *buf, size_t n, int v) {
    snprintf(buf, n, "%s=%d", g_limit_mode ? "nodes" : "elo", v);
    return buf;
}

static const char STARTPOS_FEN[] =
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

/* One ply into the shard record. Our own plies carry the search's π —
 * dense over ALL root children in child order (== engine movegen order,
 * the v3 contract), 0.0f for unvisited — plus the mover-view root value
 * and the actual budget; Stockfish's plies carry 1.0f on the played move
 * (dense over the legal moves, zeros elsewhere) with sims=0 so a reader
 * can tell them apart. */
static void rec_ply(Slot *s, PuctGame *g, Move mv, int our_turn) {
    if (g_dump_fd < 0 || !s->rec)
        return;
    SpPlyRec *p = &s->rec[s->n_rec++];
    memset(p, 0, sizeof *p);
    p->mover = (uint8_t)(s->ply & 1);          /* startpos: White first */
    move_to_uci(mv, p->played_uci);
    if (our_turn) {
        uint16_t mv_[MAX_MOVES];
        uint32_t vis[MAX_MOVES];
        double q[MAX_MOVES];
        float pri[MAX_MOVES];
        int n = puct_root_children(g, mv_, vis, q, pri);
        double tot = 0.0;
        for (int i = 0; i < n; i++)
            tot += vis[i];
        p->n_pi = (uint16_t)n;
        for (int i = 0; i < n; i++)
            p->pi_prob[i] = vis[i] ? (float)((double)vis[i] / tot) : 0.0f;
        uint32_t rv;
        double wsum;
        puct_root_stats(g, &rv, &wsum);
        p->outlook = rv ? -(wsum / rv) : 0.0;   /* root W is parent-view */
        p->sims = (uint16_t)g_sims;
    } else {
        MoveList ml;
        gen_legal(puct_state(g), &ml);          /* the on-disk π order */
        p->n_pi = (uint16_t)ml.count;
        for (int k = 0; k < ml.count; k++)
            if (ml.moves[k] == mv) {
                p->pi_prob[k] = 1.0f;
                break;
            }
        p->sims = 0;
    }
}

/* Flush the finished game into the shard, then start the next one. */
static void rec_game(Slot *s, double our_score, bool adjudicated) {
    if (g_dump_fd < 0 || !s->rec || s->n_rec == 0)
        return;
    SpGameRec gr;
    memset(&gr, 0, sizeof gr);
    gr.game_index = g_game_idx++;
    gr.n_plies = (uint32_t)s->n_rec;
    gr.end_kind = adjudicated ? SP_END_ADJUDICATED : SP_END_NATURAL;
    gr.resigner = -1;
    gr.n_sims = (uint32_t)g_sims;
    gr.z_white = s->we_white ? our_score : 1.0 - our_score;
    snprintf(gr.start_fen, SP_FEN_MAX, "%s", STARTPOS_FEN);
    if (sp_dump_game_head(g_dump_fd, &gr) == 0)
        for (int i = 0; i < s->n_rec; i++)
            sp_dump_ply(g_dump_fd, &s->rec[i]);
    s->n_rec = 0;
}

static FILE *g_moves_log;

static void tally_game(int i, const Slot *s, int elo, double sc) {
    const char *res;
    if (sc == 1.0)
        res = s->we_white ? "white_win" : "black_win";
    else if (sc == 0.0)
        res = s->we_white ? "black_win" : "white_win";
    else
        res = "draw";
    char lv[32];
    printf("GAME %s slot=%d result=%s plies=%d\n",
           lim_label(lv, sizeof lv, elo), i, res, s->ply);
    if (g_moves_log) {
        fprintf(g_moves_log, "LOG slot=%d result=%s plies=%d moves=%s\n",
                i, res, s->ply, s->moves[0] ? s->moves + 1 : "");
        fflush(g_moves_log);
    }
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s <ckpt> <sims> <elo_or_nodes_csv> [games=100] [pool=games]\n"
                "           [movetime=200] [sample_plys=10] [max_plys=300]\n"
                "           (SF_LIMIT=elo|nodes selects how the CSV is read)\n"
                "           [sf=stockfish] [device=cuda] [rows=0] [seed=1]\n",
                argv[0]);
        return 2;
    }
    const char *ckpt = argv[1];
    int sims = atoi(argv[2]);
    const char *elo_csv = argv[3];
    int games_per_level = argc > 4 ? atoi(argv[4]) : 100;
    int pool = argc > 5 ? atoi(argv[5]) : games_per_level;
    g_movetime = argc > 6 ? atoi(argv[6]) : 200;
    const char *log_path = getenv("SF_LOG_MOVES");
    if (log_path && *log_path) {
        g_moves_log = fopen(log_path, "a");
        if (!g_moves_log) {
            fprintf(stderr, "cannot open SF_LOG_MOVES=%s\n", log_path);
            return 2;
        }
    }
    const char *lim = getenv("SF_LIMIT");
    if (lim && strcmp(lim, "nodes") == 0)
        g_limit_mode = 1;
    else if (lim && strcmp(lim, "elo") != 0) {
        fprintf(stderr, "SF_LIMIT must be \"elo\" or \"nodes\"\n");
        return 2;
    }
    int sample_plys = argc > 7 ? atoi(argv[7]) : 10;
    int max_plys = argc > 8 ? atoi(argv[8]) : 300;
    const char *sf_path = argc > 9 ? argv[9] : "stockfish";
    const char *device = argc > 10 ? argv[10] : "cuda";
    int rows = argc > 11 ? atoi(argv[11]) : 0;
    uint64_t seed = argc > 12 ? strtoull(argv[12], NULL, 10) : 1;
    rng_seed(seed);
    if (pool < 1)
        pool = 1;
    if (pool > games_per_level)
        pool = games_per_level;

    int elos[16], n_elos = 0;
    char elos_buf[256];
    snprintf(elos_buf, sizeof elos_buf, "%s", elo_csv);
    for (char *tok = strtok(elos_buf, ","); tok && n_elos < 16;
         tok = strtok(NULL, ","))
        elos[n_elos++] = atoi(tok);
    if (n_elos == 0) {
        fprintf(stderr, "bad elo_csv\n");
        return 2;
    }

    chess_init();
    if (rows <= 0)
        rows = 8 * pool;       /* batch scale comes from the pool */
    PuctGame *games[pool];                  /* handles, filled once at new */
    if (puct_mux_new(pool, 0, games) != 0) {
        fprintf(stderr, "mux_new failed\n");
        return 1;
    }
    if (puct_attach_nn(ckpt, device, rows) != 0) {
        fprintf(stderr, "puct_attach_nn failed\n");
        return 1;
    }
    g_sims = sims;
    const char *rec_dir = getenv("SP_RECORD");
    if (rec_dir && *rec_dir) {
        g_dump_fd = sp_dump_open_shard(rec_dir, 0, (uint64_t)seed);
        if (g_dump_fd < 0) {
            fprintf(stderr, "cannot open SP_RECORD=%s\n", rec_dir);
            return 2;
        }
    }
    Slot *slots = calloc((size_t)pool, sizeof *slots);
    int n_slots = 0;           /* == pool after the startup attach */
    double t_all0 = now_s();

    for (int li = 0; li < n_elos; li++) {
        g_level_val = elos[li];   /* Elo or node budget per SF_LIMIT */
        for (int i = 0; i < n_slots; i++) {      /* reconfigure + restart */
            sf_send(slots[i].sf, "setoption name UCI_Elo value %d", elos[li]);
            slot_start(&slots[i], games[i], i, true);
        }
        uint32_t wins = 0, draws = 0, losses = 0;
        int played = 0;
        double t0 = now_s();
        while (played < games_per_level) {
            /* attach each slot's stockfish child once, at startup */
            while (n_slots < pool) {
                int i = n_slots++;
                if (g_dump_fd >= 0)
                    slots[i].rec = calloc((size_t)max_plys + 8,
                                          sizeof(SpPlyRec));
                slots[i].sf = malloc(sizeof *slots[i].sf);
                *slots[i].sf = sf_spawn(sf_path);
                sf_setup(slots[i].sf, elos[li]);
                slot_start(&slots[i], games[i], i, false);
            }
            /* advance every our-turn game by one full think */
            puct_mux_step(sims);
            /* play our moves, hand the turn to stockfish */
            for (int i = 0; i < n_slots; i++) {
                Slot *s = &slots[i];
                if (s->sf_turn)
                    continue;
                PuctGame *g = games[i];
                if (!puct_think_done(g))
                    continue;      /* defensive: post-step all are done */
                Move mv = select_move(g, s->ply, sample_plys);
                if (mv)
                    rec_ply(s, g, mv, 1);   /* π before the root moves */
                if (!mv || puct_play(g, mv) != 0) {
                    fprintf(stderr, "bad our move at slot %d\n", i);
                    return 1;
                }
                moves_append(s, mv);
                s->ply++;
                if (state_termination((State *)puct_state(g)) != TERM_PLAYING ||
                    s->ply >= max_plys) {
                    double sc = our_score(g, s->we_white);
                    if (sc == 1.0) wins++;
                    else if (sc == 0.0) losses++;
                    else draws++;
                    played++;
                    tally_game(i, s, elos[li], sc);
                    rec_game(s, sc, s->ply >= max_plys);
                    slot_start(s, g, i, true);
                    continue;
                }
                s->sf_turn = true;
                puct_set_suspended(g, true);
                sf_ask_move(s);
            }
            /* collect stockfish's replies */
            for (int i = 0; i < n_slots; i++) {
                Slot *s = &slots[i];
                if (!s->sf_turn)
                    continue;
                PuctGame *g = games[i];
                const char *bm = sf_read_bestmove(s->sf);
                Move mv = uci_to_move(puct_state(g), bm);
                if (!mv) {
                    char fen[96];
                    fen_serialize(puct_state(g), fen, sizeof fen);
                    fprintf(stderr,
                            "stockfish illegal move '%s' at slot %d\n"
                            "  fen=%s\n  moves=[%s ] ply=%d sf_turn=%d\n",
                            bm, i, fen, s->moves, s->ply, s->sf_turn);
                    return 1;
                }
                puct_play(g, mv);
                rec_ply(s, g, mv, 0);
                moves_append(s, mv);
                s->ply++;
                if (state_termination((State *)puct_state(g)) != TERM_PLAYING ||
                    s->ply >= max_plys) {
                    double sc = our_score(g, s->we_white);
                    if (sc == 1.0) wins++;
                    else if (sc == 0.0) losses++;
                    else draws++;
                    played++;
                    tally_game(i, s, elos[li], sc);
                    rec_game(s, sc, s->ply >= max_plys);
                    slot_start(s, g, i, true);
                    continue;
                }
                s->sf_turn = false;
                puct_set_suspended(g, false);
            }
        }
        char lv[32];
        printf("SUMMARY %s games=%d w=%u d=%u l=%u score=%.3f "
               "elapsed=%.1fs\n", lim_label(lv, sizeof lv, elos[li]),
               played, wins, draws, losses,
               (wins + 0.5 * draws) / played, now_s() - t0);
        fflush(stdout);
    }
    if (g_dump_fd >= 0) {
        sp_dump_write_done(rec_dir, 1, (uint64_t)seed, (uint32_t)g_game_idx);
        close(g_dump_fd);
        printf("RECORD %s: %u games\n", rec_dir, g_game_idx);
    }
    for (int i = 0; i < n_slots; i++) {
        free(slots[i].rec);
        sf_send(slots[i].sf, "quit");
        fclose(slots[i].sf->in);
        waitpid(slots[i].sf->pid, NULL, 0);
    }
    printf("TOTAL elapsed=%.1fs\n", now_s() - t_all0);
    return 0;
}
