/*
 * uci_puct.c — minimal UCI adapter around the PUCT search + real net.
 *
 *   PUCT_CKPT=runs/<net>.pt PUCT_DEVICE=mps \
 *   PUCT_SIMS=256 PUCT_SAMPLE_PLYS=10 build/uci_puct
 *
 * Protocol: the sequential subset python-chess speaks — uci, isready,
 * setoption (ignored), ucinewgame, position (startpos|fen) [moves ...],
 * go [nodes N], quit. "go nodes N" runs N PUCT simulations; without an
 * explicit count, PUCT_SIMS is used. "stop" is accepted and ignored (the
 * search is blocking, there is nothing to stop mid-way).
 *
 * position lines are incremental: when the line EXTENDS the position the
 * engine already holds (same start, its recorded moves a prefix of the
 * incoming list — the natural "startpos + full move list" dialog), only
 * the new suffix is puct_play'ed into the EXISTING tree, so each think
 * inherits the previous think's subtree under the new root (what the
 * mux drivers always had). Bare-FEN lines and rewinds still rebuild the
 * tree from scratch, exactly the old behavior.
 *
 * Move choice: argmax visits, except for the first PUCT_SAMPLE_PLYS plies
 * where the move is sampled proportional to visits (game variety without
 * an opening book; 0 = fully deterministic).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "chess_nn_eval.h"
#include "chess_rng.h"

/* The position dialog: the start descriptor ("" = startpos, else the FEN
 * as re-joined from the position line) plus every move played into the
 * current tree, in UCI text. A position line that shares the start and
 * takes the recorded moves as a prefix only plays its suffix — tree kept.
 * Anything else resets the dialog alongside the full rebuild. */
static char dlg_start[96];
static char dlg_uci[MAX_PLY][8];
static int  dlg_n;

static void dlg_reset(const char *start) {
    snprintf(dlg_start, sizeof dlg_start, "%s", start ? start : "");
    dlg_n = 0;
}

static Move choose_move(const PuctGame *g, int ply, int sample_plys) {
    uint16_t moves[MAX_MOVES];
    uint32_t visits[MAX_MOVES];
    int n = puct_root_children(g, moves, visits, NULL, NULL);

    uint32_t total = 0;
    for (int i = 0; i < n; i++)
        total += visits[i];
    if (ply < sample_plys && total > 0) {
        double r = rng_uniform() * (double)total;
        for (int i = 0; i < n; i++) {
            r -= (double)visits[i];
            if (r < 0.0)
                return moves[i];
        }
        return moves[n - 1];
    }
    int best = 0;
    for (int i = 1; i < n; i++)
        if (visits[i] > visits[best])
            best = i;
    return moves[best];
}

/* UCI string → move, resolved against the current legal move list (from/to
 * uniquely identify a non-promotion move; promotion disambiguated by the
 * 5th char). 0 if not legal. */
static void reply(const char *fmt, const char *arg) {
    printf(fmt, arg);
    fflush(stdout);
}

int main(void) {
    const char *ckpt = getenv("PUCT_CKPT");
    const char *device = getenv("PUCT_DEVICE");
    const char *sims_env = getenv("PUCT_SIMS");
    const char *sample_env = getenv("PUCT_SAMPLE_PLYS");
    if (!device)
        device = "mps";
    int default_sims = sims_env ? atoi(sims_env) : 256;
    int sample_plys = sample_env ? atoi(sample_env) : 10;
    if (!ckpt || !*ckpt) {
        fprintf(stderr, "PUCT_CKPT not set\n");
        return 1;
    }

    chess_init();
    rng_seed(0x12345678ull);   /* fixed stream; variety comes from sample_plys */
    PuctGame *g = puct_game_new(0);
    if (puct_attach_nn(ckpt, device, 8) != 0) {   /* default walkers */
        fprintf(stderr, "puct_attach_nn failed\n");
        return 1;
    }

    char line[8192];
    while (fgets(line, sizeof line, stdin)) {
        char *cmd = strtok(line, " \t\n");
        if (!cmd)
            continue;
        if (!strcmp(cmd, "uci")) {
            reply("id name puct\n", NULL);
            reply("id author chess\n", NULL);
            reply("uciok\n", NULL);
        } else if (!strcmp(cmd, "isready")) {
            reply("readyok\n", NULL);
        } else if (!strcmp(cmd, "ucinewgame")) {
            puct_game_free(g);
            g = puct_game_new(0);         /* evaluator is process-global */
            dlg_reset("");                /* fresh dialog at startpos    */
        } else if (!strcmp(cmd, "position")) {
            char *tok = strtok(NULL, " \t\n");
            char fen[96] = "";
            int is_startpos = 0;
            const char *mv[MAX_PLY];
            int k = 0;
            if (tok && !strcmp(tok, "startpos")) {
                is_startpos = 1;
                tok = strtok(NULL, " \t\n");    /* "moves" or NULL        */
                if (tok && strcmp(tok, "moves"))
                    tok = NULL;
            } else if (tok && !strcmp(tok, "fen")) {
                char *p = fen;
                while ((tok = strtok(NULL, " \t\n")) &&
                       strcmp(tok, "moves") &&
                       p + strlen(tok) + 1 < fen + sizeof fen) {
                    p += sprintf(p, "%s ", tok);
                }
                if (p > fen)
                    p[-1] = '\0';               /* drop the trailing space */
            } else {
                fprintf(stderr, "position: expected startpos|fen\n");
                return 1;
            }
            if (tok && !strcmp(tok, "moves"))
                for (char *t = strtok(NULL, " \t\n"); t && k < MAX_PLY;
                     t = strtok(NULL, " \t\n"))
                    mv[k++] = t;

            const char *start = is_startpos ? "" : fen;
            int prefix = g && k >= dlg_n && !strcmp(start, dlg_start);
            for (int i = 0; prefix && i < dlg_n; i++)
                if (strcmp(dlg_uci[i], mv[i]))
                    prefix = 0;

            if (!prefix) {                      /* full rebuild            */
                puct_game_free(g);
                g = puct_game_new(0);
                if (!g || (!is_startpos && puct_game_set_fen(g, fen) != 0)) {
                    fprintf(stderr, "position: bad fen\n");
                    return 1;
                }
                dlg_reset(start);
            }
            int suffix = prefix ? k - dlg_n : k;
            for (int i = dlg_n; i < k; i++) {   /* suffix only             */
                Move m = uci_to_move(puct_state(g), mv[i]);
                if (!m || puct_play(g, m) != 0) {
                    fprintf(stderr, "bad move %s\n", mv[i]);
                    return 1;
                }
                snprintf(dlg_uci[dlg_n], sizeof dlg_uci[0], "%s", mv[i]);
                dlg_n++;
            }
            fprintf(stderr, "puct: tree %s (+%d moves, dialog=%d)\n",
                    prefix ? "kept" : "rebuilt", suffix, dlg_n);
        } else if (!strcmp(cmd, "go")) {
            int sims = default_sims;
            for (char *t = strtok(NULL, " \t\n"); t;
                 t = strtok(NULL, " \t\n"))
                if (!strcmp(t, "nodes")) {
                    char *v = strtok(NULL, " \t\n");
                    if (v)
                        sims = atoi(v);
                }
            puct_think(g, sims);
            Move m = choose_move(g, (int)puct_state(g)->ply, sample_plys);
            char u[8] = "0000";
            if (m)
                move_to_uci(m, u);
            reply("bestmove %s\n", u);
            /* play our own move into the tree (subtree kept) and extend
             * the dialog, so the next position line arrives as a suffix */
            if (m) {
                if (puct_play(g, m) != 0)
                    return 1;
                if (dlg_n < MAX_PLY) {
                    snprintf(dlg_uci[dlg_n], sizeof dlg_uci[0], "%s", u);
                    dlg_n++;
                }
            }
        } else if (!strcmp(cmd, "stop")) {
            /* blocking search: nothing in flight */
        } else if (!strcmp(cmd, "quit")) {
            break;
        }
    }
    puct_game_free(g);
    return 0;
}
