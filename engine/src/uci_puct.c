/*
 * uci_puct.c — minimal UCI adapter around the PUCT search + real net.
 *
 *   PUCT_CKPT=runs/policy_value_net_42k.pt PUCT_DEVICE=mps \
 *   PUCT_SIMS=256 PUCT_SAMPLE_PLYS=10 build/uci_puct
 *
 * Protocol: the sequential subset python-chess speaks — uci, isready,
 * setoption (ignored), ucinewgame, position (startpos|fen) [moves ...],
 * go [nodes N], quit. "go nodes N" runs N PUCT simulations; without an
 * explicit count, PUCT_SIMS is used. "stop" is accepted and ignored (the
 * search is blocking, there is nothing to stop mid-way).
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

static uint64_t rng_state = 0x12345678ull;

static double rng_uniform(void) {          /* splitmix64 */
    rng_state += 0x9E3779B97F4A7C15ull;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return (double)((z ^ (z >> 31)) >> 11) / 9007199254740992.0;
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
static Move uci_to_move(const State *s, const char *u) {
    if (strlen(u) < 4)
        return 0;
    int from = (u[0] - 'a') + (u[1] - '1') * 8;
    int to = (u[2] - 'a') + (u[3] - '1') * 8;
    static const char promo_char[4] = {'n', 'b', 'r', 'q'};
    MoveList ml;
    gen_legal((State *)s, &ml);
    Move match = 0;
    for (int i = 0; i < ml.count; i++) {
        Move m = ml.moves[i];
        if (move_from(m) != from || move_to(m) != to)
            continue;
        if (u[4]) {
            if (move_is_promo(m) && promo_char[move_flags(m) & 3] == u[4])
                return m;
        } else if (!move_is_promo(m)) {
            match = m;
        }
    }
    return match;
}

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
    PuctGame *g = puct_game_new(NULL, 0);
    if (puct_attach_nn(g, ckpt, device) != 0) {
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
            g = puct_game_new(NULL, 0);
            if (puct_attach_nn(g, ckpt, device) != 0)
                return 1;
        } else if (!strcmp(cmd, "position")) {
            char *tok = strtok(NULL, " \t\n");
            puct_game_free(g);
            const char *err = NULL;
            char fen[96] = "";
            if (tok && !strcmp(tok, "fen")) {
                char *p = fen;
                while ((tok = strtok(NULL, " \t\n")) &&
                       strcmp(tok, "moves") &&
                       p + strlen(tok) + 1 < fen + sizeof fen) {
                    p += sprintf(p, "%s ", tok);
                }
                g = puct_game_new(fen, 0);
            } else if (tok && !strcmp(tok, "startpos")) {
                g = puct_game_new(NULL, 0);
                tok = strtok(NULL, " \t\n");        /* "moves" or NULL  */
                if (tok && strcmp(tok, "moves"))
                    tok = NULL;
            } else {
                g = NULL;
                err = "expected startpos|fen";
            }
            if (!g) {
                fprintf(stderr, "position: %s\n", err ? err : "bad fen");
                return 1;
            }
            if (puct_attach_nn(g, ckpt, device) != 0)
                return 1;
            if (tok && !strcmp(tok, "moves"))      /* fen replay case     */
                tok = strtok(NULL, " \t\n");
            for (; tok; tok = strtok(NULL, " \t\n")) {
                Move m = uci_to_move(puct_state(g), tok);
                if (!m || puct_play(g, m) != 0) {
                    fprintf(stderr, "bad move %s\n", tok);
                    return 1;
                }
            }
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
            if (m && puct_play(g, m) != 0)
                return 1;
        } else if (!strcmp(cmd, "stop")) {
            /* blocking search: nothing in flight */
        } else if (!strcmp(cmd, "quit")) {
            break;
        }
    }
    puct_game_free(g);
    return 0;
}
