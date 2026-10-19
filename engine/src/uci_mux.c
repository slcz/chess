/*
 * uci_mux.c — multi-game think server for the gate referee
 * (scripts/gate_match.py). One process = one network + one mux hosting
 * ALL of that network's games, so a phase's thinks batch across every
 * game on the GPU (the referee plays the nets in net-lockstep waves).
 *
 *   PUCT_CKPT=runs/<net>.pt PUCT_DEVICE=cuda uci_mux <games> [rows=0]
 *
 * Line protocol on stdin, one reply line per command that produces one:
 *   game <gid> fen <FEN>            plant the game at a position
 *                                   (puct_game_set_fen — fresh tree)
 *   game <gid> moves <uci> [...]    advance the game (opponent's replies
 *                                   since the last phase; subtree kept)
 *   game <gid> go nodes <sims>      queue this game's think for the phase
 *   run                             one mux round over ALL queued games
 *   → game <gid> bestmove <uci>     … one line per queued game, in queue
 *                                     order (argmax visits; sims <= 1 is
 *                                     the raw-policy player — argmax
 *                                     prior)
 *   quit                            exit 0
 *
 * Queued games must share the phase's sims (the referee sends one value
 * per phase); a mixed phase is refused. Non-queued games stay suspended
 * and are skipped by the mux. A `fen` or `moves` line may arrive between
 * phases only. Malformed input, illegal moves, or an unknown gid are
 * fatal (exit 1) — the referee owns correctness.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chess.h"
#include "chess_nn_eval.h"
#include "chess_rng.h"

typedef struct {
    PuctGame *g;
    int      pending;      /* queued for the current phase */
    int      sims;
} Slot;

static int g_sample_plys;  /* PUCT_SAMPLE_PLYS: sampled opening plies */

static Slot *slots;
static int   n_slots;
static int  *queue;        /* gids in `go` order */
static int   queued;

static void fail(const char *msg) {
    fprintf(stderr, "uci_mux: %s\n", msg);
    exit(1);
}

static Slot *slot_of(int gid) {
    if (gid < 0 || gid >= n_slots)
        fail("bad gid");
    return &slots[gid];
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <games> [rows=0] [device=cuda]\n"
                        "env: PUCT_CKPT (required), PUCT_DEVICE, "
                        "PUCT_WALKERS\n",
                argv[0]);
        return 2;
    }
    n_slots = atoi(argv[1]);
    if (n_slots < 1)
        n_slots = 1;
    int rows = argc > 2 ? atoi(argv[2]) : 0;
    const char *device = argc > 3 ? argv[3] : "cuda";
    const char *ckpt = getenv("PUCT_CKPT");
    if (!ckpt || !*ckpt) {
        fprintf(stderr, "PUCT_CKPT not set\n");
        return 2;
    }
    if (rows <= 0)
        rows = 8 * n_slots;    /* batch scale comes from the pool */

    chess_init();
    PuctGame **handles = calloc((size_t)n_slots, sizeof *handles);
    slots = calloc((size_t)n_slots, sizeof *slots);
    queue = calloc((size_t)n_slots, sizeof *queue);
    if (!handles || !slots || !queue)
        fail("oom");
    PuctMux *m = puct_mux_new(n_slots, 0, handles);
    if (!m)
        fail("mux_new failed");
    if (puct_attach_nn(ckpt, device, rows) != 0)
        fail("puct_attach_nn failed");
    const char *wenv = getenv("PUCT_WALKERS");
    const char *senv = getenv("PUCT_SAMPLE_PLYS");
    if (senv && *senv) {
        g_sample_plys = atoi(senv);
        rng_seed(0xC0FFEEu);   /* fixed stream: reproducible sampled evals */
    }
    for (int i = 0; i < n_slots; i++) {
        slots[i].g = handles[i];
        puct_set_suspended(slots[i].g, true);
        if (wenv && *wenv)
            puct_set_walkers(slots[i].g, atoi(wenv));
    }
    printf("ready\n");     /* attach complete: the referee may start */
    fflush(stdout);

    char line[8192];
    while (fgets(line, sizeof line, stdin)) {
        char *cmd = strtok(line, " \t\n");
        if (!cmd)
            continue;
        if (!strcmp(cmd, "quit"))
            break;
        if (!strcmp(cmd, "run")) {
            if (!queued)
                continue;
            int sims = slots[queue[0]].sims;
            puct_mux_step(m, sims);   /* ends fully delivered */
            for (int q = 0; q < queued; q++) {
                Slot *s = &slots[queue[q]];
                Move mv = 0;
                if (sims <= 1) {
                    /* raw-policy player (the ladder baseline): think(1)
                     * only evaluates the root — argmax the PRIORS, not
                     * the visit counts (one child would otherwise own
                     * the single visit) */
                    uint16_t mv2[MAX_MOVES];
                    float prior[MAX_MOVES];
                    int n = puct_root_children(s->g, mv2, NULL, NULL,
                                               prior);
                    if (n <= 0)
                        fail("no legal move: game was terminal");
                    int best = 0;
                    for (int i = 1; i < n; i++)
                        if (prior[i] > prior[best])
                            best = i;
                    mv = mv2[best];
                } else {
                    mv = puct_best_move(s->g);
                    /* PUCT_SAMPLE_PLYS: self-play-style variety — sample
                     * ∝ visits (τ=1, zero-visit unplayable, exactly the
                     * selfplay driver's move choice) for the first N
                     * plies instead of argmax. Default 0 = the standing
                     * deterministic eval protocol, untouched. */
                    if (g_sample_plys > 0
                        && puct_state(s->g)->ply < g_sample_plys) {
                        uint16_t mv3[MAX_MOVES];
                        uint32_t vi[MAX_MOVES];
                        double w[MAX_MOVES];
                        int n = puct_root_children(s->g, mv3, vi, NULL,
                                                   NULL);
                        if (n <= 0)
                            fail("no legal move: game was terminal");
                        for (int i = 0; i < n; i++)
                            w[i] = (double)vi[i];
                        mv = mv3[sample_index(w, n, 1.0)];
                    }
                }
                if (!mv)
                    fail("no legal move: game was terminal");
                char u[8];
                move_to_uci(mv, u);
                printf("game %d bestmove %s\n", queue[q], u);
                /* advance our own tree with our own move: the root
                 * becomes the played child, its subtree (our analysis of
                 * the likely replies) survives for the next phase */
                if (puct_play(s->g, mv) != 0)
                    fail("puct_play failed on own bestmove");
                puct_set_suspended(s->g, true);
                s->pending = 0;
            }
            queued = 0;
            fflush(stdout);
            continue;
        }
        if (!strcmp(cmd, "game")) {
            char *gid_s = strtok(NULL, " \t\n");
            if (!gid_s)
                fail("missing gid");
            Slot *s = slot_of(atoi(gid_s));
            char *sub = strtok(NULL, " \t\n");
            if (!sub)
                fail("missing subcommand");
            if (!strcmp(sub, "fen")) {
                char *fen = strtok(NULL, "");
                if (!fen || puct_game_set_fen(s->g, fen) != 0)
                    fail("bad fen");
                if (getenv("GATE_DEBUG")) {
                    char f[96];
                    fen_serialize(puct_state(s->g), f, sizeof f);
                    fprintf(stderr, "DBG planted: %s\n", f);
                }
            } else if (!strcmp(sub, "moves")) {
                char *u;
                while ((u = strtok(NULL, " \t\n"))) {
                    Move mv = uci_to_move(puct_state(s->g), u);
                    if (!mv || puct_play(s->g, mv) != 0)
                        fail("illegal move");
                }
            } else if (!strcmp(sub, "go")) {
                char *kw = strtok(NULL, " \t\n");
                char *v = strtok(NULL, " \t\n");
                if (!kw || strcmp(kw, "nodes") || !v)
                    fail("go needs 'nodes <sims>'");
                if (s->pending)
                    fail("game already queued this phase");
                if (queued && atoi(v) != slots[queue[0]].sims)
                    fail("mixed sims in one phase");
                s->pending = 1;
                s->sims = atoi(v);
                puct_set_suspended(s->g, false);
                queue[queued++] = s - slots;
            } else {
                fail("unknown subcommand");
            }
            continue;
        }
        fprintf(stderr, "uci_mux: unknown command '%s'\n", cmd);
        exit(1);
    }
    return 0;
}
