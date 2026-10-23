/*
 * selfplay.c — self-play training-data generation (AGZ-style, one net).
 *
 *   selfplay <ckpt> <out_dir> <games> [sims=200] [pool=1024] [rows=2048]
 *            [sample_plys=10] [max_plys=300] [block_mb=4] [device=mps]
 *            [seed=1] [fast_sims=0] [fast_p=0.75]
 *
 * A pool of games thinks in lockstep on one mux — every live game
 * gets one `sims`-budget think per puct_mux_step, with the leaf evals of
 * all games batched on the GPU. The stored target and the played move are
 * the regularized policy pi-bar (Grill et al. 2020, LEARN + ACT): sample it
 * for the first sample_plys plies, argmax after (sample_plys=0: sample
 * every ply — always-ACT). There is no root Dirichlet
 * noise — the search itself samples pi-bar at every node (SEARCH).
 *
 * Playout cap randomization (KataGo, Wu 2020 §4): with fast_sims > 0,
 * each step draws ONE budget for the whole pool — fast_sims with
 * probability fast_p, else `sims`. Per-step (not per-move) because the
 * mux is lockstep: a per-move draw would make every step wait on the
 * slowest game, defeating the point; per-step keeps the KataGo per-ply
 * marginal (each ply independently 25% slow) while batches stay
 * full-pool. Every ply records its actual budget; the trainer keeps only
 * plies at the slow cap (chess_ai/train/selfplay_data.py). fast_sims=0
 * (default) is uniform-`sims`, off.
 *
 * Recording: everything comes from the standard readouts — the visit
 * distribution from puct_root_children (π = N/ΣN over visited moves,
 * temperature-free), the mover's root value from puct_root_stats
 * (-(W/N); the root's W is parent-view). No engine-side additions.
 * One SPSHARD1 shard (shard-0.spbin) + DONE.json per run directory; a
 * killed run leaves a truncated last record the reader drops. Games end
 * by the rules (mate/draw) or the ply cap (adjudicated draw, z=0.5) —
 * no resignation in v1.
 *
 * Output: one GAME line per finished game, then SUMMARY. All plies of a
 * game are recorded except the ply that ends a cap game by adjudication
 * is still recorded (it was searched and played).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "chess.h"
#include "chess_match.h"
#include "chess_nn_eval.h"
#include "chess_puct.h"
#include "chess_rng.h"
#include "chess_sp_dump.h"

/* Compact per-slot game record: parallel arrays up to max_pllys, plus a
 * variable-length pool of π pairs (6 B per (slot, prob), appended in ply
 * order — dumped sequentially). SpPlyRec itself is ~1.5 KB; holding it
 * per ply per slot (1024 slots × 300 plies) would be ~0.5 GB. */
typedef struct Slot {
    int      live;
    int      ply;
    uint32_t game_index;
    char     start_fen[SP_FEN_MAX];
    /* parallel arrays allocated to max_pllys; pairs pool grows by
     * doubling (6 B per (slot, prob) pair, appended in ply order) */
    uint8_t *mover;
    int16_t *slot;
    char    (*uci)[SP_UCI_MAX];
    double  *outlook;
    uint16_t *n_pi;
    uint16_t *ply_sims;
    uint16_t *pi_slot_pool;
    float    *pi_prob_pool;
    size_t    pairs_used, pairs_cap;
} Slot;

static Slot   *slots;
static int     max_pllys;

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "selfplay: oom\n");
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) {
        fprintf(stderr, "selfplay: oom\n");
        exit(1);
    }
    return p;
}

static void slot_arrays_alloc(Slot *s) {
    s->mover = xmalloc((size_t)max_pllys * sizeof *s->mover);
    s->slot = xmalloc((size_t)max_pllys * sizeof *s->slot);
    s->uci = xmalloc((size_t)max_pllys * sizeof *s->uci);
    s->outlook = xmalloc((size_t)max_pllys * sizeof *s->outlook);
    s->n_pi = xmalloc((size_t)max_pllys * sizeof *s->n_pi);
    s->ply_sims = xmalloc((size_t)max_pllys * sizeof *s->ply_sims);
    s->pairs_cap = 4096;
    s->pairs_used = 0;
    s->pi_slot_pool = xmalloc(s->pairs_cap * sizeof *s->pi_slot_pool);
    s->pi_prob_pool = xmalloc(s->pairs_cap * sizeof *s->pi_prob_pool);
}

/* Append one ply's π pairs; returns 0 ok, -1 when max_pllys is exceeded
 * (the caller fails the run — the ply arrays are sized by it). */
static int slot_record_ply(Slot *s, uint8_t mover, int16_t played_slot,
                           const char *uci, double outlook, uint16_t n_pi,
                           uint16_t ply_sims,
                           const uint16_t *pslots, const float *pprobs) {
    if (s->ply >= max_pllys)
        return -1;
    if (s->pairs_used + n_pi > s->pairs_cap) {
        while (s->pairs_used + n_pi > s->pairs_cap)
            s->pairs_cap *= 2;
        s->pi_slot_pool = realloc(s->pi_slot_pool,
                                  s->pairs_cap * sizeof *s->pi_slot_pool);
        s->pi_prob_pool = realloc(s->pi_prob_pool,
                                  s->pairs_cap * sizeof *s->pi_prob_pool);
        if (!s->pi_slot_pool || !s->pi_prob_pool) {
            fprintf(stderr, "selfplay: oom\n");
            exit(1);
        }
    }
    int p = s->ply;
    s->mover[p] = mover;
    s->slot[p] = played_slot;
    memcpy(s->uci[p], uci, SP_UCI_MAX);
    s->outlook[p] = outlook;
    s->n_pi[p] = n_pi;
    s->ply_sims[p] = ply_sims;
    for (uint16_t i = 0; i < n_pi; i++) {
        s->pi_slot_pool[s->pairs_used + i] = pslots[i];
        s->pi_prob_pool[s->pairs_used + i] = pprobs[i];
    }
    s->pairs_used += n_pi;
    s->ply++;
    return 0;
}

int main(int argc, char **argv) {
    /* argc > 14 = a caller on the old eps/alpha CLI; fail loudly */
    if (argc < 4 || argc > 14) {
        fprintf(stderr,
                "usage: %s <ckpt> <out_dir> <games> [sims=200] [pool=1024]\n"
                "          [rows=2048] [sample_plys=10, 0=always] [max_plys=300]\n"
                "          [block_mb=4] [device=mps] [seed=1]\n"
                "          [fast_sims=0] [fast_p=0.75]\n",
                argv[0]);
        return 2;
    }
    const char *ckpt = argv[1];
    const char *out_dir = argv[2];
    int games = atoi(argv[3]);
    int sims = argc > 4 ? atoi(argv[4]) : 200;
    int pool = argc > 5 ? atoi(argv[5]) : 1024;
    int rows = argc > 6 ? atoi(argv[6]) : 2048;
    int sample_plys = argc > 7 ? atoi(argv[7]) : 10;
    if (sample_plys < 0)
        sample_plys = 0;
    max_pllys = argc > 8 ? atoi(argv[8]) : 300;
    int block_mb = argc > 9 ? atoi(argv[9]) : 4;
    const char *device = argc > 10 ? argv[10] : "mps";
    uint64_t seed = argc > 11 ? strtoull(argv[11], NULL, 10) : 1;
    int fast_sims = argc > 12 ? atoi(argv[12]) : 0;
    double fast_p = argc > 13 ? atof(argv[13]) : 0.75;
    if (games < 1 || sims < 1 || pool < 1) {
        fprintf(stderr, "selfplay: bad args\n");
        return 2;
    }
    if (max_pllys < 1 || max_pllys > 1000) {
        fprintf(stderr, "selfplay: max_pllys must be 1..1000\n");
        return 2;
    }
    if (fast_sims < 0 || fast_sims >= sims ||
        (fast_sims > 0 && !(fast_p > 0.0 && fast_p < 1.0))) {
        fprintf(stderr, "selfplay: need 0 <= fast_sims < sims (%d) and, "
                        "when fast_sims > 0, 0 < fast_p < 1\n", sims);
        return 2;
    }

    chess_init();
    rng_seed(seed);
    PuctGame **handles = xmalloc((size_t)pool * sizeof *handles);
    if (puct_mux_new(pool, (size_t)block_mb << 20, handles) != 0) {
        fprintf(stderr, "selfplay: mux_new failed\n");
        return 1;
    }
    if (puct_attach_nn(ckpt, device, rows) != 0) {
        fprintf(stderr, "selfplay: puct_attach_nn failed\n");
        return 1;
    }
    slots = xcalloc((size_t)pool, sizeof *slots);   /* live/ply/game_index
                                                       must start at 0 */
    for (int i = 0; i < pool; i++) {
        slot_arrays_alloc(&slots[i]);
        puct_set_suspended(handles[i], true);
    }

    int fd = sp_dump_open_shard(out_dir, 0, seed);
    if (fd < 0) {
        fprintf(stderr, "selfplay: cannot open shard in %s\n", out_dir);
        return 1;
    }

    uint32_t finished = 0, next_game = 0;
    double t0 = now_s();
    while (finished < (uint32_t)games) {
        /* start (recycle) slots before the step so their first think is
         * this step, not a wasted one at a stale position */
        for (int i = 0; i < pool; i++)
            if (!slots[i].live && next_game < games) {
                Slot *s = &slots[i];
                puct_game_reset(handles[i]);
                puct_set_suspended(handles[i], false);
                s->live = 1;
                s->ply = 0;
                s->game_index = next_game++;
                s->pairs_used = 0;
                fen_serialize(puct_state(handles[i]), s->start_fen,
                              sizeof s->start_fen);
            }
        /* One budget draw per step for the whole pool (PCR; header
         * comment). fast_sims = 0 → uniform sims every step, off. */
        int step_sims = sims;
        if (fast_sims && rng_uniform() < fast_p)
            step_sims = fast_sims;
        puct_mux_step(step_sims);

        for (int i = 0; i < pool; i++) {
            Slot *s = &slots[i];
            if (!s->live)
                continue;
            PuctGame *g = handles[i];
            if (!puct_think_done(g))
                continue;        /* defensive: step ends fully delivered */

            uint32_t root_visits;
            double wsum;
            puct_root_stats(g, &root_visits, &wsum);
            double outlook = root_visits ? -(wsum / root_visits) : 0.0;

            /* π̄ is both the stored target and the play policy: sample it
             * for the first sample_plys plies, argmax after. sample_plys=0
             * samples EVERY ply (always-ACT): π̄ sharpens on its own where
             * q-gaps open, so play stays flexible early and near-forced in
             * conversions — the ply-window switch is off. */
            uint16_t pslots[SP_PI_MAX];
            float pprobs[SP_PI_MAX];
            uint16_t pmoves[SP_PI_MAX];
            int n_pi = puct_learn_pi(g, pmoves, pprobs, SP_PI_MAX);
            for (int k = 0; k < n_pi; k++)
                pslots[k] = (uint16_t)chess_policy_slot(puct_state(g),
                                                        pmoves[k]);
            int window = sample_plys > 0 ? sample_plys : max_pllys;
            Move mv = match_pick_move(pmoves, NULL, pprobs, n_pi, s->ply,
                                      window, true, 1.0, 0);
            char uci[SP_UCI_MAX] = {0};
            move_to_uci(mv, uci);
            if (slot_record_ply(s, (uint8_t)puct_state(g)->side,
                                (int16_t)chess_policy_slot(puct_state(g), mv),
                                uci, outlook, n_pi, (uint16_t)step_sims,
                                pslots, pprobs) != 0) {
                fprintf(stderr, "selfplay: ply overflow at game %u\n",
                        s->game_index);
                return 1;
            }
            if (puct_play(g, mv) != 0) {
                fprintf(stderr, "selfplay: puct_play failed\n");
                return 1;
            }

            int term = state_termination((State *)puct_state(g));
            if (term == TERM_PLAYING && s->ply < max_pllys)
                continue;

            /* game over: dump the streamed record */
            uint8_t end_kind = (term != TERM_PLAYING) ? SP_END_NATURAL
                                                      : SP_END_ADJUDICATED;
            double z_white = 0.5;
            if (term == TERM_CHECKMATE)
                z_white = (puct_state(g)->side == BLACK) ? 1.0 : 0.0;
            SpGameRec gr = {.game_index = s->game_index,
                            .n_plies = (uint32_t)s->ply,
                            .end_kind = end_kind,
                            .resigner = -1,
                            .n_sims = (uint32_t)sims,
                            .z_white = z_white,
                            .start_fen = {0},
                            .plies = NULL};
            strcpy(gr.start_fen, s->start_fen);
            if (sp_dump_game_head(fd, &gr) != 0) {
                fprintf(stderr, "selfplay: shard write failed\n");
                return 1;
            }
            size_t off = 0;
            for (int p = 0; p < s->ply; p++) {
                SpPlyRec pr = {.mover = s->mover[p],
                               .played_slot = s->slot[p],
                               .played_uci = {0},
                               .n_pi = s->n_pi[p],
                               .outlook = s->outlook[p],
                               .sims = s->ply_sims[p]};
                memcpy(pr.played_uci, s->uci[p], SP_UCI_MAX);
                memcpy(pr.pi_slot, &s->pi_slot_pool[off],
                       s->n_pi[p] * sizeof *pr.pi_slot);
                memcpy(pr.pi_prob, &s->pi_prob_pool[off],
                       s->n_pi[p] * sizeof *pr.pi_prob);
                off += s->n_pi[p];
                if (sp_dump_ply(fd, &pr) != 0) {
                    fprintf(stderr, "selfplay: shard write failed\n");
                    return 1;
                }
            }
            const char *zs = z_white == 1.0 ? "1" : z_white == 0.0 ? "0" : "0.5";
            printf("GAME idx=%u plies=%d end=%s z=%s\n", s->game_index,
                   s->ply, end_kind == SP_END_NATURAL ? "natural" : "cap",
                   zs);
            fflush(stdout);
            finished++;
            s->live = 0;
            if (next_game >= games)
                puct_set_suspended(g, true);   /* winding down: stop
                                                  thinking at the corpse */
        }
    }
    close(fd);
    if (sp_dump_write_done(out_dir, 1, seed, finished) != 0) {
        fprintf(stderr, "selfplay: DONE.json write failed\n");
        return 1;
    }
    double dt = now_s() - t0;
    /* memory pressure: sampling explores wider than an argmax rule, so the
     * per-game node block is worth watching rather than assuming */
    uint64_t oom = 0, compact = 0;
    for (int i = 0; i < pool; i++) {
        oom += puct_oom_count(handles[i]);
        compact += puct_compact_count(handles[i]);
    }
    printf("SUMMARY games=%u sims=%d fast_sims=%d fast_p=%.2f pool=%d "
           "rows=%d elapsed=%.1fs rate=%.2f games/s oom=%llu compact=%llu\n",
           finished, sims, fast_sims, fast_p, pool, rows, dt,
           dt > 0 ? finished / dt : 0.0,
           (unsigned long long)oom, (unsigned long long)compact);
    return 0;
}
