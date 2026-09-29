/*
 * test_search.c — C unit tests for the Phase 3.1 Gumbel search core
 * (chess_search.h / search.c), in the hand-rolled CHECK style of
 * test_units.c / test_nn.c.
 *
 * The deepest verification — the L2 exact diff against the sequential
 * Python reference (chess_ai/selfplay/gumbel.py), same seeds ⇒ identical
 * games — lives across the boundary in tests/test_search_l2.py, where both
 * engines run in one process under the same evaluator. What is verified
 * here is the C-only surface those tests cannot reach:
 *
 *   1. RNG spec: splitmix64/xoshiro256++ known vectors, uniform/gumbel
 *      golden values generated from rng.py, distribution sanity,
 *      game_seed derivation (the L2 precondition).
 *   2. Policy-slot mapping vectors pinned against features.py.uci_to_index
 *      (from-to, own underpromotion, enemy/mirrored underpromotion).
 *   3. Search properties with fixture evaluators: π′ well-formedness,
 *      forced-move general path (m = 1: one eval, no descents, π′ one-hot,
 *      Eq. 10 outlook), mate-in-1 found and played, resignation trigger
 *      and labels, handoff accounting (no root eval after move 1; a forced
 *      move ends the handoff chain and forces the next root eval).
 *   4. Determinism: same seed ⇒ identical records (the L2 precondition,
 *      C side), different seeds diverge.
 *   5. Long-game arena stress (the per-move reset + handoff copy cycle).
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "chess.h"
#include "chess_nn.h"
#include "chess_rng.h"
#include "chess_search.h"

/* Static game slots (deviation 9 — games live in caller storage). Tests
   run sequentially; at most two games are alive at once (determinism). */
static _Alignas(64) uint8_t g_slot0[SP_GAME_SLOT_SIZE];
static _Alignas(64) uint8_t g_slot1[SP_GAME_SLOT_SIZE];

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

#define FORCED_FEN "k7/8/1K6/8/8/8/R7/8 b - - 0 1"  /* black: only a8b8    */
#define MATE_FEN   "k7/8/1K6/8/8/8/8/2R5 w - - 0 1" /* white: Rc1-c8 = mate */

/* ------------------------------------------------------------------ */
/* RNG spec                                                            */
/* ------------------------------------------------------------------ */

static void test_rng_vectors(void) {
    /* splitmix64(0) first output — the rng.py-pinned vector. */
    uint64_t st = 0;
    CHECK(splitmix64_next(&st) == 0xE220A8397B1DCDAFull);

    /* game_seed derivation (rng.py game_seed, hex from Python). */
    CHECK(rng_game_seed(123, 0) == 0xe050a2a38d8ef504ull);
    CHECK(rng_game_seed(123, 4) == 0xe7beb69cf093afc3ull);

    /* xoshiro256++ stream from seed 7 — the first raw outputs, generated
     * from rng.py (the internal state is private to the Python class; the
     * output stream is the parity contract). */
    Rng256 r;
    rng_init(&r, 7);
    CHECK(rng_next_u64(&r) == 0x0e2c1a002aae913dull);
    CHECK(rng_next_u64(&r) == 0x2c0fc8ddfa4e9e14ull);
    CHECK(rng_next_u64(&r) == 0xb7b311b3b0d45872ull);
    CHECK(rng_next_u64(&r) == 0x6d5d9f6a6318013cull);

    /* Fresh generator: 8 uniforms, then 8 Gumbels — the consumption order
     * the goldens were generated with. */
    rng_init(&r, 7);
    static const double uni_golden[8] = {
        0.055360436478333164, 0.17211585444811778, 0.7175761283586595,
        0.4272098192915053,   0.9636595218812296, 0.4657036891404785,
        0.7239070952365361,   0.32983942955252804,
    };
    static const double gum_golden[8] = {
        4.026566713140335, -0.9606581113852944, -0.774470427845018,
        -0.565567213322398, 1.172840983239658, -0.7791453811906938,
        0.3510694898924522, -0.8459542992917048,
    };
    for (int i = 0; i < 8; i++)
        CHECK(rng_uniform(&r) == uni_golden[i]);
    for (int i = 0; i < 8; i++)
        CHECK(rng_gumbel(&r) == gum_golden[i]);
}

static void test_rng_sanity(void) {
    Rng256 r;
    rng_init(&r, 7);
    double sum = 0.0;
    for (int i = 0; i < 100000; i++) {
        double u = rng_uniform(&r);
        CHECK(u > 0.0 && u < 1.0);
        sum += u;
    }
    CHECK(fabs(sum / 100000.0 - 0.5) < 0.01);
    sum = 0.0;
    for (int i = 0; i < 100000; i++)
        sum += rng_gumbel(&r);
    CHECK(fabs(sum / 100000.0 - 0.5772) < 0.02);   /* Euler–Mascheroni */

    /* Per-game seeds: stable per (seed, index), distinct across indices. */
    CHECK(rng_game_seed(123, 7) == rng_game_seed(123, 7));
    Rng256 seen[64];
    for (uint32_t i = 0; i < 64; i++)
        rng_init(&seen[i], rng_game_seed(123, i));
    for (uint32_t i = 0; i < 64; i++)
        for (uint32_t j = i + 1; j < 64; j++)
            CHECK(seen[i].s[0] != seen[j].s[0] || seen[i].s[1] != seen[j].s[1]
                  || seen[i].s[2] != seen[j].s[2]
                  || seen[i].s[3] != seen[j].s[3]);
}

/* ------------------------------------------------------------------ */
/* Policy-slot mapping vectors (features.py.uci_to_index pins)          */
/* ------------------------------------------------------------------ */

static void test_policy_slot_vectors(void) {
    /* From-to region: c1c8 = (2, 58) → 186 — the value pinned by
     * tests/test_gumbel_ref.py (MATE_SLOT). */
    CHECK(nn_policy_slot(move_make(C1, C8, F_QUIET), WHITE) == 186);

    /* Own underpromotion, white frame: b7xa8=n → rel dir +7, letter n=0:
     * 4096 + (49−48)·9 + 0·3 + 0 = 4105. Queen promo lives in from-to. */
    CHECK(nn_policy_slot(move_make(B7, A8, F_PROMO_CAP_N), WHITE) == 4105);
    CHECK(nn_policy_slot(move_make(B7, B8, F_PROMO_Q), WHITE) == 49 * 64 + 57);

    /* Enemy (mirrored) region — the aux-reply case, current mover's frame
     * for the opponent's move: black b2b1=n mapped in the WHITE frame →
     * dir −8: 4168 + (9−8)·9 + 1·3 + 0 = 4180. */
    CHECK(nn_policy_slot(move_make(B2, B1, F_PROMO_N), WHITE) == 4180);
    /* Same move in the mover's own (black) frame is an own promo: 4108. */
    CHECK(nn_policy_slot(move_make(B2, B1, F_PROMO_N), BLACK) == 4108);
}

/* ------------------------------------------------------------------ */
/* Fixture evaluators (mirrors of gumbel.py's TableEvaluator)           */
/* ------------------------------------------------------------------ */

/* Uniform prior (all raws 0 → logp = −log k) + value from a flag:
 * +1 for the mover (white-wins fixture) or ⅓⅓⅓. */
typedef struct {
    int calls;
    int white_wins;   /* 0: (⅓,⅓,⅓); 1: mover always winning */
} EvalCtx;

static void fixture_eval(void *ctx, const SpGame *g, double *logp,
                         double *wdl) {
    EvalCtx *c = ctx;
    c->calls++;
    int k = sp_request_count(g);
    double lp = -log((double)k);
    for (int i = 0; i < k; i++)
        logp[i] = lp;
    if (c->white_wins) {
        if (sp_request_mover(g) == 0) {
            wdl[0] = 1.0; wdl[1] = 0.0; wdl[2] = 0.0;
        } else {
            wdl[0] = 0.0; wdl[1] = 0.0; wdl[2] = 1.0;
        }
    } else {
        wdl[0] = wdl[1] = wdl[2] = 1.0 / 3.0;
    }
}

/* One nonzero prior at a policy slot (the mate-in-1 fixture). */
typedef struct {
    EvalCtx base;
    int slot;
    double raw;
} PriorCtx;

static void prior_eval(void *ctx, const SpGame *g, double *logp,
                       double *wdl) {
    PriorCtx *p = ctx;
    fixture_eval(&p->base, g, logp, wdl);   /* value part; logp redone below */
    p->base.calls--;                        /* don't double-count */
    int k = sp_request_count(g);
    const uint16_t *slots = sp_request_slots(g);
    double raws[MAX_MOVES], exs[MAX_MOVES];
    for (int i = 0; i < k; i++)
        raws[i] = (slots[i] == p->slot) ? p->raw : 0.0;
    double m = raws[0];
    for (int i = 1; i < k; i++)
        if (raws[i] > m) m = raws[i];
    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        exs[i] = exp(raws[i] - m);
        sum += exs[i];
    }
    double lse = m + log(sum);
    for (int i = 0; i < k; i++)
        logp[i] = raws[i] - lse;
}

/* ------------------------------------------------------------------ */
/* Search properties                                                   */
/* ------------------------------------------------------------------ */

static SpConfig base_cfg(int n) {
    SpConfig cfg;
    sp_config_init(&cfg);
    cfg.n = n;
    return cfg;
}

static void test_pi_properties_startpos(void) {
    SpConfig cfg = base_cfg(8);
    SpGame *g = sp_game_init(g_slot0, &cfg, NULL, 42, 0);
    CHECK(g != NULL);
    EvalCtx ctx = {0, 0};
    sp_play(g, fixture_eval, &ctx, 6);
    CHECK(sp_result_kind(g) == SP_END_ADJUDICATED);
    CHECK(sp_record_count(g) == 6);
    for (int i = 0; i < sp_record_count(g); i++) {
        int n_pi = sp_rec_n_pi(g, i);
        CHECK(n_pi == 8);   /* m = min(n=8, m_cap=16, k=20) */
        double total = 0.0;
        int played = sp_rec_played_slot(g, i);
        int played_in_pi = 0;
        for (int j = 0; j < n_pi; j++) {
            double p = sp_rec_pi_prob(g, i, j);
            CHECK(p >= 0.0);
            total += p;
            if (sp_rec_pi_slot(g, i, j) == played) played_in_pi = 1;
        }
        CHECK(fabs(total - 1.0) < 1e-6);   /* π′ is stored f32 */
        CHECK(played_in_pi);            /* played move comes from the support */
        CHECK(fabs(sp_rec_outlook(g, i)) <= 1.0 + 1e-12);
        CHECK(sp_rec_ply(g, i) == i);   /* plies recorded in order */
        CHECK(sp_rec_mover(g, i) == i % 2);
    }
}

static void test_forced_move_general_path(void) {
    /* No k==1 special case: m = 1, one root eval, no descents, π′ exactly
     * one-hot, outlook = v̂ − c_fpu = 0 under the uniform net. */
    SpConfig cfg = base_cfg(8);
    cfg.sample_plies = 30;

    SpGame *g = sp_game_init(g_slot0, &cfg, FORCED_FEN, 3, 0);
    EvalCtx ctx = {0, 0};
    sp_play(g, fixture_eval, &ctx, 1);   /* move 1 only */
    CHECK(sp_record_count(g) == 1);
    CHECK(ctx.calls == 1);               /* the root eval; no descents at m=1 */
    CHECK(sp_rec_n_pi(g, 0) == 1);
    CHECK(sp_rec_pi_prob(g, 0, 0) == 1.0);
    CHECK(sp_rec_pi_slot(g, 0, 0) == sp_rec_played_slot(g, 0));
    char uci[8];
    sp_rec_played_uci(g, 0, uci);
    CHECK(strcmp(uci, "a8b8") == 0);
    CHECK(sp_rec_outlook(g, 0) == 0.0);

    /* The game progresses normally afterwards (moves 2–3 are searched). */
    SpGame *g2 = sp_game_init(g_slot0, &cfg, FORCED_FEN, 3, 0);
    EvalCtx ctx2 = {0, 0};
    sp_play(g2, fixture_eval, &ctx2, 3);
    CHECK(sp_record_count(g2) == 3);
    CHECK(ctx2.calls > 1);               /* searched moves cost descents */
}

static void test_mate_in_1(void) {
    SpConfig cfg = base_cfg(32);
    cfg.sample_plies = 0;   /* deterministic survivor */
    SpGame *g = sp_game_init(g_slot0, &cfg, MATE_FEN, 5, 0);
    PriorCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    ctx.slot = nn_policy_slot(move_make(C1, C8, F_QUIET), WHITE);
    ctx.raw = 10.0;
    CHECK(ctx.slot == 186);   /* pinned by tests/test_gumbel_ref.py */
    sp_play(g, prior_eval, &ctx, -1);
    CHECK(sp_result_kind(g) == SP_END_NATURAL);
    CHECK(sp_result_z_white(g) == 1.0);
    char uci[8];
    sp_rec_played_uci(g, 0, uci);
    CHECK(strcmp(uci, "c1c8") == 0);
    CHECK(ctx.base.calls <= 1 + 2 * cfg.n);   /* documented overshoot bound */
}

/* Step-loop driver that counts root evaluations per move — the direct
 * probe of the handoff property (sp_request_is_root). */
typedef struct {
    EvalCtx base;
    int root_evals_move1;
    int root_evals_after_move1;
} RootCtx;

static void root_counting_eval(void *ctx, const SpGame *g, double *logp,
                               double *wdl) {
    RootCtx *r = ctx;
    fixture_eval(&r->base, g, logp, wdl);
    if (sp_request_is_root(g)) {
        if (sp_record_count(g) == 0)
            r->root_evals_move1++;
        else
            r->root_evals_after_move1++;
    }
}

static void play_n_moves(SpGame *g, RootCtx *ctx, int moves) {
    double logp[MAX_MOVES], wdl[3];
    while (sp_record_count(g) < moves) {
        int ev = sp_game_step(g);
        if (ev == SP_EVENT_DONE) break;
        if (ev == SP_EVENT_NEED_EVAL) {
            root_counting_eval(ctx, g, logp, wdl);
            sp_deliver(g, logp, wdl);
        }
    }
}

static void test_handoff_roots_free(void) {
    /* Sampling at every ply (default) means the played child always exists
     * ⇒ every root after move 1 is handed off, never re-evaluated. */
    SpConfig cfg = base_cfg(8);
    SpGame *g = sp_game_init(g_slot0, &cfg, NULL, 6, 0);
    RootCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    play_n_moves(g, &ctx, 6);
    CHECK(sp_record_count(g) == 6);
    CHECK(ctx.root_evals_move1 == 1);
    CHECK(ctx.root_evals_after_move1 == 0);
}

static void test_forced_move_breaks_handoff_chain(void) {
    /* A forced move's child is never created ⇒ no handoff ⇒ the next move
     * needs a fresh root eval; after that searched move the chain resumes. */
    SpConfig cfg = base_cfg(8);
    SpGame *g = sp_game_init(g_slot0, &cfg, FORCED_FEN, 3, 0);
    RootCtx ctx;
    memset(&ctx, 0, sizeof ctx);
    play_n_moves(g, &ctx, 3);
    CHECK(sp_record_count(g) == 3);
    CHECK(ctx.root_evals_move1 == 1);          /* move 1: forced, fresh root */
    CHECK(ctx.root_evals_after_move1 == 1);    /* move 2: no handoff         */
}

static void test_resign_labels(void) {
    SpConfig cfg = base_cfg(8);
    cfg.sample_plies = 0;
    SpGame *g = sp_game_init(g_slot0, &cfg, NULL, 9, 0);
    EvalCtx ctx = {0, 1};   /* white-wins fixture: black's outlook is −1 */
    sp_play(g, fixture_eval, &ctx, -1);
    CHECK(sp_result_kind(g) == SP_END_RESIGN);
    CHECK(sp_result_resigner(g) == 1);
    CHECK(sp_result_z_white(g) == 1.0);   /* resigner loses */
    CHECK(sp_record_count(g) == 2);       /* white's move + the resign record */
    CHECK(sp_rec_mover(g, 1) == 1);
    CHECK(sp_rec_played_slot(g, 1) == -1);
    char uci[8];
    sp_rec_played_uci(g, 1, uci);
    CHECK(uci[0] == '\0');
    CHECK(sp_rec_outlook(g, 1) == -1.0);
}

static void test_winning_side_never_resigns(void) {
    SpConfig cfg = base_cfg(8);
    cfg.sample_plies = 0;
    SpGame *g = sp_game_init(g_slot0, &cfg, NULL, 9, 0);
    EvalCtx ctx = {0, 1};
    sp_play(g, fixture_eval, &ctx, 1);
    CHECK(sp_result_kind(g) == SP_END_ADJUDICATED);
    CHECK(sp_record_count(g) == 1);
    CHECK(sp_rec_mover(g, 0) == 0);
    CHECK(sp_rec_outlook(g, 0) == 1.0);
}

/* ------------------------------------------------------------------ */
/* Records: field-for-field comparison (determinism)                   */
/* ------------------------------------------------------------------ */

static int records_equal(const SpGame *a, const SpGame *b) {
    if (sp_record_count(a) != sp_record_count(b)) return 0;
    if (sp_result_kind(a) != sp_result_kind(b)) return 0;
    if (sp_result_resigner(a) != sp_result_resigner(b)) return 0;
    if (sp_result_z_white(a) != sp_result_z_white(b)) return 0;
    char fa[CHESS_FEN_MAX], fb[CHESS_FEN_MAX];
    sp_result_start_fen(a, fa);
    sp_result_start_fen(b, fb);
    if (strcmp(fa, fb) != 0) return 0;
    for (int i = 0; i < sp_record_count(a); i++) {
        if (sp_rec_ply(a, i) != sp_rec_ply(b, i)) return 0;
        if (sp_rec_mover(a, i) != sp_rec_mover(b, i)) return 0;
        if (sp_rec_played_slot(a, i) != sp_rec_played_slot(b, i)) return 0;
        sp_rec_played_uci(a, i, fa);
        sp_rec_played_uci(b, i, fb);
        if (strcmp(fa, fb) != 0) return 0;
        sp_rec_fen_before(a, i, fa);
        sp_rec_fen_before(b, i, fb);
        if (strcmp(fa, fb) != 0) return 0;
        if (sp_rec_n_pi(a, i) != sp_rec_n_pi(b, i)) return 0;
        for (int j = 0; j < sp_rec_n_pi(a, i); j++) {
            if (sp_rec_pi_slot(a, i, j) != sp_rec_pi_slot(b, i, j)) return 0;
            if (sp_rec_pi_prob(a, i, j) != sp_rec_pi_prob(b, i, j)) return 0;
        }
        if (sp_rec_outlook(a, i) != sp_rec_outlook(b, i)) return 0;
    }
    return 1;
}

static void test_determinism_and_divergence(void) {
    SpConfig cfg = base_cfg(16);
    cfg.sample_plies = 30;

    SpGame *a1 = sp_game_init(g_slot0, &cfg, NULL, 11, 4);
    SpGame *a2 = sp_game_init(g_slot1, &cfg, NULL, 11, 4);
    EvalCtx c1 = {0, 0}, c2 = {0, 0};
    sp_play(a1, fixture_eval, &c1, 10);
    sp_play(a2, fixture_eval, &c2, 10);
    CHECK(records_equal(a1, a2));

    SpGame *b1 = sp_game_init(g_slot0, &cfg, NULL, 1, 0);
    SpGame *b2 = sp_game_init(g_slot1, &cfg, NULL, 2, 0);
    EvalCtx d1 = {0, 0}, d2 = {0, 0};
    sp_play(b1, fixture_eval, &d1, 10);
    sp_play(b2, fixture_eval, &d2, 10);
    CHECK(sp_record_count(b1) == 10 && sp_record_count(b2) == 10);
    int differ = 0;
    for (int i = 0; i < 10; i++)
        if (sp_rec_played_slot(b1, i) != sp_rec_played_slot(b2, i))
            differ = 1;
    CHECK(differ);
}

static void test_long_game_arena_stress(void) {
    /* 120 moves of arena reset + handoff subtree copy per move. */
    SpConfig cfg = base_cfg(8);
    SpGame *g = sp_game_init(g_slot0, &cfg, NULL, 77, 3);
    EvalCtx ctx = {0, 0};
    sp_play(g, fixture_eval, &ctx, 120);
    CHECK(sp_record_count(g) == 120);
    CHECK(sp_result_kind(g) == SP_END_ADJUDICATED);
}

int main(void) {
    chess_init();

    test_rng_vectors();
    test_rng_sanity();
    test_policy_slot_vectors();
    test_pi_properties_startpos();
    test_forced_move_general_path();
    test_mate_in_1();
    test_handoff_roots_free();
    test_forced_move_breaks_handoff_chain();
    test_resign_labels();
    test_winning_side_never_resigns();
    test_determinism_and_divergence();
    test_long_game_arena_stress();

    printf("test_search: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
