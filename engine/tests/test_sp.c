/*
 * test_sp.c — C unit tests for the Phase 3.3b parallel self-play engine
 * (chess_sp.h / sp_engine.c), in the CHECK style of the other test
 * binaries. Uses plain C fixture batch evaluators (no Python — the
 * embedded-CPython evaluator is exercised from tests/test_sp_l3.py).
 *
 * The gates:
 *   1. **Parallel ≡ sequential (exact).** With a batch-invariant evaluator,
 *      the wave-concurrent engine must reproduce the sequential mode's
 *      games bit-for-bit: candidate subtrees are disjoint, same-candidate
 *      descents are serial behind the wave barrier, and the only
 *      delivery-order-dependent accumulation (the root's own stats) is
 *      never read by selection. This is the strongest possible check of
 *      the parallel machinery — any schedule dependence shows up as a
 *      record mismatch. Run under varying worker counts and tiny buffers
 *      (B=4 forces many epochs, stragglers and deferred parks). The
 *      parallel records are verified through the SHARD FILE round trip:
 *      the workers dump them (chess_sp_dump.h), the test parses the files
 *      back and diffs each against a freshly played sequential twin.
 *   2. Deterministic-interleave mode (R6/L3): bit-reproducible run to run,
 *      and — with the batch-invariant fixture — equal to the threaded runs
 *      and the sequential mode. Any divergence is a data race.
 *   3. Budget accounting: exactly game_budget games complete
 *      (zero overshoot via the games_started reservation), telemetry
 *      consistent (games_started ≤ budget, evals ≥ 1 per move searched,
 *      every epoch evaluated ≥ 1 request, max_batch ≤ capacity).
 *   4. Record invariants on the dumped output (π′ sums to 1 over A₀,
 *      played slot in the support, mover alternation, resignation labels).
 *
 * Fixture value note: the batch contract hands the engine f32 logp/wdl,
 * which the engine widens to double on delivery. The sequential driver in
 * the comparison therefore casts its fixtures through float too — the two
 * engines must be fed bit-identical numbers for the exact gate to be
 * meaningful. (π′ probabilities are f32 in the in-memory records too, so
 * the file round trip is lossless for them.)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "chess.h"
#include "chess_sp.h"
#include "chess_sp_dump.h"
#include "chess_sp_eval.h"

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

/* ------------------------------------------------------------------ */
/* Fixture batch evaluator (batch-invariant: the parallel engine's      */
/* games then depend only on their seeds, like the sequential mode's)   */
/* ------------------------------------------------------------------ */

typedef struct {
    int white_wins;   /* 0: (⅓,⅓,⅓) for the mover; 1: mover always winning */
} BatchCtx;

static int fixture_batch(void *ctx, SpEvalBatch *b) {
    BatchCtx *c = ctx;
    int size = sp_batch_size(b);
    const int32_t *offs = sp_batch_offsets(b);
    const int32_t *mover = sp_batch_mover(b);
    float *lp = sp_batch_out_logp(b);
    float *wdl = sp_batch_out_wdl(b);
    for (int i = 0; i < size; i++) {
        int k = offs[i + 1] - offs[i];
        float v = (float)(-log((double)k));   /* f32 like the net's tensor */
        for (int j = 0; j < k; j++)
            lp[offs[i] + j] = v;
        float *w = wdl + 3 * i;
        if (c->white_wins && mover[i] == 0) {
            w[0] = 1.0f; w[1] = 0.0f; w[2] = 0.0f;
        } else if (c->white_wins) {
            w[0] = 0.0f; w[1] = 0.0f; w[2] = 1.0f;
        } else {
            w[0] = w[1] = w[2] = 1.0f / 3.0f;
        }
    }
    return 0;
}

/* Sequential-mode twin of the batch fixture: identical numbers, cast
 * through float exactly like the engine's delivery path. */
static void fixture_seq(void *ctx, const SpGame *g, double *logp,
                        double *wdl) {
    BatchCtx *c = ctx;
    int k = sp_request_count(g);
    double v = (double)(float)(-log((double)k));
    for (int i = 0; i < k; i++)
        logp[i] = v;
    if (c->white_wins && sp_request_mover(g) == 0) {
        wdl[0] = 1.0; wdl[1] = 0.0; wdl[2] = 0.0;
    } else if (c->white_wins) {
        wdl[0] = 0.0; wdl[1] = 0.0; wdl[2] = 1.0;
    } else {
        wdl[0] = wdl[1] = wdl[2] = 1.0 / 3.0;
    }
}

/* ------------------------------------------------------------------ */
/* Shard file reader — the format twin of chess_sp_dump.h. Tests run on  */
/* little-endian hosts (macOS arm64 / Linux x86_64), so fields decode    */
/* with direct memcpy loads. The trainer's Python reader (spengine.py)   */
/* is the authoritative independent implementation.                      */
/* ------------------------------------------------------------------ */

typedef struct { uint16_t slot; float prob; } TPi;
typedef struct {
    uint8_t  mover;
    int16_t  played;
    char     uci[8];
    uint16_t n_pi;
    double   outlook;
    TPi      pi[SP_REC_CANDIDATES];
} TPly;
typedef struct {
    uint32_t gi, n_plies, n_sims;
    int      kind, resigner;
    double   z;
    char     start_fen[CHESS_FEN_MAX];
    TPly    *plies;              /* into the arena below */
} TGame;

#define TGAME_MAX 512
static TGame g_games[TGAME_MAX];
static int   g_ngames;
static uint8_t  g_arena[8u << 20];      /* ply storage, bump-allocated */
static size_t   g_arena_used;

static void reset_games(void) { g_ngames = 0; g_arena_used = 0; }

static void *arena_alloc(size_t bytes) {
    if (g_arena_used + bytes > sizeof g_arena)
        return NULL;
    void *p = g_arena + g_arena_used;
    g_arena_used += bytes;
    return p;
}

static uint16_t rd_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static int16_t  rd_i16(const uint8_t *p) { int16_t v; memcpy(&v, p, 2); return v; }
static uint32_t rd_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

/* Parse one shard file, appending its games. 0 ok, -1 on any malformed
 * content (a truncated LAST record from a crashed run also lands here —
 * these tests only read runs that finished cleanly). */
static int parse_shard(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 24) { fclose(f); return -1; }
    uint8_t *buf = malloc((size_t)len);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        free(buf); fclose(f); return -1;
    }
    fclose(f);
    if (memcmp(buf, "SPSHARD1", 8) != 0) { free(buf); return -1; }

    size_t off = 24;
    while (off < (size_t)len) {
        if (g_ngames >= TGAME_MAX || off + 22u + CHESS_FEN_MAX > (size_t)len)
            goto fail;
        TGame *t = &g_games[g_ngames];
        t->gi = rd_u32(buf + off);
        t->n_plies = rd_u32(buf + off + 4);
        t->kind = buf[off + 8];
        t->resigner = (int8_t)buf[off + 9];
        t->n_sims = rd_u32(buf + off + 10);
        memcpy(&t->z, buf + off + 14, 8);
        memcpy(t->start_fen, buf + off + 22, CHESS_FEN_MAX);
        off += 22u + CHESS_FEN_MAX;
        t->plies = arena_alloc(t->n_plies * sizeof(TPly));
        if (!t->plies) goto fail;
        for (uint32_t i = 0; i < t->n_plies; i++) {
            TPly *p = &t->plies[i];
            if (off + 21u > (size_t)len) goto fail;
            p->mover = buf[off];
            p->played = rd_i16(buf + off + 1);
            memcpy(p->uci, buf + off + 3, 8);
            p->n_pi = rd_u16(buf + off + 11);
            memcpy(&p->outlook, buf + off + 13, 8);
            off += 21u;
            if (p->n_pi > SP_REC_CANDIDATES
                || off + (size_t)p->n_pi * 6u > (size_t)len)
                goto fail;
            for (int j = 0; j < p->n_pi; j++) {
                p->pi[j].slot = rd_u16(buf + off);
                memcpy(&p->pi[j].prob, buf + off + 2, 4);
                off += 6u;
            }
        }
        g_ngames++;
    }
    free(buf);
    return 0;
fail:
    free(buf);
    return -1;
}

static int read_all_shards(const char *dir, int workers) {
    reset_games();
    for (int w = 0; w < workers; w++) {
        char path[SP_DUMP_PATH_MAX + 32];
        sp_dump_shard_path(dir, w, path, sizeof path);
        if (parse_shard(path) != 0) return -1;
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* Run helpers                                                         */
/* ------------------------------------------------------------------ */

typedef struct {
    int workers, budget, buffer, max_plies, interleave;
    uint64_t seed;
} RunShape;

/* Static storage (deviation 9): the engine and the sequential comparison
   game live in caller-reserved anonymous mappings — zero-filled like
   BSS, committed lazily, reserved up front (a too-small machine fails
   loudly here, per the design). The engine itself never allocates. Tests
   run engines sequentially, so one block is reused. */
static void *reserve_static(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(p != MAP_FAILED);
    return p;
}
static void *g_seq_slot = NULL;

/* FNV-1a over what a record carries — the R6 "golden game hash". The
 * per-ply FENs are not hashed: they are not in the shards either (the
 * reader replays them), and their correctness is the rules engine's
 * own gate. */
static uint64_t hash_u64(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        h ^= (v >> (8 * i)) & 0xFF;
        h *= 0x100000001B3ull;
    }
    return h;
}
static uint64_t hash_double(uint64_t h, double d) {
    union { double d; uint64_t u; } u = { .d = d };
    return hash_u64(h, u.u);
}

static uint64_t parsed_hash(const TGame *t) {
    uint64_t h = 0xCBF29CE484222325ull;
    h = hash_u64(h, t->kind);
    h = hash_u64(h, (uint64_t)(int64_t)t->resigner);
    h = hash_double(h, t->z);
    for (const char *p = t->start_fen; *p; p++) h = hash_u64(h, (uint8_t)*p);
    for (uint32_t i = 0; i < t->n_plies; i++) {
        const TPly *p = &t->plies[i];
        for (const char *c = p->uci; *c; c++) h = hash_u64(h, (uint8_t)*c);
        h = hash_u64(h, i);                       /* ply == index */
        h = hash_u64(h, p->mover);
        h = hash_u64(h, (uint64_t)(int64_t)p->played);
        h = hash_double(h, p->outlook);
        h = hash_u64(h, p->n_pi);
        for (int j = 0; j < p->n_pi; j++) {
            h = hash_u64(h, p->pi[j].slot);
            h = hash_double(h, p->pi[j].prob);
        }
    }
    return h;
}

/* Sequential twin: play game_index with the same seed/config via sp_play.
 * Lives in g_seq_slot until the next call. */
static SpGame *run_sequential(uint64_t seed, int game_index, int max_plies,
                              BatchCtx *ctx) {
    SpConfig cfg;
    sp_config_init(&cfg);
    cfg.n = 8;
    SpGame *g = sp_game_init(g_seq_slot, &cfg, NULL, seed,
                             (uint32_t)game_index);
    CHECK(g != NULL);
    sp_play(g, fixture_seq, ctx, max_plies);
    return g;
}

/* Field-for-field exact diff of a parsed shard game against the freshly
 * played sequential twin. Per-ply FENs are not compared — they are
 * derived by replay on BOTH sides of the file boundary, so the diff
 * covers everything the file carries; FEN generation itself is the rules
 * engine's gate (test_units/perft). */
static int parsed_equals_sequential(const TGame *t, const SpGame *g) {
    if (t->gi != (uint32_t)sp_result_game_index(g)) return 0;
    if (t->kind != sp_result_kind(g)) return 0;
    if (t->resigner != sp_result_resigner(g)) return 0;
    if (t->z != sp_result_z_white(g)) return 0;
    if (t->n_sims != (uint32_t)sp_result_n_sims(g)) return 0;
    char fen[CHESS_FEN_MAX];
    sp_result_start_fen(g, fen);
    if (strncmp(t->start_fen, fen, CHESS_FEN_MAX) != 0) return 0;
    if (t->n_plies != (uint32_t)sp_record_count(g)) return 0;
    for (uint32_t i = 0; i < t->n_plies; i++) {
        const TPly *p = &t->plies[i];
        if (sp_rec_ply(g, (int)i) != (int)i) return 0;
        if (p->mover != (uint8_t)sp_rec_mover(g, (int)i)) return 0;
        if (p->played != (int16_t)sp_rec_played_slot(g, (int)i)) return 0;
        char uci[8];
        sp_rec_played_uci(g, (int)i, uci);
        if (strncmp(p->uci, uci, 8) != 0) return 0;
        if (p->n_pi != (uint16_t)sp_rec_n_pi(g, (int)i)) return 0;
        if (p->outlook != sp_rec_outlook(g, (int)i)) return 0;
        for (int j = 0; j < p->n_pi; j++) {
            if (p->pi[j].slot != (uint16_t)sp_rec_pi_slot(g, (int)i, j))
                return 0;
            if (p->pi[j].prob != (float)sp_rec_pi_prob(g, (int)i, j))
                return 0;
        }
    }
    return 1;
}

/* Run the engine to completion, dumping records into `record_dir` (NULL:
 * dump off). Verifies the budget/accounting telemetry. Returns the run's
 * max_batch (the batch-fill gate below wants it). */
static int run_engine(RunShape rs, BatchCtx *eval_ctx, const char *record_dir) {
    SpEngineConfig cfg;
    sp_engine_config_init(&cfg);
    cfg.game_cfg.n = 8;
    cfg.unit_seed = rs.seed;
    cfg.game_budget = rs.budget;
    cfg.n_workers = rs.workers;
    cfg.buffer_requests = rs.buffer;
    cfg.max_plies = rs.max_plies;
    cfg.interleave = rs.interleave;
    if (record_dir)
        snprintf(cfg.record_dir, sizeof cfg.record_dir, "%s", record_dir);

    SpEvaluator ev = { .ctx = eval_ctx, .init = NULL,
                       .batch = fixture_batch, .free_fn = NULL };
    if (!g_seq_slot)
        g_seq_slot = reserve_static(SP_GAME_SLOT_SIZE);
    void *engine_storage = reserve_static(SP_ENGINE_STATIC_SIZE);
    SpEngine *e = sp_engine_init(engine_storage,
                                 SP_ENGINE_STATIC_SIZE, &cfg, &ev);
    CHECK(e != NULL);
    CHECK(sp_engine_start(e) == 0);
    sp_engine_wait(e);
    CHECK(sp_engine_error(e) == 0);

    CHECK(sp_engine_games_done(e) == rs.budget);   /* cumulative counter */
    CHECK(sp_engine_games_started(e) <= rs.budget);
    CHECK(sp_engine_evals(e) > 0);
    CHECK(sp_engine_max_batch(e) <= rs.workers * rs.buffer);
    int max_batch = sp_engine_max_batch(e);
    sp_engine_destroy(e);
    munmap(engine_storage, SP_ENGINE_STATIC_SIZE);
    return max_batch;
}

/* Read back a finished run's shards and diff every game against its
 * sequential twin. Returns the number of mismatches (-1: unreadable). */
static int check_shards_vs_sequential(const char *dir, RunShape rs,
                                      BatchCtx *bctx) {
    if (read_all_shards(dir, rs.workers) != 0) return -1;
    if (g_ngames != rs.budget) return -1;
    int failures = 0;
    for (int i = 0; i < g_ngames; i++) {
        SpGame *seq = run_sequential(rs.seed, (int)g_games[i].gi,
                                     rs.max_plies, bctx);
        if (!parsed_equals_sequential(&g_games[i], seq))
            failures++;
    }
    return failures;
}

/* ------------------------------------------------------------------ */
/* The gates                                                           */
/* ------------------------------------------------------------------ */

static const char *g_shard_dir;   /* shared mkdtemp dir, set in main */

static void test_parallel_matches_sequential(void) {
    /* Vary the schedule pressure: worker counts, tiny buffers (B=4 forces
     * many epochs, stragglers and deferred parks), interleave mode. The
     * exact gate runs through the shard round trip: every dumped game is
     * diffed against its sequential twin. */
    const RunShape shapes[] = {
        { .workers = 1, .budget = 6,  .buffer = 256, .max_plies = 12, .interleave = 0, .seed = 500 },
        { .workers = 3, .budget = 12, .buffer = 4,   .max_plies = 12, .interleave = 0, .seed = 500 },
        { .workers = 4, .budget = 12, .buffer = 16,  .max_plies = 12, .interleave = 0, .seed = 500 },
        { .workers = 3, .budget = 12, .buffer = 4,   .max_plies = 12, .interleave = 1, .seed = 500 },
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        BatchCtx bctx = { .white_wins = 0 };
        run_engine(shapes[s], &bctx, g_shard_dir);
        CHECK(check_shards_vs_sequential(g_shard_dir, shapes[s], &bctx) == 0);
    }
}

static void test_parallel_resign_path(void) {
    /* The white-wins fixture: black resigns on its first turn. The resign
     * records must match the sequential mode exactly too. */
    RunShape rs = { .workers = 2, .budget = 4, .buffer = 8,
                    .max_plies = -1, .interleave = 0, .seed = 901 };
    BatchCtx bctx = { .white_wins = 1 };
    run_engine(rs, &bctx, g_shard_dir);
    CHECK(read_all_shards(g_shard_dir, rs.workers) == 0);
    CHECK(g_ngames == rs.budget);
    int label_failures = 0;
    for (int i = 0; i < g_ngames; i++) {
        const TGame *t = &g_games[i];
        if (!(t->kind == SP_END_RESIGN && t->resigner == 1 && t->z == 1.0))
            label_failures++;
    }
    CHECK(label_failures == 0);
    CHECK(check_shards_vs_sequential(g_shard_dir, rs, &bctx) == 0);
}

static void test_interleave_is_bit_reproducible(void) {
    /* Golden game hashes (R6): two interleave runs and one threaded run,
     * identical seeds — the per-game hashes must all agree. Bit-level
     * reproducibility, verified over the shard round trip. */
    RunShape rs = { .workers = 3, .budget = 8, .buffer = 4,
                    .max_plies = 12, .interleave = 1, .seed = 777 };
    BatchCtx fx = { .white_wins = 0 };
    static uint64_t h1[TGAME_MAX], h2[TGAME_MAX], h3[TGAME_MAX];

    run_engine(rs, &fx, g_shard_dir);
    CHECK(read_all_shards(g_shard_dir, rs.workers) == 0);
    CHECK(g_ngames == rs.budget);
    for (int i = 0; i < g_ngames; i++) h1[g_games[i].gi] = parsed_hash(&g_games[i]);

    run_engine(rs, &fx, g_shard_dir);
    CHECK(read_all_shards(g_shard_dir, rs.workers) == 0);
    CHECK(g_ngames == rs.budget);
    for (int i = 0; i < g_ngames; i++) h2[g_games[i].gi] = parsed_hash(&g_games[i]);

    RunShape threaded = rs;
    threaded.interleave = 0;
    run_engine(threaded, &fx, g_shard_dir);
    CHECK(read_all_shards(g_shard_dir, threaded.workers) == 0);
    CHECK(g_ngames == rs.budget);
    for (int i = 0; i < g_ngames; i++) h3[g_games[i].gi] = parsed_hash(&g_games[i]);

    for (int gi = 0; gi < rs.budget; gi++) {
        CHECK(h1[gi] == h2[gi]);   /* interleave reproducible            */
        CHECK(h1[gi] == h3[gi]);   /* == threaded                       */
    }
}

static void test_record_invariants(void) {
    RunShape rs = { .workers = 2, .budget = 6, .buffer = 8,
                    .max_plies = 12, .interleave = 0, .seed = 321 };
    BatchCtx fx = { .white_wins = 0 };
    run_engine(rs, &fx, g_shard_dir);
    CHECK(read_all_shards(g_shard_dir, rs.workers) == 0);
    CHECK(g_ngames == rs.budget);
    int failures = 0;
    for (int i = 0; i < g_ngames; i++) {
        const TGame *t = &g_games[i];
        for (uint32_t p = 0; p < t->n_plies; p++) {
            const TPly *ply = &t->plies[p];
            if (ply->n_pi < 1 || ply->n_pi > 16) { failures++; continue; }
            double total = 0.0;
            int played_in = 0;
            for (int j = 0; j < ply->n_pi; j++) {
                double prob = ply->pi[j].prob;
                if (prob < 0.0) failures++;
                total += prob;
                if (ply->pi[j].slot == (uint16_t)ply->played) played_in = 1;
            }
            if (!(fabs(total - 1.0) < 1e-6) || !played_in
                || ply->mover != p % 2)
                failures++;
        }
    }
    CHECK(failures == 0);
}

/* THE decoupling gate (deviation 10): budget 300 > the single worker's
 * 128 pool slots — the run can only finish if finished games dump their
 * records and return their slots at collect time. Short games keep it
 * fast. */
static void test_budget_exceeds_pool(void) {
    RunShape rs = { .workers = 1, .budget = 300, .buffer = 8,
                    .max_plies = 2, .interleave = 0, .seed = 4242 };
    BatchCtx bctx = { .white_wins = 0 };
    run_engine(rs, &bctx, g_shard_dir);
    CHECK(check_shards_vs_sequential(g_shard_dir, rs, &bctx) == 0);
}

/* THE batch-fill gate (deviation 12, corrected 2026-09-29): per-game
 * multi-park prefetching — a game contributes its whole WAVE to the epoch
 * (one pending descent per surviving candidate, |A| up to m=8 here), not
 * one request. Few warm games must still fill a big batch: that is what
 * decouples the GPU batch size from the in-flight game count (memory).
 * Empirically this config yields max_batch = budget × 8 (the round-1
 * wave); the check keeps a 2× safety margin while still proving
 * ≥2 parks per game per epoch. */
static void test_few_games_fill_the_batch(void) {
    RunShape rs = { .workers = 1, .budget = 8, .buffer = 256,
                    .max_plies = 40, .interleave = 0, .seed = 511 };
    BatchCtx fx = { .white_wins = 0 };
    int max_batch = run_engine(rs, &fx, NULL);
    CHECK(max_batch >= 2 * rs.budget);
}

static int failing_batch(void *ctx, SpEvalBatch *b) {
    (void)ctx; (void)b;
    return 1;   /* the net "crashed" */
}

static void test_evaluator_failure_fails_run(void) {
    /* A failing evaluator must fail the run cleanly: workers wake, wait
     * returns, the error flag is set — the cloud-instances-are-unreliable
     * rule from AGENTS.md. No record_dir: also exercises the dump-off
     * path. */
    SpEngineConfig cfg;
    sp_engine_config_init(&cfg);
    cfg.game_budget = 4;
    cfg.n_workers = 2;
    cfg.buffer_requests = 8;
    cfg.max_plies = 4;
    SpEvaluator ev = { .ctx = NULL, .init = NULL,
                       .batch = failing_batch, .free_fn = NULL };
    void *engine_storage = reserve_static(SP_ENGINE_STATIC_SIZE);
    SpEngine *e = sp_engine_init(engine_storage, SP_ENGINE_STATIC_SIZE,
                                 &cfg, &ev);
    CHECK(e != NULL);
    CHECK(sp_engine_start(e) == 0);
    sp_engine_wait(e);
    CHECK(sp_engine_error(e) == 1);
    sp_engine_destroy(e);
}

int main(void) {
    chess_init();

    /* One shared shard dir (files are truncated per run); cleaned up on
     * exit. mkdtemp avoids collisions between concurrent test binaries. */
    char dir_template[] = "/tmp/test_sp_shards.XXXXXX";
    g_shard_dir = mkdtemp(dir_template);
    if (!g_shard_dir) {
        printf("test_sp: mkdtemp failed\n");
        return 1;
    }

    test_parallel_matches_sequential();
    test_parallel_resign_path();
    test_interleave_is_bit_reproducible();
    test_record_invariants();
    test_budget_exceeds_pool();
    test_few_games_fill_the_batch();
    test_evaluator_failure_fails_run();

    /* Cleanup: unlink the shards, rmdir. (Dumps are small here; on
     * failure the dir is left for inspection.) */
    if (g_fail == 0) {
        for (int w = 0; w < SP_ENGINE_MAX_WORKERS; w++) {
            char path[SP_DUMP_PATH_MAX + 32];
            sp_dump_shard_path(g_shard_dir, w, path, sizeof path);
            unlink(path);
        }
        rmdir(g_shard_dir);
    }

    printf("test_sp: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail != 0;
}
