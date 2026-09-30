/*
 * chess_selfplay_main.c — the standalone self-play work-unit runner
 * (Phase 3.4: "a chess_selfplay CLI binary for the salad nodes:
 * Py_Initialize + engine, work-unit sharding/checkpointing per the cloud
 * rules" — PLAN §0).
 *
 * One process plays ONE work unit: `--seed S --games N --out DIR`. The
 * engine (libchess_sp sources, linked statically here) runs its usual
 * parallel run — N C worker threads, THIS main thread playing the
 * evaluator between epochs — and finished games land in DIR's per-worker
 * shard files (chess_sp_dump.h). On success the process writes
 * DIR/DONE.json (atomic tmp+rename) carrying the manifest the trainer's
 * reader needs (unit_seed, n_workers, games_done) plus telemetry.
 *
 * Cloud-rules semantics (PLAN §5 — instances vanish at any moment):
 *   - Idempotent re-issue: DONE.json present ⇒ print and exit 0 without
 *     touching the shards.
 *   - A kill mid-run leaves partial shards and NO DONE.json; re-running
 *     the same command re-plays the unit deterministically into the
 *     O_TRUNC'd shards (same seed ⇒ same games — the engine's
 *     determinism gate). The lost work is bounded by one unit's in-flight
 *     games, which is exactly the "≤ one work unit lost" rule. (Finer
 *     resume — game-index offsets, shard append — stays a Phase 4.2
 *     refinement; units are sized so this coarse rule is cheap.)
 *
 * The Python side (the evaluator module) is imported through the
 * embedded interpreter: --pythonpath (or PYTHONPATH) must expose both the
 * repo root (chess_ai...) and the venv's site-packages (numpy/torch).
 * Only the evaluator thread ever holds the GIL, only around eval_batch
 * (PARALLEL_DESIGN F8); the GIL is released across sp_engine_wait so the
 * interleaving matches the Python-hosted path.
 *
 * Build: `make -C engine build/chess_selfplay` (links the interpreter's
 * libpython via sysconfig; the _sp LIBRARY stays dlopen-only).
 */
#include <errno.h>
#include <fcntl.h>
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "chess.h"
#include "chess_sp.h"

#define DONE_NAME "DONE.json"

/* Diverse game init: --start-fens FILE lists one FEN per line; game i
 * starts from pool[i % n] (sp_engine.c). Storage is static — the engine
 * copies only the config struct, so the strings must outlive the run. */
#define MAX_START_FENS 1024
static char     fen_pool[MAX_START_FENS][CHESS_FEN_MAX];
static const char *fen_ptrs[MAX_START_FENS];

typedef struct {
    uint64_t seed;
    long     games;
    char     out[512];
    const char *module;
    const char *spec;
    const char *pythonpath;
    const char *start_fens_file;
    long     workers, buffer, n, m_cap, sample_plies, max_plies;
    double   c_visit, c_scale, c_fpu, resign;
    int      interleave;
} Args;

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s --seed U64 --games N --out DIR [options]\n"
        "  one self-play work unit: play N games seeded from the unit seed,\n"
        "  dump per-worker shards into DIR, write DIR/%s on completion.\n"
        "options (defaults = the engine's reference config):\n"
        "  --pythonpath P  colon-separated sys.path prefixes (repo root and\n"
        "                  the venv's site-packages; PYTHONPATH also works)\n"
        "  --module M      evaluator module  [%s]\n"
        "  --spec S        evaluator spec    [fixture]  e.g. torch:ckpt=net.pt\n"
        "  --workers N     worker threads    [8]        --buffer N  [256]\n"
        "  --n N           sims per move     [8]        --m-cap N   [16]\n"
        "  --c-visit F [50]  --c-scale F [1]  --c-fpu F [0]  --resign F [-0.9]\n"
        "  --sample-plies N  [-1 = sample every ply (training default)]\n"
        "  --max-plies N     [-1 = unbounded]\n"
        "  --start-fens FILE  one FEN per line; game i starts from line\n"
        "                  (i mod #lines) [default: standard start position]\n"
        "  --interleave      deterministic single-thread mode (debug)\n",
        prog, DONE_NAME, "chess_ai.selfplay.sp_evaluator");
}

static int parse_long(const char *s, long *out) {
    char *end;
    errno = 0;
    *out = strtol(s, &end, 10);
    return errno == 0 && *end == '\0' ? 0 : -1;
}

static int parse_double(const char *s, double *out) {
    char *end;
    errno = 0;
    *out = strtod(s, &end);
    return errno == 0 && *end == '\0' ? 0 : -1;
}

static int parse_args(int argc, char **argv, Args *a) {
    memset(a, 0, sizeof *a);
    a->module = "chess_ai.selfplay.sp_evaluator";
    a->spec = "fixture";
    a->pythonpath = "";
    a->workers = 8; a->buffer = 256; a->n = 8; a->m_cap = 16;
    a->sample_plies = -1; a->max_plies = -1;
    a->c_visit = 50.0; a->c_scale = 1.0; a->c_fpu = 0.0; a->resign = -0.90;
    for (int i = 1; i < argc; i++) {
        const char *v = (i + 1 < argc) ? argv[i + 1] : "";
        if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            char *end; errno = 0;
            a->seed = strtoull(v, &end, 10);
            i++;
            if (errno || *end) return -1;
        } else if (!strcmp(argv[i], "--games") && i + 1 < argc) {
            if (parse_long(v, &a->games)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            snprintf(a->out, sizeof a->out, "%s", v);
            i++;
        } else if (!strcmp(argv[i], "--module") && i + 1 < argc) {
            a->module = v; i++;
        } else if (!strcmp(argv[i], "--spec") && i + 1 < argc) {
            a->spec = v; i++;
        } else if (!strcmp(argv[i], "--pythonpath") && i + 1 < argc) {
            a->pythonpath = v; i++;
        } else if (!strcmp(argv[i], "--workers") && i + 1 < argc) {
            if (parse_long(v, &a->workers)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--buffer") && i + 1 < argc) {
            if (parse_long(v, &a->buffer)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--n") && i + 1 < argc) {
            if (parse_long(v, &a->n)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--m-cap") && i + 1 < argc) {
            if (parse_long(v, &a->m_cap)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--c-visit") && i + 1 < argc) {
            if (parse_double(v, &a->c_visit)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--c-scale") && i + 1 < argc) {
            if (parse_double(v, &a->c_scale)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--c-fpu") && i + 1 < argc) {
            if (parse_double(v, &a->c_fpu)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--resign") && i + 1 < argc) {
            if (parse_double(v, &a->resign)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--sample-plies") && i + 1 < argc) {
            if (parse_long(v, &a->sample_plies)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--max-plies") && i + 1 < argc) {
            if (parse_long(v, &a->max_plies)) return -1;
            i++;
        } else if (!strcmp(argv[i], "--start-fens") && i + 1 < argc) {
            a->start_fens_file = v;
            i++;
        } else if (!strcmp(argv[i], "--interleave")) {
            a->interleave = 1;
        } else {
            return -1;
        }
    }
    if (!a->out[0] || a->games < 0)
        return -1;
    return 0;
}

/* mkdir -p for the shard dir (the engine opens DIR/shard-w*.spbin with
 * O_CREAT but does not create DIR itself). */
static int mkdirs(const char *path) {
    char buf[512];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(buf, 0755) != 0 && errno != EEXIST)
                return -1;
            *p = '/';
        }
    }
    return mkdir(buf, 0755) != 0 && errno != EEXIST ? -1 : 0;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Minimal JSON string escaping (quotes, backslash, control chars) — the
 * only free-form field in the manifest is the eval spec. */
static void json_escape(const char *in, char *out, size_t cap) {
    size_t o = 0;
    for (const char *p = in; *p && o + 7 < cap; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') {
            out[o++] = '\\'; out[o++] = (char)c;
        } else if (c < 0x20) {
            o += (size_t)snprintf(out + o, cap - o, "\\u%04x", c);
        } else {
            out[o++] = (char)c;
        }
    }
    out[o] = '\0';
}

static int write_done(const char *dir, const Args *a, const SpEngine *e,
                      double elapsed) {
    char path[600], tmp[620], spec_esc[1024];
    snprintf(path, sizeof path, "%s/%s", dir, DONE_NAME);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    json_escape(a->spec, spec_esc, sizeof spec_esc);
    char body[2048];
    int n = snprintf(body, sizeof body,
        "{\n"
        "  \"format\": \"SPSHARD1\", \"version\": 1,\n"
        "  \"unit_seed\": %llu, \"game_budget\": %ld, \"n_workers\": %ld,\n"
        "  \"games_done\": %d, \"evals\": %ld, \"epochs\": %ld, \"max_batch\": %d,\n"
        "  \"elapsed_s\": %.3f, \"eval_spec\": \"%s\",\n"
        "  \"n\": %ld, \"m_cap\": %ld, \"c_visit\": %.17g, \"c_scale\": %.17g,\n"
        "  \"c_fpu\": %.17g, \"sample_plies\": %ld, \"resign_value\": %.17g,\n"
        "  \"max_plies\": %ld, \"interleave\": %d\n"
        "}\n",
        (unsigned long long)a->seed, a->games, a->workers,
        sp_engine_games_done(e), sp_engine_evals(e), sp_engine_epochs(e),
        sp_engine_max_batch(e), elapsed, spec_esc,
        a->n, a->m_cap, a->c_visit, a->c_scale,
        a->c_fpu, a->sample_plies, a->resign,
        a->max_plies, a->interleave);
    if (n < 0 || (size_t)n >= sizeof body)
        return -1;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        return -1;
    const char *p = body;
    while (*p) {
        ssize_t w = write(fd, p, strlen(p));
        if (w < 0) { close(fd); return -1; }
        p += w;
    }
    if (close(fd) != 0)
        return -1;
    return rename(tmp, path) != 0 ? -1 : 0;   /* atomic completion marker */
}

/* Prepend each ':'-separated component of `pathlist` to sys.path, back to
 * front so the given order survives the insert-at-0. */
static int prepend_sys_path(const char *pathlist) {
    if (!pathlist || !*pathlist)
        return 0;
    PyObject *sys_path = PySys_GetObject("path");   /* borrowed */
    if (!sys_path)
        return -1;
    char buf[2048];
    snprintf(buf, sizeof buf, "%s", pathlist);
    size_t len = strlen(buf);
    for (size_t end = len; end > 0; ) {
        size_t start = end;
        while (start > 0 && buf[start - 1] != ':')
            start--;
        if (start < end) {
            buf[end] = '\0';
            PyObject *s = PyUnicode_FromString(buf + start);
            if (!s)
                return -1;
            int rc = PyList_Insert(sys_path, 0, s);
            Py_DECREF(s);
            if (rc != 0)
                return -1;
        }
        end = (start > 0) ? start - 1 : 0;
    }
    return 0;
}

/* Load the start-FEN pool: one FEN per line, '#' comments and blank lines
 * skipped. Returns the pool size, or -1 on any read/format error. The
 * strings land in the static fen_pool so they outlive the run. */
static int load_start_fens(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open --start-fens %s: %s\n", path,
                strerror(errno));
        return -1;
    }
    int n = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t')
            s++;
        if (*s == '\n' || *s == '\0' || *s == '#')
            continue;
        size_t len = strlen(s);
        while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r' ||
                           s[len - 1] == ' ' || s[len - 1] == '\t'))
            s[--len] = '\0';
        if (len == 0 || *s == '#')
            continue;
        if (len >= CHESS_FEN_MAX) {
            fprintf(stderr, "--start-fens: FEN longer than %d on line %d\n",
                    CHESS_FEN_MAX - 1, n + 1);
            fclose(f);
            return -1;
        }
        if (n >= MAX_START_FENS) {
            fprintf(stderr, "--start-fens: more than %d FENs\n",
                    MAX_START_FENS);
            fclose(f);
            return -1;
        }
        snprintf(fen_pool[n], CHESS_FEN_MAX, "%s", s);
        fen_ptrs[n] = fen_pool[n];
        n++;
    }
    int read_err = ferror(f);
    fclose(f);
    if (read_err) {
        fprintf(stderr, "--start-fens: read error on %s\n", path);
        return -1;
    }
    if (n == 0) {
        fprintf(stderr, "--start-fens: no FENs in %s\n", path);
        return -1;
    }
    return n;
}

int main(int argc, char **argv) {
    Args a;
    if (parse_args(argc, argv, &a) != 0) {
        usage(argv[0]);
        return 1;
    }

    char done_path[600];
    snprintf(done_path, sizeof done_path, "%s/%s", a.out, DONE_NAME);
    struct stat st;
    if (stat(done_path, &st) == 0) {
        printf("unit complete: %s (%ld games) — nothing to do\n",
               done_path, a.games);
        return 0;
    }
    if (mkdirs(a.out) != 0) {
        fprintf(stderr, "cannot create %s: %s\n", a.out, strerror(errno));
        return 1;
    }

    int n_start_fens = 0;
    if (a.start_fens_file &&
        (n_start_fens = load_start_fens(a.start_fens_file)) < 0)
        return 1;

    Py_Initialize();
    if (prepend_sys_path(a.pythonpath) != 0) {
        PyErr_Print();
        Py_FinalizeEx();
        return 1;
    }

    chess_init();   /* magic tables — read-only after this */

    SpEngineConfig cfg;
    sp_engine_config_init(&cfg);
    cfg.game_cfg.n = (int)a.n;
    cfg.game_cfg.m_cap = (int)a.m_cap;
    cfg.game_cfg.c_visit = a.c_visit;
    cfg.game_cfg.c_scale = a.c_scale;
    cfg.game_cfg.c_fpu = a.c_fpu;
    cfg.game_cfg.sample_plies = (int)a.sample_plies;
    cfg.game_cfg.resign_value = a.resign;
    cfg.unit_seed = a.seed;
    cfg.game_budget = (int32_t)a.games;
    cfg.n_workers = (int)a.workers;
    cfg.buffer_requests = (int)a.buffer;
    cfg.max_plies = (int)a.max_plies;
    cfg.interleave = a.interleave;
    cfg.start_fens = n_start_fens > 0 ? fen_ptrs : NULL;
    cfg.n_start_fens = n_start_fens;
    if (snprintf(cfg.record_dir, sizeof cfg.record_dir, "%s", a.out)
            >= (int)sizeof cfg.record_dir) {
        fprintf(stderr, "--out path too long\n");
        Py_FinalizeEx();
        return 1;
    }

    SpEvaluator ev;
    if (sp_py_evaluator_make(a.module, a.spec, &ev) != 0) {
        fprintf(stderr, "sp_py_evaluator_make failed\n");
        Py_FinalizeEx();
        return 1;
    }

    /* The engine's static block: one anonymous mmap, lazily committed —
     * untouched pool/staging pages cost nothing (same discipline as the
     * Python seam's mmap). */
    size_t sz = sp_engine_static_size();
    void *storage = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                         MAP_ANON | MAP_PRIVATE, -1, 0);
    if (storage == MAP_FAILED) {
        fprintf(stderr, "mmap(%zu) failed: %s\n", sz, strerror(errno));
        Py_FinalizeEx();
        return 1;
    }

    SpEngine *e = sp_engine_init(storage, sz, &cfg, &ev);
    if (!e) {
        fprintf(stderr, "sp_engine_init failed\n");
        munmap(storage, sz);
        Py_FinalizeEx();
        return 1;
    }

    double t0 = now_s();
    int rc = sp_engine_start(e);
    /* wait() runs unconditionally: even a failed start may have spawned
     * threads (pthread_create failure wakes them to exit) — wait() joins
     * them, which sp_engine_destroy's assert requires. */
    PyThreadState *tstate = PyEval_SaveThread();
    sp_engine_wait(e);          /* blocks: workers + this thread evaluate */
    PyEval_RestoreThread(tstate);
    double elapsed = now_s() - t0;

    int failed = rc != 0 || sp_engine_error(e);
    if (failed)
        fprintf(stderr, "engine run failed (evaluator or dump error)\n");
    else {
        printf("unit done: %lld games, %ld evals, %ld epochs, "
               "max batch %d, %.3f s\n",
               (long long)sp_engine_games_done(e), sp_engine_evals(e),
               sp_engine_epochs(e), sp_engine_max_batch(e), elapsed);
        if (write_done(a.out, &a, e, elapsed) != 0) {
            fprintf(stderr, "writing %s failed: %s\n",
                    done_path, strerror(errno));
            failed = 1;
        }
    }

    sp_engine_destroy(e);           /* runs the evaluator's free hook (GIL ok) */
    munmap(storage, sz);
    if (Py_FinalizeEx() != 0)
        failed = 1;
    return failed ? 2 : 0;
}
