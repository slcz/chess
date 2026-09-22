/*
 * perft.c — perft functions + a small CLI around them (Phase 1.4 vehicle).
 *
 * The CLI is the standard tool for verifying the move generator against
 * published node counts, and for debugging a mismatch (divide breaks the
 * count down per root move so it can be bisected against another engine).
 *
 *   perft "<fen>" <depth>       node count + timing
 *   perft divide "<fen>" <d>    one line per root move, with subtree counts
 *   perft suite                 run every fixture in perft_fixtures.h
 *   perft legal "<fen>"         print the legal moves as UCI strings
 *   perft batch-legal           REPL for oracle fuzzing: read a FEN per line
 *                               on stdin, print "<fen> <moves...>" per line
 *                               (the FEN is echoed after a serialize/parse
 *                               round-trip, so the caller checks both)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "chess.h"
#include "suite_loader.h"

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

static double now_seconds(void) {
    return (double)clock() / CLOCKS_PER_SEC;
}

static int run_suite(void) {
    SuiteEntry entries[SUITE_MAX_ENTRIES];
    int n = suite_load(entries, SUITE_MAX_ENTRIES);
    if (n < 0) return 1;

    int failures = 0;
    double t0 = now_seconds();
    uint64_t checked = 0;

    for (int i = 0; i < n; i++) {
        const SuiteEntry *e = &entries[i];
        State s;
        if (fen_parse(&s, e->fen) != 0) {
            printf("FAIL %-17s FEN parse error\n", e->name);
            failures++;
            continue;
        }
        uint64_t got = perft(&s, e->depth);
        checked += got;
        if (got != e->nodes) {
            printf("FAIL %-17s d%d: got %llu, want %llu\n", e->name, e->depth,
                   (unsigned long long)got, (unsigned long long)e->nodes);
            failures++;
        }
    }

    printf("%s: %d suite lines, %llu nodes checked in %.2fs\n",
           failures ? "FAILURES" : "ALL OK", n, (unsigned long long)checked,
           now_seconds() - t0);
    return failures;
}

static void print_legal(State *s) {
    MoveList ml;
    gen_legal(s, &ml);
    for (int i = 0; i < ml.count; i++) {
        char uci[8];
        move_to_uci(ml.moves[i], uci);
        printf("%s ", uci);
    }
    printf("\n");
}

static int batch_legal(void) {
    char line[512];
    State s;
    while (fgets(line, sizeof line, stdin)) {
        line[strcspn(line, "\n")] = '\0';
        if (fen_parse(&s, line) != 0) {
            printf("ERR\n");
            continue;
        }
        char fen[96];
        fen_serialize(&s, fen, sizeof fen);
        printf("%s", fen);
        MoveList ml;
        gen_legal(&s, &ml);
        for (int i = 0; i < ml.count; i++) {
            char uci[8];
            move_to_uci(ml.moves[i], uci);
            printf(" %s", uci);
        }
        printf("\n");
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char **argv) {
    chess_init();

    if (argc >= 2 && !strcmp(argv[1], "suite")) {
        return run_suite() ? 1 : 0;
    }
    if (argc >= 2 && !strcmp(argv[1], "batch-legal")) {
        return batch_legal();
    }
    if (argc == 4 && !strcmp(argv[1], "divide")) {
        State s;
        if (fen_parse(&s, argv[2]) != 0) { fprintf(stderr, "bad FEN\n"); return 1; }
        perft_divide(&s, atoi(argv[3]));
        return 0;
    }
    if (argc == 3 && !strcmp(argv[1], "legal")) {
        State s;
        if (fen_parse(&s, argv[2]) != 0) { fprintf(stderr, "bad FEN\n"); return 1; }
        print_legal(&s);
        return 0;
    }
    if (argc == 3) {
        State s;
        if (fen_parse(&s, argv[1]) != 0) { fprintf(stderr, "bad FEN\n"); return 1; }
        int depth = atoi(argv[2]);
        double t0 = now_seconds();
        uint64_t n = perft(&s, depth);
        double dt = now_seconds() - t0;
        printf("perft(%d) = %llu  (%.2fs, %.1f Mnps)\n", depth,
               (unsigned long long)n, dt, dt > 0 ? n / dt / 1e6 : 0.0);
        return 0;
    }

    fprintf(stderr,
            "usage: %s \"<fen>\" <depth> | divide \"<fen>\" <d> | legal \"<fen>\" "
            "| suite | batch-legal < fenfile\n", argv[0]);
    return 2;
}
