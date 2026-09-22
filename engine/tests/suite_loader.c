/*
 * suite_loader.c — tiny parser for perft_suite.txt.
 *
 * Line format (see the file header):  name | FEN | depth | node-count
 * '#' comments and blank lines are skipped; tokens are trimmed. Strictly
 * hand-rolled: the project is dependency-free by design, and the grammar is
 * four pipe-separated fields.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "suite_loader.h"

#ifndef PERFT_SUITE_PATH
#define PERFT_SUITE_PATH "tests/perft_suite.txt"
#endif

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    char *end = s + strlen(s);
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' || end[-1] == '\r'))
        *--end = '\0';
    return s;
}

int suite_load(SuiteEntry out[], int max) {
    static const char *paths[] = {
        PERFT_SUITE_PATH,
        "tests/perft_suite.txt",       /* running from engine/     */
        "../tests/perft_suite.txt",    /* running from engine/build */
        "perft_suite.txt",
    };

    FILE *f = NULL;
    for (size_t i = 0; i < sizeof paths / sizeof paths[0]; i++) {
        if ((f = fopen(paths[i], "r")) != NULL) break;
    }
    if (!f) {
        fprintf(stderr, "suite_load: cannot find perft_suite.txt (tried");
        for (size_t i = 0; i < sizeof paths / sizeof paths[0]; i++)
            fprintf(stderr, " %s", paths[i]);
        fprintf(stderr, ")\n");
        return -1;
    }

    char line[1024];
    int n = 0, lineno = 0;
    while (fgets(line, sizeof line, f)) {
        lineno++;

        char *hash = strchr(line, '#');
        if (hash) *hash = '\0';            /* strip trailing comments */

        char *fields[4] = {0};
        int nf = 0;
        char *p = line;
        while (nf < 4) {
            char *bar = strchr(p, '|');
            if (bar) *bar = '\0';
            char *tok = trim(p);
            if (*tok) {
                if (nf == 4) break;
                fields[nf++] = tok;
            }
            if (!bar) break;
            p = bar + 1;
        }

        if (nf == 0) continue;             /* blank/comment line */
        if (nf != 4) {
            fprintf(stderr, "suite_load: %s:%d: expected 4 fields, got %d\n",
                    PERFT_SUITE_PATH, lineno, nf);
            fclose(f);
            return -1;
        }
        if (n >= max) {
            fprintf(stderr, "suite_load: more than %d entries\n", max);
            fclose(f);
            return -1;
        }

        snprintf(out[n].name, sizeof out[n].name, "%s", fields[0]);
        snprintf(out[n].fen, sizeof out[n].fen, "%s", fields[1]);
        out[n].depth = atoi(fields[2]);
        out[n].nodes = strtoull(fields[3], NULL, 10);
        if (out[n].depth < 1) {
            fprintf(stderr, "suite_load: %s:%d: bad depth '%s'\n",
                    PERFT_SUITE_PATH, lineno, fields[2]);
            fclose(f);
            return -1;
        }
        n++;
    }
    fclose(f);
    return n;
}
