/*
 * suite_loader.h — loads engine/tests/perft_suite.txt for the test binary
 * and the perft CLI (both share the same data file).
 *
 * The suite file is the human-editable source of truth; the Makefile bakes
 * its absolute path into the binaries via -DPERFT_SUITE_PATH so tests work
 * from any working directory.
 */
#ifndef SUITE_LOADER_H
#define SUITE_LOADER_H

#include <stdint.h>

#define SUITE_MAX_ENTRIES 128

typedef struct {
    char     name[40];
    char     fen[128];
    int      depth;
    uint64_t nodes;
} SuiteEntry;

/* Returns the number of entries loaded, or -1 if the file cannot be found
 * (the paths tried are printed to stderr). Caller provides the array. */
int suite_load(SuiteEntry out[], int max);

#endif /* SUITE_LOADER_H */
