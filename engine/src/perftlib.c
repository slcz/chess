/*
 * perftlib.c — the perft function itself, part of the shared library.
 *
 * Kept out of perft.c (the CLI) so the test binary — and later the FFI
 * bindings (Phase 0.3/1.5 gate: "perft via FFI") — can link it without a
 * main() clash.
 */
#include <stdio.h>

#include "chess.h"

uint64_t perft(State *s, int depth) {
    if (depth == 0) return 1;

    /* Bulk counting: at the last ply the legal-move count IS the leaf
     * count, so we skip making the moves — and use the count-only
     * generator, which never materializes the move list at all. Standard
     * perft optimization; changes nothing about the numbers. */
    if (depth == 1) return (uint64_t)gen_legal_count(s);

    MoveList ml;
    gen_legal(s, &ml);

    uint64_t nodes = 0;
    for (int i = 0; i < ml.count; i++) {
        make_move(s, ml.moves[i]);
        nodes += perft(s, depth - 1);
        unmake_move(s, ml.moves[i]);
    }
    return nodes;
}

void perft_divide(State *s, int depth) {
    MoveList ml;
    gen_legal(s, &ml);

    uint64_t total = 0;
    for (int i = 0; i < ml.count; i++) {
        char uci[8];
        move_to_uci(ml.moves[i], uci);
        make_move(s, ml.moves[i]);
        uint64_t n = depth <= 1 ? 1 : perft(s, depth - 1);
        unmake_move(s, ml.moves[i]);
        printf("%s: %llu\n", uci, (unsigned long long)n);
        total += n;
    }
    printf("total: %llu\n", (unsigned long long)total);
}
