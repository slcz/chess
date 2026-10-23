/* parity_dump.c — canonical search transcript for the wasm parity gate.
 *
 * Built natively (make build/parity_dump) AND re-driven move-for-move
 * through the Emscripten module by web/test/parity_node.mjs; the two
 * transcripts must be BIT-IDENTICAL. That works because the default
 * (dummy) evaluator is deterministic, walkers run on the calling thread,
 * and all search math is IEEE doubles — the io thread's timing can hurry
 * or delay deliveries but never changes which rows land in which bank.
 *
 * Determinism notes baked into the schedule:
 *   - every think is followed by a play (the done-guard forbids re-walking
 *     an unplayed root), which also exercises the tree-reuse path — unless
 *     the sampling ended the game (mate/draw), when the schedule stops:
 *     a finished position has no move to play;
 *   - doubles print as their IEEE754 bit pattern so libc-vs-JS number
 *     formatting cannot fake agreement;
 *   - root children print in generation order (movegen order — the same C
 *     code on both sides).
 *
 * Usage: parity_dump            print the transcript
 *        parity_dump > file     compared with the Node side by the gate.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "chess.h"
#include "chess_puct.h"

static const char *FENS[] = {
    NULL,   /* the standard start position */
    "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq g3 0 2",
    "7k/8/8/8/8/8/R7/1R5K w - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
};
static const int SIMS[] = {16, 64, 257};
enum { BLK = 4u << 20 };   /* small node block: exercises compaction too */

static const char *dbits(double d) {
    static char b[18];
    uint64_t u;
    memcpy(&u, &d, 8);
    snprintf(b, sizeof b, "%016llx", (unsigned long long)u);
    return b;
}

static void dump_root(const char *tag, int sims, PuctGame *g) {
    uint16_t mv[MAX_MOVES];
    uint32_t vis[MAX_MOVES];
    double q[MAX_MOVES];
    char u[8];
    int n = puct_root_children(g, mv, vis, q, NULL);
    move_to_uci(puct_best_move(g), u);
    uint32_t rn;
    double rs;
    puct_root_stats(g, &rn, &rs);
    printf("%s sims=%d best=%s rootN=%u rootW=%s\n", tag, sims, u, rn,
           dbits(rs));
    for (int i = 0; i < n; i++) {
        move_to_uci(mv[i], u);
        printf("C %s n=%u q=%s\n", u, vis[i], dbits(q[i]));
    }
}

int main(void) {
    chess_init();
    puct_zone_init(8);
    for (size_t f = 0; f < sizeof FENS / sizeof *FENS; f++) {
        PuctGame *g = puct_game_new(BLK);
        if (FENS[f])
            if (puct_game_set_fen(g, FENS[f]) != 0)
                return 1;
        for (size_t s = 0; s < sizeof SIMS / sizeof *SIMS; s++) {
            puct_think(g, SIMS[s]);
            dump_root("T", SIMS[s], g);
            if (puct_game_over(g))
                break;   /* the schedule's sampling found mate/draw: there
                            is no move to play (best would be 0) */
            puct_play(g, puct_best_move(g));
        }
        if (!puct_game_over(g)) {
            puct_think(g, 64);
            dump_root("P", 64, g);
        }
        puct_game_free(g);
    }
    printf("DONE\n");
    return 0;
}
