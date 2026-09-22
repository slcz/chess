# engine/ — the C play path

The performance-critical half of the system (same split as Lc0/KataGo): a shared library with **no mutable globals** (hundreds of game instances per process) and **zero dynamic memory** after init — fixed-size `State` structs, stack move lists, a fixed undo stack inside each State. This design removes most C pitfalls up front and is what lets one Python process feed a batched GPU.

## Status (Phase 1.1–1.2 done)

- Bitboard board representation (`State`: mailbox + per-piece bitboards kept in sync), make/unmake with a fixed undo stack, FEN parse/serialize
- Full pseudo-legal movegen + legality filter: castling (incl. through check), en passant (incl. ep pins), promotions, check detection; mate/stalemate fall out of the filter (empty legal list + in-check flag)
- Perft library + CLI; verified exact against the standard fixture suite

## Layout

- `include/chess.h` — public types & API (start here; the file header explains the design)
- `src/tables.c` — write-once attack/castling lookup tables (`chess_init()`)
- `src/board.c` — attack detection, make/unmake
- `src/fen.c` — FEN parse/serialize
- `src/movegen.c` — pseudo-legal generation + legality filter
- `src/perftlib.c` — `perft()` / `perft_divide()` (in the shared lib for the future FFI gate)
- `src/perft.c` — the CLI (below)
- `tests/` — C unit-test binary, perft suite data file (`perft_suite.txt`), suite loader
- `Makefile`

## Build & verify

```sh
make            # build libchess.(dylib|so), perft CLI, test binary (into build/)
make check      # run C unit tests + full perft suite (the Phase 1 gate, ~15 s)
make check-asan # same under AddressSanitizer + UBSan (~1 min)
```

CLI:

```sh
build/perft "<fen>" <depth>      # node count + Mnps
build/perft divide "<fen>" <d>   # per-root-move counts (bisect a mismatch)
build/perft legal "<fen>"        # legal moves as UCI strings
build/perft suite                # every line of tests/perft_suite.txt
build/perft batch-legal          # REPL: FEN per line on stdin → "<fen> <moves…>"
```

`batch-legal` is the oracle-fuzzing interface used by `tests/test_engine.py`.

## Performance (perft, 2026-09-22, Apple M4, single thread, `-O2`)

All counts verified exact; "bulk counting" = leaves counted without making the
last ply's moves (same convention as the reference numbers below).

| Position | Depth | Nodes | Time | Mnps |
|---|---|---|---|---|
| startpos | 6 | 119,060,324 | 1.80 s | 66 |
| startpos | 7 | 3,195,901,860 | 51.3 s | 62 |
| Kiwipete | 5 | 193,690,690 | 3.27 s | 59 |
| pos6 | 5 | 164,075,551 | 2.31 s | 71 |

Reference points: Stockfish 8 / shakmaty / jordanbray-chess measured ~147 /
~202 / ~262 Mnps on startpos perft 5 (i7-6850K, shakmaty README, no
hash tables); typical hobby engines 1–20 Mnps (TalkChess). Raw make/unmake
throughput ≈ 100M pairs/s. `-O3 -mcpu=native` gave no gain over `-O2`;
ASan+UBSan costs ~3.1×. Known headroom if Phase 3 profiling ever needs it:
pin-aware strictly-legal generation (skip the per-move make/unmake filter,
~1.5–3×) and magic bitboards for sliders (~1.2–2×).

## What is still ahead (Phase 1.3+)

- Termination rules: 50-move rule, threefold repetition, insufficient material
- Full-scale (≥10⁶-walk) python-chess oracle fuzz as a CI job
- Input-plane encoder, MCTS (Phases 2–3)
