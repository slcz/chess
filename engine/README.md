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

## What is still ahead (Phase 1.3+)

- Termination rules: 50-move rule, threefold repetition, insufficient material
- Full-scale (≥10⁶-walk) python-chess oracle fuzz as a CI job
- Input-plane encoder, MCTS (Phases 2–3)
