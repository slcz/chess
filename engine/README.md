# engine/ — the C play path

The performance-critical half of the system (same split as Lc0/KataGo): a shared library with **no mutable globals** (hundreds of game instances per process) and **zero dynamic memory** after init — fixed-size `State` structs, stack move lists, a fixed undo stack inside each State. This design removes most C pitfalls up front and is what lets one Python process feed a batched GPU.

## Status (Phase 1 done: rules, termination, FFI)

- Bitboard board representation (`State`: mailbox + per-piece bitboards kept in sync), make/unmake with a fixed undo stack, FEN parse/serialize
- Full pseudo-legal movegen + legality filter: castling (incl. through check), en passant (incl. ep pins), promotions, check detection
- Termination (Phase 1.3): checkmate, stalemate, insufficient material, 50/75-move clocks, threefold/fivefold repetition — semantics byte-exact with python-chess `outcome(claim_draw=True)` (the fuzz oracle), including the FIDE en-passant refinement for "same position" and the claim rules' one-move lookahead. Repetition uses an incrementally-maintained Zobrist key history in `State` (also the future transposition-table hash)
- FFI handle API (Phase 0.3): `include/chess_ffi.h` — opaque per-game handles, UCI in/out, packed legal-move buffers, termination and perft through the boundary; consumed by `chess_ai/engine.py` (cffi)
- Perft library + CLI; verified exact against the standard fixture suite (CLI *and* through the FFI)

## Layout

- `include/chess.h` — public types & API (start here; the file header explains the design)
- `include/chess_ffi.h` — the C-ABI surface for foreign runtimes (Python/cffi)
- `src/tables.c` — write-once attack/castling/Zobrist tables (`chess_init()`)
- `src/board.c` — attack detection, make/unmake (+ incremental position key)
- `src/fen.c` — FEN parse/serialize
- `src/movegen.c` — pseudo-legal generation + legality filter
- `src/termination.c` — game-over detection (mate, stalemate, material, clocks, repetition)
- `src/ffi.c` — implementation of the chess_ffi.h handle API
- `src/perftlib.c` — `perft()` / `perft_divide()` (in the shared lib for the FFI gate)
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

`batch-legal` is a legacy oracle-fuzzing interface; the fuzz in `tests/test_engine.py` now goes through the FFI (`chess_ai.engine`).

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

## What is still ahead

- Full-scale (≥10⁶-walk) python-chess oracle fuzz as a CI job
- Input-plane encoder, MCTS (Phases 2–3)
