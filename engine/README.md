# engine/ — the C play path

The performance-critical half of the system (same split as Lc0/KataGo): a shared library with **no mutable globals** (hundreds of game instances per process) and **zero dynamic memory** after init — fixed-size `State` structs, stack move lists, a fixed undo stack inside each State. This design removes most C pitfalls up front and is what lets one Python process feed a batched GPU.

## Status (Phase 1 done: rules, termination, FFI; 1.6 perf pass done)

- Bitboard board representation (`State`: mailbox + per-piece bitboards kept in sync), make/unmake with a fixed undo stack, FEN parse/serialize
- **Strictly-legal move generation** (subtask 1.6): checkers, pin rays and the enemy danger map are computed once per node and only legal moves are emitted — no make/unmake legality filter. Magic bitboards for sliders (portable plain-C: no PEXT/CRC/asm, identical fast path on arm64 and x86_64), tables generated + exhaustively self-verified at every `chess_init()`. En passant verified against the post-capture occupancy (covers the ep-pin cases). The old pseudo-legal generator is kept as a test oracle and cross-checked against the strictly-legal one in the unit tests (1.8M checks)
- Termination (Phase 1.3): checkmate, stalemate, insufficient material, 50/75-move clocks, threefold/fivefold repetition — semantics byte-exact with python-chess `outcome(claim_draw=True)` (the fuzz oracle), including the FIDE en-passant refinement for "same position" and the claim rules' one-move lookahead. Repetition uses an incrementally-maintained Zobrist key history in `State` (also the future transposition-table hash)
- FFI handle API (Phase 0.3): `include/chess_ffi.h` — opaque per-game handles, UCI in/out, packed legal-move buffers, termination and perft through the boundary; consumed by `chess_ai/engine.py` (cffi)
- Perft library + CLI; verified exact against the standard fixture suite (CLI *and* through the FFI)

## Layout

- `include/chess.h` — public types & API (start here; the file header explains the design)
- `include/chess_ffi.h` — the C-ABI surface for foreign runtimes (Python/cffi)
- `src/tables.c` — write-once attack/castling/Zobrist tables + magic bitboards (`chess_init()`)
- `src/board.c` — attack detection, make/unmake (+ incremental position key)
- `src/fen.c` — FEN parse/serialize
- `src/movegen.c` — strictly-legal generation (checkers/pins/danger map); the pseudo-legal generator is kept as a test oracle
- `src/termination.c` — game-over detection (mate, stalemate, material, clocks, repetition)
- `src/ffi.c` — implementation of the chess_ffi.h handle API
- `src/perftlib.c` — `perft()` / `perft_divide()` (in the shared lib for the FFI gate)
- `src/perft.c` — the CLI (below)
- `tests/` — C unit-test binary, perft suite data file (`perft_suite.txt`), suite loader
- `Makefile`

## Build & verify

```sh
make            # build libchess.(dylib|so), perft CLI, test binary (into build/)
make check      # run C unit tests + full perft suite (the Phase 1 gate, ~3 s)
make check-asan # same under AddressSanitizer + UBSan (~20 s)
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

## Performance (perft, 2026-09-23 after subtask 1.6, Apple M4, single thread, `-O2`)

All counts verified exact; "bulk counting" = leaves counted without making
the last ply's moves (same convention as the reference numbers below — no
hash tables anywhere).

| Position | Depth | Nodes | Time | Mnps |
|---|---|---|---|---|
| startpos | 6 | 119,060,324 | 0.22 s | 547 |
| startpos | 7 | 3,195,901,860 | 5.66 s | 565 |
| Kiwipete | 5 | 193,690,690 | 0.28 s | 702 |
| Kiwipete | 6 | 8,031,647,685 | 12.95 s | 620 |
| pos6 | 5 | 164,075,551 | 0.24 s | 679 |
| pos3 | 6 | 11,030,083 | 0.03 s | 422 |

Reference points, startpos perft 5, i7-6850K, hashless + bulk counting:
Stockfish-8 ≈ 147, shakmaty ≈ 202, jordanbray/chess ≈ 262 Mnps. This
engine's 1.6 baseline was 57–67 Mnps on the M4; the two steps of the
optimization (commits f7aa56c, e0d337a) took it to **422–702 Mnps**
(7–12× by position) — several times the fastest of those references
even granting the M4 a per-clock advantage:

1. **Magic bitboards** (~1.4×): slider attacks via one
   mask-multiply-shift-load instead of 4-ray scanning. Portable C11 by
   design — no PEXT/CRC/inline-asm — so the same code is the fast path on
   arm64 and x86_64 cloud instances. Constants are baked in and every
   table is re-verified exhaustively against the ray scanner at each
   `chess_init()`.
2. **Strictly-legal generation** (~6×): checkers, pin rays and the enemy
   danger map (occupancy minus our king) computed once per node; only
   legal moves are emitted; no make/unmake legality filter. En passant —
   the only move where two pieces leave the board at once — is verified
   explicitly against the post-capture occupancy.

(A third step — a count-only generator for the bulk-counted last ply,
reaching ~0.9–1.7 Gnps — was implemented, then deliberately removed: it
only sped up the perft benchmark, not any engine path, and duplicated
the generation logic. The numbers above are the engine-relevant ones.)

Measured with `-O3`, results were slightly *worse* than `-O2` (as before).
ASan+UBSan costs ~4× on the suite (was ~3.1× pre-1.6). Verification
after every step: perft suite exact (53 fixtures), startpos d7 exact on
CLI **and** FFI legs, 1.8M C unit checks (strict generator vs the old
make/unmake filter over hundreds of random games, magic vs ray scanning),
and the python-chess oracle fuzz (incl. a 1000-game run) — all green.

## What is still ahead

- Full-scale (≥10⁶-walk) python-chess oracle fuzz as a CI job
- Input-plane encoder, MCTS (Phases 2–3)
