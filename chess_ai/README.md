# chess_ai/ — the Python package

Orchestration, learning, and data — everything where cheap iteration matters more than raw speed. Local dev runs on MPS; cloud workers run CUDA.

| Subpackage | Contents | Phase |
|---|---|---|
| `net/` | **Built (Phase 2)**: `features.py` (D1 move space — 64×64+promos, 4240 slots — + Python reference encoder) and `model.py` (CF-6M-style transformer: Shaw relpos, bilinear policy head, WDL value head, aux reply head) | 2 |
| `train/` | **Built (Phase 2)**: `data.py` (GameRecord / ReplayBuffer / sample building / seeded fixtures) and `trainer.py` (policy-CE + WDL-CE + aux-CE loss, Adam, atomic checkpointing, device auto-pick) | 2 |
| `selfplay/` | Python-orchestrated batched self-play: C advances N games in lockstep until leaves pend, one batched forward pass per round; game recorder + replay-buffer writer; checkpoint-ladder match runner | 3 |

Package root (live now):

- `engine.py` — cffi (ABI mode) binding over `engine/build/libchess`: `Board` (fen, legal moves, push/pop, termination, perft), `Termination` enum. `CHESS_LIB` env var swaps the library (e.g. the ASan build).
- `strategies.py` — dummy players: `RandomStrategy`, `GreedyMaterialStrategy` (one-ply capture logic + immediate-mate). The fixed opponents for early gauges.
- `match.py` — `play_game` / `play_match` between two strategies.
- `uci.py` — minimal hand-rolled UCI client (`UCIEngine`, `UCIStrategy`) for playing against external engines (test/calibration only — never training data).
- `san.py` — UCI → SAN ("Nf3", "exd5", "O-O", "Qh5#") for human-facing output; parity-tested against python-chess's `Board.san()`.

Try it:
```sh
```
