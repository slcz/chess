# chess_ai/ — the Python package

Orchestration, learning, and data — everything where cheap iteration matters more than raw speed. Local dev runs on MPS; cloud workers run CUDA.

| Subpackage | Contents | Phase |
|---|---|---|
| `net/` | PyTorch ResNet trunk + policy/value heads; input-plane shapes; policy-move encoding (decision D1 in PLAN.md §11) | 2 |
| `train/` | Replay buffer, policy-CE + value-MSE loss, Adam, checkpointing with immediate push off-instance | 2 |
| `selfplay/` | Python-orchestrated batched self-play: C advances N games in lockstep until leaves pend, one batched forward pass per round; game recorder + replay-buffer writer; checkpoint-ladder match runner | 3 |

Package root (live now):

- `engine.py` — cffi (ABI mode) binding over `engine/build/libchess`: `Board` (fen, legal moves, push/pop, termination, perft), `Termination` enum. `CHESS_LIB` env var swaps the library (e.g. the ASan build).
- `strategies.py` — dummy players: `RandomStrategy`, `GreedyMaterialStrategy` (one-ply capture logic + immediate-mate). The fixed opponents for early gauges.
- `match.py` — `play_game` / `play_match` between two strategies.
- `uci.py` — minimal hand-rolled UCI client (`UCIEngine`, `UCIStrategy`) for playing against external engines (test/calibration only — never training data).
- `san.py` — UCI → SAN ("Nf3", "exd5", "O-O", "Qh5#") for human-facing output; parity-tested against python-chess's `Board.san()`.
- `gui.py` + `static/board.html` — local browser GUI: human vs greedy/random/UCI engine, human vs human, or watch two machines. Click a piece to see its legal destinations; step/autoplay through history. Zero dependencies beyond stdlib.

Try it:
```sh
.venv/bin/python scripts/play_demo.py   # terminal demo: matches + engine game
.venv/bin/python scripts/play_gui.py    # browser GUI (you vs greedy by default)
```
