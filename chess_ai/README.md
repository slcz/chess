# chess_ai/ — the Python package

Orchestration, learning, and data — everything where cheap iteration matters more than raw speed. Local dev runs on MPS; cloud workers run CUDA.

| Subpackage | Contents | Phase |
|---|---|---|
| `net/` | PyTorch ResNet trunk + policy/value heads; input-plane shapes; policy-move encoding (decision D1 in PLAN.md §11) | 2 |
| `train/` | Replay buffer, policy-CE + value-MSE loss, Adam, checkpointing with immediate push off-instance | 2 |
| `selfplay/` | Python-orchestrated batched self-play: C advances N games in lockstep until leaves pend, one batched forward pass per round; game recorder + replay-buffer writer; checkpoint-ladder match runner | 3 |

The FFI bindings to the C engine (ctypes/cffi, array-in/array-out, no strings in the hot path, GIL released during C calls) live at the package root — scaffolded in Phase 0.3.
