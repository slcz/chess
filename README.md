# chess — a self-play chess AI

A chess engine that improves itself by playing. The play path is
hand-written C: bitboard rules, strictly-legal move generation, batched
MCTS, and the NN input encoder (64 squares × 8-ply history, 117
features per square). The learner is a hand-written PyTorch transformer
(8 pre-LN blocks, d256, 8 heads, 2D RoPE; policy + WDL +
opponent-reply heads) trained on its own search-improved games. PyTorch
is only the tensor/autodiff backend.

The search is MCTS with **regularized policy optimization** (Grill et
al., 2020): descent samples π̄, the exact solution of
`argmax_y [qᵀy − λ·KL(π_θ, y)]` with `λ = c·√N/(|A|+N)`. The same π̄ is
the training target and the played move (sampled for the opening plies,
argmax after); there is no Dirichlet noise, no temperature, and no
visit-count target. The C module keeps its historical `puct` name and
`puct_*` exports.

The pipeline was bootstrapped once from supervised pretraining on human
games: human knowledge entered exactly once, as gen1 (the 42k-step
supervised net). Everything after that is pure self-play —
playout-cap-randomized (KataGo PCR) games on rented cloud GPUs, trained
through a memory-frugal streaming ingest, and gated by head-to-head
matches.

## Status (Oct 2026)

- **Supervised lineage:** gen2 is one self-play round from gen1 (280k
  games, 2 epochs, 300/50-sim PCR, lr 2e-5). It scores **0.639** vs
  gen1 over 400 games; Stockfish-ladder crossover ≈ **1835** vs ≈
  **1730** for gen1 (≈ +100 Elo).
- **Zero-knowledge lineage:** a random-init net reached gen1 parity in
  one 287k-game round (gate 0.510 vs 0.500), still ≈100 Elo below gen2.
  Separately, a from-scratch lineage on the same loop has run 20
  generations.
- **Web GUI:** one static page, no server: C compiled to WASM
  (single-threaded, stepped evaluation), ONNX net via
  onnxruntime-web (WebGPU EP, wasm fallback), play only, PGN export. No
  SharedArrayBuffer, no cross-origin isolation.

## Repository layout

| Path | Contents |
|---|---|
| [`engine/`](engine/) | C play path: rules, movegen, NN encoder, batched MCTS/RPO search, self-play and match drivers |
| [`chess_ai/`](chess_ai/) | Python: cffi bindings, `net/`, training data/loop, shard reader, SAN |
| [`scripts/`](scripts/) | Pipeline entry points: self-play loop, round training, gate matches, ONNX export, shard migration |
| [`tests/`](tests/) | pytest: C engine/perft gates, FFI, python-chess oracle fuzz, model/export, data-path parity |
| [`web/`](web/) | Static WASM GUI (play-only) |

Container/deploy/cloud/operator files, checkpoints (`runs/`), and game
data are intentionally local-only and gitignored. Design authority is
the headers in [`engine/include/`](engine/include/) and the module
docstrings.

## Build and test

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make -C engine check                          # C gates + perft suite
.venv/bin/python -m pytest tests/ -q          # Python gates
```

`make -C engine check` runs the rules/NN/search unit suites (single- and
multi-walk-task), the ply-wall build, and the full perft suite. pytest
adds the cffi boundary, python-chess oracle fuzzing, model/export
contracts, and end-to-end self-play data-path checks.

## Web GUI

```sh
make -C engine build/parity_dump    # native reference for the parity gate
sh web/get_vendor.sh                # ORT-web runtime (page + net gate)
web/build.sh                        # needs emsdk + Python >= 3.10
node web/test/parity_node.mjs       # wasm search ≡ native, bit-for-bit
node web/test/bridge_node.mjs       # stepped eval protocol ≡ native
node web/test/net_node.mjs          # real ONNX net through the loop
```

All three gates need the wasm build; the first two also need the native
`parity_dump` binary, and `net_node.mjs` additionally needs an `.onnx`
under `web/models/`. `.onnx` nets are loaded from Hugging Face through
the panel.

To assemble the deployable static site:

```sh
web/build.sh                        # wasm -> engine/build/wasm/
sh web/get_vendor.sh                # ORT-web -> web/vendor/
python3 scripts/build_web_dist.py   # -> dist/
```

`dist/` preserves the page's relative paths and needs no server; serve it
with `python3 -m http.server -d dist 8000`. The included
`.github/workflows/pages.yml` builds the wasm/ORT assets and deploys
`dist/` through GitHub Pages (enable Settings → Pages → GitHub Actions).
Without WebGPU the page falls back to the single-threaded wasm EP, using
8 walkers and a 200-sim ceiling (100 default); with WebGPU it uses 256
walkers and the full 5000-sim slider.

## Pipeline scripts

| Script | Purpose |
|---|---|
| [`scripts/selfplay_loop.py`](scripts/selfplay_loop.py) | one-machine generate → train → gate → adopt loop |
| [`scripts/sp_round_train.py`](scripts/sp_round_train.py) | train on a directory of SPSHARD1 v3 shards (optional replay buffer) |
| [`scripts/gate_match.py`](scripts/gate_match.py) | two-net gate matches through `uci_mux` |
| [`scripts/export_onnx.py`](scripts/export_onnx.py) | checkpoint → ONNX + torch/ORT parity gate |
| [`scripts/spbin_to_v3.py`](scripts/spbin_to_v3.py) | convert legacy SPSHARD1 v1/v2 shards to v3 |

The native drivers (`selfplay`, `uci_mux`, `sf_gauntlet`, `parity_dump`)
are built from [`engine/Makefile`](engine/Makefile); the Python scripts
orchestrate them.

## Roadmap

1. **Hyperparameter tuning** — lr / batch / PCR budgets / epochs around
   the gen2 recipe; candidates must clear the 400-game gate (≥ 0.500 vs
   their base) and the Stockfish ladder.
2. **Absolute strength of the zero-knowledge lineage** — later
   from-scratch generations are gated only relative to their
   predecessors; anchor them on the Stockfish ladder and compare with
   gen2.
