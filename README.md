# chess — a zero-knowledge chess AI

A chess engine that learns to play **purely from self-play** — AlphaZero-style: it knows the rules, plays against itself, and trains on the results. No human games, no engine-labeled data, no opening books. Everything is hand-written (board representation, move generation, MCTS, the neural network, the training pipeline); PyTorch is used only as a tensor/autodiff backend.

**Goal:** expert strength — roughly **1800–2100 human-equivalent Elo** (calibrated against node-limited Stockfish, ±200). Stretch: 2200–2500.

This is an educational project: the point is to build and understand every piece ourselves. The full phased plan — architecture, acceptance gates, compute/cost estimates, risk register — lives in [PLAN.md](PLAN.md).

## Status

**Phase 0 — repo & tooling bootstrap.** No engine code yet. Current state and next subtask are tracked in the status table at the top of [PLAN.md](PLAN.md).

## Architecture at a glance

Two languages, one boundary (the same split as Lc0 / KataGo):

- **C play path** — rules, board state, move generation, MCTS search, and input-plane encoding, compiled as a shared library with no globals and zero dynamic memory. Pure Python is 10–100× too slow to feed a batched GPU.
- **Python + PyTorch** — orchestration, batched NN inference, training, and the data pipeline: everything where iteration speed matters more than raw speed.
- **The boundary: Python-orchestrated batching.** C advances many games until they need NN evaluation and returns packed input planes for all pending leaves; Python runs one batched forward pass and returns priors + values as arrays. Array-in/array-out, no strings in the hot path.

## Repository layout

| Path | Contents |
|---|---|
| [`engine/`](engine/) | C play path: rules → movegen → MCTS → input-plane encoder (Phases 1–3) |
| [`chess_ai/`](chess_ai/) | Python package: FFI bindings, `net/`, `train/`, `selfplay/` |
| [`scripts/`](scripts/) | salad.com ops: worker bootstrap, work queue, churn drills, cost meter |
| [`tests/`](tests/) | pytest: perft fixtures, rules unit tests, python-chess oracle fuzzing |
| [`PLAN.md`](PLAN.md) | The development plan (phases, gates, estimates, risks) |
| [`AGENTS.md`](AGENTS.md) | Working agreements for developing this repo |

## Compute tiers & cost

Dev on a 16 GB Mac; training on salad.com cloud GPUs (RTX 3090/4090/5090), treated as **preemptible cattle**: every long-running job is checkpointed, resumable, and keeps no durable state on an instance. No cloud spend until the tiny-game smoke test passes (Phase 3 gate).

Approximate salad community rates (2026): 3090 ≈ $0.20–0.30/hr, 4090 ≈ $0.35–0.50/hr, 5090 ≈ $0.50–0.70/hr. "Full run" = ~0.5–2M self-play games.

| Tier | Hardware | Time to ~1800–2100 | Full-run cost | Notes |
|---|---|---|---|---|
| **T0 — local Mac only** | MacBook 16 GB | 2–6 months; realistically plateaus ~1500–1800 | $0 | Dev, perft, smoke tests, 5–20k-game pilots. 16 GB RAM forces small replay buffers. |
| **T1 — single GPU** ⭐ | 1× 4090-class on salad | ~1–2 weeks/run | ~$100–350/run | Recommended minimum for serious runs; Phases 5–6 start here. A week of runs+tuning ≈ $300–800. |
| **T2 — multi-GPU** | 4× 4090-class (one box, or 2–4 independent workers — churn-tolerant) | ~3–6 days/run | ~$200–700/run | Expert run + parallel hyperparameter sweeps. Project total **$500–2,000**. |
| **T3 — stretch campaign** | 8× 4090/A100-class | for **2200–2500** | $1.5–5k | Bigger net (16–20 × 192–256), 400–800 eval sims, 2–5M more games, 2–4 GPU-weeks. |
| *(Reference)* Lc0 / AlphaZero class | thousands of GPUs/TPUs | GPU-years | $10⁵–10⁶+ | 3000+ engines. Not a solo project. |

The key compute lever vs. a naive AlphaZero re-run: **Gumbel MCTS with 2–32 simulations** (MiniZero: Gumbel at 2–16 sims ≈ vanilla at 200) plus KataGo-style training improvements — a 10–50× reduction using zero human knowledge. Reasoning and sources: PLAN.md §8.

## Development setup

Coming in Phase 0.2/0.3: pytest + ruff wiring, C build via Makefile (default + ASan/UBSan targets), cffi binding scaffold, pinned Python environment with a lockfile, PyTorch-on-MPS verification, and the worker-bootstrap skeleton.
