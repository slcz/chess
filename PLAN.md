# Chess AI — Development Plan (v2.1)

**Zero-knowledge chess AI**: an engine that learns chess purely from self-play (AlphaZero-style — rules only, no human games, no engine-labeled data), with everything hand-written (movegen, FEN, MCTS, NN, training pipeline); PyTorch is used only as a tensor/autodiff backend.

This document has two parts:

1. **Execution plan (§3–§5)** — phased subtasks with acceptance gates. This is what we actually work through, one subtask at a time (pause and align after each, per AGENTS.md).
2. **Reference analysis (§6–§12)** — effort/compute/cost estimates, research anchors, risks, and verification methodology behind the plan. Read once; consult when numbers are questioned.

**Working environment** (from AGENTS.md): development on the 16 GB Mac; training on salad.com cloud instances (RTX 3090/4090/5090) which are cheap but **can disappear at any moment** — so every long-running job is designed to be checkpointed, resumable, and stateless per instance. All source lives in the GitHub repository `chess`.

---

## 0. Current status (update as we go)

| Item | State |
|---|---|
| Plan | v2.1 — execution plan + reference analysis; **language split decided**: C play path + Python training/batched inference, Python-orchestrated batching (2026-09-22) |
| GitHub repo `chess` | **created (private)**: <https://github.com/slcz/chess> — 0.1 done (AGENTS/PLAN/README/.gitignore committed); 0.2 layout scaffolded (folders + per-dir READMEs) |
| Code | none yet (layout only) |
| Current phase | **Phase 0 — repo & tooling bootstrap** |
| Next subtask | 0.2: wire up pytest + ruff; then 0.3 (Makefile + cffi scaffold + pinned env) |
| Training runs | none |
| Cloud spend so far | $0 |

---

## 1. Goal & locked decisions

| Decision | Choice |
|---|---|
| Target strength | **Expert ~1800–2100 Elo** (human-equivalent, calibrated vs. node-limited Stockfish); stretch: 2200–2500 |
| Learning method | **Pure self-play** (AlphaZero-style, zero human/engine knowledge) |
| Build scope | **Truly everything hand-written**; PyTorch/JAX only as tensor/autodiff backends |
| Language split | **C** for the play path (rules, state, search, input-plane encoding) as a shared library; **Python + PyTorch** for orchestration, batched NN inference, training, data pipeline. Boundary = **Python-orchestrated batching** (§3) |
| Budget | Open — cost tiers in §6; principle: **no cloud spend until the tiny-game smoke test passes** (§4, Phase 3) |
| Dev machine | MacBook, 16 GB — great for dev, perft, smoke tests, small pilot runs; weak for full training |
| Training hardware | salad.com instances (RTX 3090/4090/5090), treated as preemptible cattle (§5) |

## 2. Estimates at a glance

| Question | Answer |
|---|---|
| **Effort to build** | **~3–6 focused months** solo (experienced dev); ~6–10 months if also learning chess-engine/ML systems along the way. ~2–4 weeks of that is rules/movegen alone. |
| **Compute to train (recommended efficient path)** | **1× RTX 4090-class for ~1–2 weeks**, or 4× for ~3–6 days. Roughly **$300–1,200 cloud per serious run** at salad-class prices; budget **$500–2,000 total** across the project's training. |
| **Compute if you literally re-run vanilla AlphaZero (800 sims, 20×256 net)** | 10–40× more → ~1–4 GPU-*years* on consumer hardware. Don't. |
| **Simulation volume** | ~0.5–2M self-play games (~50–80 plies each) ≈ 10⁹–10¹⁰ NN evaluations. |
| **Training wall-clock** | 1×4090-class: ~1–2 weeks/run. 4×: ~3–6 days/run. Plus weeks of human iteration *between* runs. |
| **Local Mac alone reaches…** | ~1500–1800 realistically; 1800–2100 only with extreme patience. |
| **Stretch 2200–2500** | +2–4 GPU-weeks on 4–8 GPUs, bigger net, 2–5M more games. |
| **3000+ (Stockfish/Lc0 tier)** | Out of reach solo — GPU-years of distributed compute. |

**Biggest lever:** modern zero-knowledge efficiency tricks — **Gumbel MCTS with 2–32 simulations** (MiniZero: Gumbel at 2–16 sims ≈ vanilla at 200 sims) and KataGo-style training improvements. They cut compute **10–50×** vs. a naive AlphaZero reimplementation while using zero human knowledge. Full reasoning in §8.

---

## 3. System architecture

| Component | Choice | Why |
|---|---|---|
| Language split | **Play path in C** (rules, state, MCTS, feature encoding) compiled as a shared library with **no globals** (hundreds of game instances per process); **Python 3.11+ / PyTorch** (MPS locally, CUDA on cloud), pytest, ruff for everything else | Same split as Lc0/KataGo. A batched forward pass can service tens of thousands of NN evals/sec; only C game logic + search can feed it (pure Python is 10–100× too slow, §9.2). Python stays wherever iteration is cheap (training, data, ops). |
| Rules/movegen | **Bitboards** (fallback 0x88) in C — fixed-size `State` struct, stack move lists, make/unmake with a fixed undo stack, **zero dynamic memory**; full state: castling, en passant, promotion, check/mate/stalemate, 50-move, threefold, insufficient material; FEN in/out | "Everything hand-written" scope. The zero-malloc design removes most C pitfalls up front. **Verified with perft** (standard node-count fixtures) — non-negotiable before any ML code. |
| Net (first run) | ResNet **6–10 blocks × 64–128 filters** (~1–5M params); policy head + value head (tanh) | MiniZero got strong with 2–3 blocks on smaller games; Lc0 started at 64×6. 20×256 is overkill until 2400+. Exact policy encoding (4672-move vs. simpler 64×64+promo) decided in Phase 2. |
| Net (expert run) | 8–12 blocks × 128 filters (~5–15M params) | Room to grow without 10× training cost. |
| Input planes | 12–18 planes (pieces, side-to-move, castling rights, ep file, halfmove clock, repetition history) | Standard AZ encoding; repetition history mitigates draw-collapse (§10.4). |
| Search | **In C**: Gumbel MCTS + sequential halving (2–32 sims/move in self-play); plain PUCT-MCTS (100–200 sims, Dirichlet root noise) for eval matches. Tree nodes from an arena/node-pool, reset per move | Gumbel is the biggest zero-knowledge compute lever. C also makes high-sim eval matches pleasant — Python per-node overhead would hurt exactly there. |
| Training | Replay buffer (recent 40–50k games / ~500k–1M positions), policy target = MCTS visit distribution, value target = game outcome (z), Adam, batch 512–1024, temperature τ=1 for first ~8–16 plies then →0 | Matches MiniZero/AZ practice. |
| Self-play infra | **Python-orchestrated batching** (boundary A): Python outer loop over N games in lockstep; one coarse C call per game — "advance until eval needed" — returns packed input planes for pending leaves; Python runs a single batched forward pass and returns priors + value as arrays. Array-in/array-out, no strings in the hot path; ctypes/cffi release the GIL during C calls, so worker threads parallelize | All fast-path code in C, all experimental knobs in Python. C-orchestrated actors with embedded inference (libtorch/ONNX) are a later *additive* optimization, only if profiling demands it. |
| Eval/calibration | Fixed-match gauntlet vs. **node-limited Stockfish** (10–100k nodes/move ≈ roughly 1800–2200) + mate-in-1/2/3 tactic suite | The only sane way to claim "1800–2100 human-equivalent" without human playtesting; report with ±200 calibration caveat. |

---

## 4. Execution plan (phases, subtasks, acceptance gates)

Each phase ends with a **gate**. We do not start the next phase (or incur cloud cost) until the gate passes and we've aligned. "wks" = focused solo person-weeks — estimates, not promises.

### Phase 0 — Repo & tooling bootstrap (~0.5 wk)
| # | Subtask |
|---|---|
| 0.1 | Create GitHub repo **`chess`** (private); commit AGENTS.md, PLAN.md, README (with the tier table from §6), `.gitignore` |
| 0.2 | Repo layout: `engine/` (C: rules → search), `chess_ai/` (Python: cffi bindings, `net/`, `train/`, `selfplay/`), `scripts/` (salad ops), `tests/` (incl. perft fixtures); pytest + ruff wired up |
| 0.3 | C build (Makefile: shared-lib + ASan/UBSan targets) + cffi binding scaffold (array-in/array-out conventions); pinned Python env (lockfile); verify PyTorch on MPS |
| 0.4 | Worker-bootstrap skeleton (§5): script that goes from bare machine → clone → install → pull checkpoint → run one toy work unit → push result. Runs on the Mac first |

**Gate:** fresh `git clone` → C library builds clean in default + ASan modes and loads via cffi; `pytest` green on the Mac; bootstrap script completes a toy work unit end-to-end locally.

### Phase 1 — Rules engine in C (3–5 wks)
| # | Subtask |
|---|---|
| 1.1 | Board representation (fixed-size `State`, stack move lists, fixed undo stack — zero malloc); make/unmake; FEN parse/serialize |
| 1.2 | Full legal movegen: castling (incl. through check), en passant (incl. ep pins), promotions, check detection |
| 1.3 | Termination: mate, stalemate, 50-move, threefold repetition, insufficient material |
| 1.4 | Perft suite (startpos, Kiwipete, …) via a small C CLI (`perft`, `perft divide`) |
| 1.5 | Oracle fuzzing vs. **python-chess** (test-only — never touches training data): legal-move sets, termination calls, and FEN round-trips agree across ≥10⁶ random game walks |

Deceptively fiddly — one en-passant edge-case bug poisons all future training data silently (§10.1).

**Gate:** perft matches known node counts **exactly** at depth 5–6 on all fixtures (C CLI and via FFI); oracle fuzzing clean; unit tests for the edge cases above; engine plays random legal games against itself.

### Phase 2 — NN + training loop (1–2 wks)
| # | Subtask |
|---|---|
| 2.1 | Input-plane encoder (implemented once, in the C engine — shared by self-play and training data prep); ResNet trunk; policy + value heads; decide policy encoding (D1: 4672 vs. 64×64+promo) |
| 2.2 | Loss (policy cross-entropy + value MSE), Adam, replay buffer, checkpointing |
| 2.3 | Sanity tests: overfit on a tiny fixed dataset; illegal moves masked to zero probability |

**Gate:** net memorizes ~100 fixed games' policy/value targets; illegal-move masking asserted in tests.

### Phase 3 — Search & self-play (2–5 wks; includes FFI/batching integration)
| # | Subtask |
|---|---|
| 3.1 | MCTS in C — PUCT with Dirichlet root noise, temperature schedule, visit-distribution extraction; tree nodes from an arena/node-pool reset per move |
| 3.2 | Gumbel MCTS + sequential halving variant on the same C core |
| 3.3 | Python-orchestrated batched leaf evaluation across N games in lockstep — the throughput core: C returns packed planes for pending leaves, Python runs one batched forward pass, priors/value return as arrays; GIL released during C calls so worker threads parallelize |
| 3.4 | Game recorder + replay buffer writer |
| 3.5 | **Tiny-game smoke test**: same training/batching pipeline on tic-tac-toe or 5×5 connect-4 (trivial pure-Python game impl is fine — this validates the *learn loop*, not the C engine) → perfect play in minutes |

**Gate:** tiny game solved (validates the entire learn loop **before any cloud spend**); chess self-play throughput measured and reported (games/GPU-hour); checkpoint ladder machinery works (net@A vs. net@B match runner).

### Phase 4 — Cloud orchestration on salad.com (1–2 wks)
| # | Subtask |
|---|---|
| 4.1 | Containerize the worker (salad runs Docker images); the image **builds the C engine from source** — no binary shipping, and Mac-arm64 vs. cloud-x86_64 stops being a problem; bootstrap = pull image → fetch work unit → run → push artifacts → heartbeat loop |
| 4.2 | Work-queue protocol: idempotent units ("from checkpoint N, play K games, upload shard"); Mac as control plane submitting units and watching heartbeats |
| 4.3 | Checkpoint/push protocol: trainer checkpoints every few minutes and pushes off-instance immediately; artifacts land in the stores chosen in §5 |
| 4.4 | Churn drill: kill an instance mid-job, verify ≤ one work unit lost and state reconstructs from remote stores alone |
| 4.5 | Cost meter: hours + $ per run, alert threshold |

**Gate:** churn drill passes; a 5–20k-game pilot run completes on 1 salad GPU (or the Mac if a cheap-enough pilot isn't needed) with clean resume after a forced kill.

### Phase 5 — First full training run (1–2 wks wall-clock on 1×4090-class)
Run to ~50–200k games. Watch the checkpoint ladder (net@10k < net@50k < net@200k in gauntlet matches); fix the inevitable bugs; measure Elo vs. node-limited Stockfish.

**Gate:** monotone checkpoint ladder; estimated Elo reported with the calibration caveat; go/no-go decision on scaling to the expert run.

### Phase 6 — Expert run + tuning (3–8+ wks, iterative)
Scale data to 0.5–2M games (§7). Sweep sims count, net size, LR/batch. Candidates: several **independent 1-GPU salad workers** (churn-tolerant, self-play parallelizes trivially) vs. one 4-GPU box — decide by measured $/game. Stretch 2200–2500 needs a bigger net (16–20 × 192–256) and 2–5M more games (§6, T3).

**Gate (final):** 500-game matches vs. node-limited Stockfish at 3–5 node budgets + tactic suite; claim "≈1800–2100 human-equivalent ±200" only with this evidence.

### Timeline reality check
Phases 0–4 ≈ **9–15 wks** of build (includes the C/FFI tooling); Phases 5–6 add **4–10 wks** of iteration (most from-scratch AlphaZero projects die here — the riskiest phase is deliberately the longest). Total ≈ **3–6 months** experienced; **6–10 months** if learning the domains along the way. Milestones: end of wk 2 = perft-clean engine; ~month 2 = learning pipeline visibly beating its old checkpoints; months 3–5 = expert net.

---

## 5. Operating on salad.com (unreliability-first design)

Salad instances are cheap but **can vanish at any moment**. The whole ops design follows from treating that as *normal operation*, not an exception:

1. **No durable state on an instance.** Code, checkpoints, games, and metrics live in remote stores; instances only cache. Any instance can be destroyed at any time with ≤ minutes of lost work.
2. **Work units are small and idempotent** — e.g., "from checkpoint N, play 100 games, upload shard G." A killed worker loses one unit, which the control plane re-issues.
3. **Trainer checkpoints frequently and pushes immediately** (every few minutes / few hundred samples). Never multi-GB local buffers without a remote copy.
4. **One-command bootstrap**: container image + entrypoint that goes bare-instance → working worker with no manual steps.
5. **The Mac is the control plane**: submits work units, watches heartbeats, replaces dead workers, runs pilot runs itself, and is the fallback if salad is entirely down (testing must always be runnable locally).
6. **Every script has a backup plan** — collaborate with the user before long unattended runs (per AGENTS.md): expected duration, cost cap, kill switch.

**Artifact stores (open decision D2 in §11):** code → GitHub `chess`; checkpoints/metrics → GitHub (releases or LFS); game archives → rolling window + optional full archive (HuggingFace dataset or equivalent), since multi-GB game data does not belong in a git repo.

---

## 6. Compute tiers & cost (salad.com-class pricing, 2026)

Approximate salad community rates: **RTX 3090 ≈ $0.20–0.30/hr, 4090 ≈ $0.35–0.50/hr, 5090 ≈ $0.50–0.70/hr** — verify at purchase time; churn costs an effective ~10–30% overhead (lost units, restarts) already priced into the estimates below. Efficient config throughout: Gumbel 8–32 sims, ≤ 10×128 net. "Full run" = ~0.5–2M games to plateau.

| Tier | Hardware | Time to ~1800–2100 | Full-run cost | Notes |
|---|---|---|---|---|
| **T0 — local Mac only** | MacBook 16 GB | 2–6 months; realistically plateaus ~1500–1800 | $0 | Dev, perft, smoke tests, 5–20k-game pilots. 16 GB RAM forces small replay buffers. |
| **T1 — single GPU** ⭐ | 1× 4090-class on salad | ~1–2 weeks/run | ~$100–350/run | Recommended minimum for serious runs; Phases 5–6 start here. A week of runs+tuning ≈ $300–800. |
| **T2 — multi-GPU** | 4× 4090-class (one box, or 2–4 independent workers — churn-tolerant) | ~3–6 days/run | ~$200–700/run | Expert run + parallel hyperparameter sweeps. Project total **$500–2,000**. |
| **T3 — stretch campaign** | 8× 4090/A100-class | for **2200–2500** | $1.5–5k | Bigger net (16–20 × 192–256), 400–800 eval sims, 2–5M more games, 2–4 GPU-weeks. |
| *(Reference)* Lc0/AlphaZero class | thousands of GPUs/TPUs | GPU-years | $10⁵–10⁶+ | 3000+ engines. Not a solo project. |

**CPU/RAM/storage (any tier):** ~16–32 vCPU host for actors (game logic is CPU-side), 32–64 GB RAM, ~100–500 GB scratch disk on-instance (disposable). Self-play is CPU+GPU hybrid; the GPU is the bottleneck only if leaf evaluation is batched properly.

## 7. Training dynamics reference

**Volume per full run:** ~0.5–2M games → ~30–150M positions → ~10⁹–10¹⁰ NN evals (Gumbel-32 ≈ 32 sims × ~60–80 plies × 10⁶ games). NN *training* steps (~1–5M updates at batch 512–1024) are the cheap part; **self-play generation dominates (~90%+ of compute)**.

**Indicative Elo vs. self-play games** (efficient config, ±200):

| Self-play games | Approx. strength | Wall-clock (T2) |
|---|---|---|
| 5–20k | ~800–1300 (sees 1-move tactics) | hours |
| 50–100k | ~1400–1700 | 1–2 days |
| 300–800k | ~1700–2000 | 3–8 days |
| 1–2M | ~1900–2100 (target) | 1.5–3 wks cumulative w/ tuning |

---

## 8. Research anchors (verified, not folklore)

| System | Compute | Result |
|---|---|---|
| **AlphaZero** (Science 2018) | 5,000 TPUv1 self-play + 64 TPUv2 training; 20×256 net | Beat Stockfish 8 after ~4 h *at that scale*. The "4 hours" is meaningless without the 5,000-TPU pod. |
| **MuZero** (board games) | 1,000 TPUs self-play + 16 TPUs training | Matched AlphaZero chess/shogi at ~1M steps |
| **KataGo** (arXiv 1902.10565) | ~50× less compute than comparable AZ runs; main run ≈ 1.2–1.4 V100-years | Authors: "strong or top human amateur strength starting from nothing on even single GPUs in mere days" (Go, with their improvements) |
| **MiniZero** (arXiv 2310.11305) | One machine: 4× GTX 1080 Ti; 600k games | Gumbel at 2–16 sims ≈ vanilla at 200 sims, equal wall-clock — on games *smaller* than chess |
| **Leela Chess Zero** | Distributed volunteer GPUs, cumulative GPU-years | Chess self-play works at scale; nets 64×6 → 256×20 |
| **Pgx** (arXiv 2303.17503) | Vectorized game sims 10–100× faster than naive Python | Design target for our batched self-play loop |
| **NNUE** | Supervised on engine-labeled positions | **Rejected** — violates zero-knowledge |

**Reading for our case:** expert human level (1800–2100) is far below superhuman. Chess is harder than Othello/9×9 Go but much easier than 19×19 Go. With Gumbel + small net + batched self-play, a single modern consumer GPU in days–weeks is the right order of magnitude — KataGo's single-GPU claim, scaled from Go to expert-level chess. A naive AZ reimplementation is ~10× more expensive.

---

## 9. Risk register (what blows estimates up)

1. **Rules bugs** — one en-passant/castling edge-case poisons all training data silently. → Perft fixtures + python-chess oracle fuzzing gate Phase 1, before any ML code.
2. **Naive self-play throughput** — one-game-at-a-time Python MCTS yields 10–100× less data/hour. The #1 reason hobby AZ projects "take forever". → Batched leaf eval across parallel games from day one (Phase 3.3).
3. **Value-target sign / illegal-move masking bugs** — training "converges" but strength plateaus ~1200. → Overfit tests + zero-visit assertions on illegal moves (Phase 2 gate).
4. **Draw-collapse / repetition blindness** — net shuffles into draws. → Repetition history planes; adjudicate long shuffling games.
5. **Calibration drift** — "beats 10k-node Stockfish" ≠ FIDE 2000. → Always report "≈ human-equivalent, calibrated vs. node-limited Stockfish ±200"; true Elo only via ~50 human/bot games.
6. **Instance churn on salad** — boxes vanish mid-run. → First-class design constraint, §5 (stateless workers, idempotent units, immediate pushes); not a footnote.
7. **Iteration quicksand (Phases 5–6)** — most from-scratch AZ projects die in debugging/tuning loops. → Explicit gates, checkpoint ladder as the pipeline-health signal, timebox each debug cycle and reassess with the user.
8. **Two-language overhead** — FFI/build complexity and C memory bugs can stall momentum early. → Zero-malloc rules engine + arena-allocated trees; ASan/UBSan targets from Phase 0; python-chess oracle pinpoints disagreements fast; the container builds the C source, so local/arch mismatch never leaks into ops.

## 10. Verification methodology

1. **Perft suite** — standard positions (startpos, Kiwipete, …) to depth 5–6, exact node counts, via the C CLI and through the FFI. *(Phase 1 gate)*
2. **Rules unit tests** — castling through check, ep disambiguation, promotion, 50-move/threefold/insufficient material. *(Phase 1 gate)*
3. **Oracle fuzzing** — python-chess agrees on legal-move sets, termination calls, and FEN round-trips across ≥10⁶ random walks. python-chess and Stockfish are **test/calibration oracles only** — they never generate training data or labels, so zero-knowledge is intact. *(Phase 1 gate; re-run in CI)*
4. **C hygiene** — ASan/UBSan clean under the pytest suite. *(every phase touching `engine/`)*
5. **Tiny-game smoke test** — full pipeline reaches perfect play on tic-tac-toe/connect-4 in minutes, before any cloud spend. *(Phase 3 gate)*
6. **Overfit test** — net memorizes 100 fixed games' targets. *(Phase 2 gate)*
7. **Checkpoint ladder** — net@10k < net@50k < net@200k in gauntlet matches; monotone strength = pipeline sound. *(Phase 5 gate)*
8. **Churn drill** — kill an instance mid-job; ≤ one work unit lost; state rebuilds from remote stores. *(Phase 4 gate)*
9. **Final gauntlet** — 500-game matches vs. node-limited Stockfish at 3–5 node budgets + mate-in-1/2/3 tactic suite. *(Phase 6 gate)*

## 11. Open decisions (align with user before the relevant phase)

| # | Decision | Needed by | Recommendation |
|---|---|---|---|
| D1 | Policy-head encoding: 4672-move (LC0-style) vs. simpler 64×64+promo | Phase 2 | 4672 — standard, no harder to mask, matches published lore |
| D2 | Game-archive store: discard-beyond-buffer vs. GitHub releases vs. HuggingFace dataset | Phase 4 | Rolling buffer in-repo + full archive on HuggingFace (free private storage; multi-GB doesn't belong in git) |
| D3 | Expert-run topology: one 4-GPU instance vs. 2–4 independent 1-GPU workers | Phase 6 | Independent workers — churn-tolerant, self-play parallelizes trivially, matches salad's reality |
| D4 | Repo visibility: private vs. public | Phase 0 | Private until the first successful run, then decide |

## 12. Sources

- [AlphaZero — Wikipedia](https://en.wikipedia.org/wiki/AlphaZero) · [MuZero — Wikipedia](https://en.wikipedia.org/wiki/MuZero)
- [KataGo: Accelerating Self-Play Learning in Go — arXiv 1902.10565](https://arxiv.org/abs/1902.10565)
- [MiniZero — arXiv 2310.11305](https://arxiv.org/abs/2310.11305)
- [Gumbel MuZero / Policy Improvement by Planning with Gumbel — DeepMind](https://arxiv.org/abs/2104.06303)
- [Leela Chess Zero](https://www.chessprogramming.org/Leela_Chess_Zero) · [NNUE](https://www.chessprogramming.org/NNUE) — Chessprogramming Wiki
- [Pgx: hardware-accelerated game simulators — arXiv 2303.17503](https://arxiv.org/abs/2303.17503)
