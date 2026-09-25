# chess_ai/supervised/ — value-net validation on real data (side track)

**Branch-only.** This package exists on `supervised_learning` and never
merges to main. The dataset, packs, checkpoints, and results are
quarantined on this branch (data dirs are gitignored); only bug fixes
found here get distilled to main. Main-branch development keeps the pure
zero-knowledge self-play path (PLAN §0 side-track row).

## What & why (agreed with the user, 2026-09-24)

Train **only the WDL value head** of the Phase 2.1 transformer on real
human games — no policy supervision. Play = evaluate every legal move
with the value net (position after the move, sign-flipped to our
perspective; terminal moves scored exactly) and take the argmax: a 1-ply
greedy value player. If that bot plays sensible chess, the Phase 2.1
trunk + value architecture is sound *before* self-play cloud spend.
Mirrors how arXiv 2409.12272 itself validated (supervised).

## Dataset

Lichess monthly standard-rated dumps (`database.lichess.org/standard`),
one zstd PGN per month. Chosen because the encoder needs full move
sequences (8-ply history + repetition flags), which flat FEN+eval dumps
cannot provide, and human games carry outcome labels for free.
Pilot: **2013-01** (18 MB zst / 121k games → filtered 24k games /
1.8M positions, avg Elo 1775). Cloud run: a bigger month (2016-01,
872 MB, ~4M games) — take a filtered prefix via `--max-games`.

Default filters: both Elo ≥ 1600, time-control base ≥ 180 s,
10–300 plies. All knobs on `preprocess`, stats written next to every
pack (`*.stats.json`).

## Layout

| module | role |
|---|---|
| `pgn.py` | streaming Lichess-PGN parser + SAN→UCI on our own engine (python-chess stays test-only) |
| `dataset.py` | filter → compact games pack; `PositionSampler` encodes positions on demand (13 ms / batch-256 on the M4 — the Phase 3 streaming path, forced early as planned) |
| `play.py` | `ValueStrategy`: 1-ply greedy value player; mate-in-1 exact, one batched forward per move |
| `train_value.py` | CLI: `download / preprocess / train / eval / match`; atomic checkpoints, optional HF push (`--hf-repo`, HF_TOKEN in .env) |

`trainer.py` (main's Phase 2 loop, reused here) gained a
`TrainConfig.policy_weight` knob — 0.0 gives value-only training; the
untouched policy head's accuracy is printed as a control. Training with
`aux_reply=False` nets also exposed and fixed a latent trainer bug (aux
loss computed without an aux head → KeyError).

## Known pathology

Greedy value play has no progress pressure: ahead in material, it
shuffles into threefold draws. That's the 1-ply consumer's limit, not
the net's — report draw rates, don't hide them.

## Run log

- 2026-09-24, Mac pilot, 2013-01 pack: full 6M net, MPS, batch 256.
  MPS is ~0.6 s/step (encode is 13 ms — MPS forward/backward is the
  whole cost) → the Mac is for pipeline proof only; real training on
  salad 3090/4090 (container work pending, user-driven).
- 2026-09-25, salad 5090, 2016-01 pack (300k games / 21.9M positions):
  fp32 b1024 20k steps → 57.8% val (+10.9 over prior); fp32 b3072 →
  59.1% at 27k; **bf16 = 2.0× step rate, quality-neutral → adopted**;
  b4096 fits in bf16 (marginal per-sample gain). **torch.compile
  (reduce-overhead) REJECTED**: no speedup (0.31→0.30 s/step) and it
  DEGRADED training (val 59 → 57.2 within 500 steps — CUDA-graph
  replay vs grad-clip/fused-Adam interaction; only visible once the
  resume-recompile bug was fixed). Deep lever parked for main branch:
  SDPA/Flash rewrite of Shaw attention. Best net: 30.5k steps, 59% val
  (runs/value_cloud/final_30k.pt, HF canonical, GUI-serving).
- 2026-09-25, round 3 — **joint policy+value** (fresh net, b1024, bf16,
  21.5k steps, same data): held-out **policy top-1 41.7%** (baseline 3%)
  and **value 60.5% / NLL 0.797 — both above the value-only runs** (the
  multi-task trunk helps). Matches (20 games): policy vs greedy **9-1-10**,
  policy vs random 13-0-7, value-head vs greedy **12-2-6** (the 1-ply
  value player beats greedy once its value comes from the joint trunk),
  self-demo policy-vs-value 0-0-10 (shared-trunk draws). The imitation
  bot plays human-style chess → the bootstrap-seeding question (see
  PLAN side-track row) is answered affirmatively.
