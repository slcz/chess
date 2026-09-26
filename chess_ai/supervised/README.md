# chess_ai/supervised/ — supervised pretraining pipeline (production seed path)

**Status (D6, 2026-09-25): this code is part of the mainline.** The
supervised track validated the Phase 2.1 architecture on real data and
was promoted to the production initialization path: self-play (Phase 3)
boots from its strongest checkpoint. The quarantine that governed the
track's experimental phase survives in its data half, which is still
binding:

- **What merges**: this code, the trainer knobs, the ops scripts.
- **What never enters git or training**: the datasets, packs, and
  checkpoints (`data/`, `runs/` are gitignored). The seed checkpoint
  travels via HuggingFace (`chriszhang01/chess-value-net`) and the
  operator's `runs/` dir — never via the repo.
- **The one-time boundary**: per D6, human knowledge enters the
  production system exactly once — as weight initialization. No human
  games in replay buffers, targets, or post-init training data.

## What & why (agreed 2026-09-24; training regime updated after round 3)

Validate the Phase 2.1 transformer on real human games *before* Phase 3
self-play cloud spend — mirroring how arXiv 2409.12272 itself validated
(supervised). The track ran two regimes:

- **Rounds 1/2 — value-only** (the original agreement): train **only
  the WDL value head**, no policy supervision. Play = evaluate every
  legal move with the value net (position after the move, sign-flipped
  to our perspective; terminal moves scored exactly) and take the
  argmax: a 1-ply greedy value player. If that bot plays sensible
  chess, the Phase 2.1 trunk + value architecture is sound.
- **Rounds 3/3b — joint policy+value (current regime)**: add policy
  imitation on the mover's actual moves (`--policy-weight 1.0`). The
  multi-task trunk helped both heads (value 60.2% / policy top-1 50.7%,
  Maia-1900 territory), and the imitation bot's human-style play
  answered the bootstrap-seeding question. **The promoted seed
  checkpoint (`runs/policy_value_net_42k.pt`) is a joint net.**

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
| `play.py` | `ValueStrategy`: 1-ply greedy value player; mate-in-1 exact, one batched forward per move. `PolicyStrategy`: imitation of the policy head (the round-3 bot) |
| `train_value.py` | CLI: `download / preprocess / train / eval / match`; atomic checkpoints, optional HF push (`--hf-repo`, HF_TOKEN in .env) |

`trainer.py` (main's Phase 2 loop, reused here) gained a
`TrainConfig.policy_weight` knob — 0.0 gives value-only training; the
untouched policy head's accuracy is printed as a control. Training with
`aux_reply=False` nets also exposed and fixed a latent trainer bug (aux
loss computed without an aux head → KeyError). The CLI default is
`--policy-weight 1.0` (joint — the round-3/3b regime that produced the
promoted checkpoint); `0.0` brings back the value-only regime.

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
- 2026-09-25, round 3b — pack doubled to 600k games (43.8M positions,
  fresh second half; built on the Mac, scp'd — node-side fastprep
  quarantined after a worker-starvation bug produced a 3 KB "600k" pack
  that briefly memorized: value acc 0.99 told on it; caught by user).
  +20k steps at b4096 bf16 (total 42k): **held-out policy 50.7%**
  (Maia-1900 territory, mixed 1600-2730 pool), value 60.2% / NLL
  0.794. Matches: policy vs greedy **17-0-3**, policy vs random 18-0-2,
  value-head vs greedy **19-0-1**, self-demo policy-vs-value **10-0**
  (the sharper policy now beats its own value head).
