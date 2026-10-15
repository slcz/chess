# Hyperparameter journal

Base **gen1** (`runs/gen1.pt`, 42k supervised steps). Gate = 400 games vs
gen1, 200 sims, colors swapped: 0.5 = no difference, binomial stderr ±0.025,
but the measured **gen1-vs-gen1 null control is 0.500 (waves 0.468/0.532)**,
so treat anything under ~0.03 as noise. All runs: Adam (wd=0), slow-cap
plies only, seed 1.

```yaml
- id: REF-gen2
  nature: "The 2-epoch 280k recipe the sweep is measured against."
  params: {games: 280000, epochs: 2, lr: 2.0e-5, batch: 1024, steps: 14364, base: gen1}
  loss: null
  result: {gate_vs_gen1: 0.639, games: 400, wdl: [228, 55, 117], ckpt: selfplay/selfplay-gen2.pt}

- id: REF-gen2-e1
  nature: "Same 280k data at 1 epoch — how much of gen2 comes from the second pass?"
  params: {games: 280000, epochs: 1, lr: 2.0e-5, batch: 1024, steps: 7182, base: gen1}
  loss: null
  result: {gate_vs_gen1: 0.573, games: null, ckpt: selfplay/iterN-e1.pt}

- id: EXP-001
  nature: "Does 1 epoch on only 25k games already move the net off gen1?"
  params: {games: 25000, epochs: 1, lr: 2.0e-5, batch: 1024, steps: 618, base: gen1}
  loss: [2.6061, 2.5508, 2.5405, 2.5337, 2.5257, 2.5115]   # total loss, every 100 steps
  result: {gate_vs_gen1: 0.534, games: 400, wdl: [178, 71, 151], ckpt: selfplay/hp-g25k-e1.pt}

- id: EXP-002
  nature: "How many self-play games until the net clearly beats gen1: one 287k-game run, 2 epochs, a checkpoint every 2,791 steps."
  params: {games: 287527, epochs: 2, lr: 2.0e-5, batch: 1024, steps_per_epoch: 7353, base: gen1}
  loss: [2.6097, 2.5036, 2.4865, 2.4807, 2.4721, 2.4741, 2.4593, 2.4603, 2.4619, 2.4489]  # 2.61 → 2.45, sampled every ~1,500 steps (step 100 … 14,700)
  result: {gates_vs_gen1_by_games_seen_k: {109: 0.569, 218: 0.571, 288: 0.616, 327: 0.603, 437: 0.667, 575: 0.647}, null_control: 0.500, ckpt: selfplay/hp-g282k-*}

- id: EXP-003
  nature: "Is lr 2e-5 leaving strength on the table, and does batch size matter? Five adaptive points at a fixed 5,582-step budget, all on the same 287k pool."
  params: {games: 287527, steps: 5582, base: gen1, optim: "Adam wd=0", points: "A1 lr1e-4/b1024 · A2 adaptive · B1 b2048 lr×2 (linear) · B2 b256 lr/4 · C best at a full epoch"}
  loss: {A1_lr1e-4: [2.5729, 2.4971, 2.4816, 2.4712, 2.4654, 2.4638], A2_lr2e-4: [2.5712, 2.5038, 2.4901, 2.4790, 2.4752, 2.4750]}   # every 1,000 steps (step 100 … 5,100)
  result: {gate_vs_gen1: {anchor_snap5582_lr2e-5_b1024: 0.571, A1_lr1e-4_b1024: 0.625, A2_lr2e-4_b1024: 0.554, B1_b2048_lr1e-4_bf16c_1ep: 0.644}, A1_regate_new_protocol: 0.589, ckpt: selfplay/hp-lr*-e1.pt}
  note: "LR peak at 1e-4 (2e-5 too small, 2e-4 too big); B1 (batch 2048, lr 1e-4, 1 epoch, bf16+compile) = 0.644 in 33 min = 3,814 samples/s vs 2,660 for fp32/b1024. CAVEAT (2026-10-14): A1's re-gate on the rebuilt binary + compiled eval gave 0.589 vs 0.625 (wave 0 bit-identical, wave 1 0.647→0.575), so cross-protocol deltas below ~0.04 are not trustworthy. Harmonise by re-gating key points on ONE box+binary+eval before trusting the ranking."

- id: EXP-004
  nature: "Can a fresh net (rope2d trunk + RopePolicyHead, random init) be trained from zero on one 287k-game self-play round alone?"
  params: {games: 287527, epochs: 3, lr: 1.0e-4, batch: 2048, steps_per_epoch: 3677, base: "random init", precision: "bf16+compile", aux_w: 0.0}
  loss: [3.7251, 3.0302, 2.8509, 2.7896, 2.7376, 2.6979, 2.6592, 2.6392, 2.6097, 2.5633, 2.5219, 2.4933]   # e1..e5, ~900-step samples then each epoch end
  result: {gate_vs_gen1: {e1_3677: 0.052, e2_7354: 0.117, e3_11031: 0.291, e4_14708: 0.384, e5_18385: 0.510}, ckpt: selfplay/hp-scratch-rope-e*}
  note: "HEADLINE: a random-init rope2d + RopePolicyHead net reaches gen1 PARITY (0.510 vs 0.500) on one 287k-game self-play round alone — 5 epochs, 37.7M samples, ~3.2 h on one 3090, no human-supervised bootstrap. Increments: +0.065, +0.174, +0.093, +0.126 (not monotone); e5 waves 0.495/0.525 straddle 0.500, so this is parity within the +-0.030 gate noise, not a win. The loss is a weak proxy throughout (2.610 -> 2.493 while the gate went 0.117 -> 0.510). Still ~0.11-0.13 below the fine-tuned lineage (0.62-0.64). Stockfish 14.1 ladder (e5, 60 games/rung, movetime 200, sims 256): 1300/0.863, 1550/0.775, 1800/0.344, 2050/0.125, 2300/0.049 -> crossover ~1710 Elo, rung-implied 1688-1785 (mean 1737 excluding the 1300 rung) — i.e. statistically indistinguishable from the documented same-setting gen1 reference (~1730), ~100 below gen2 (~1835). Independent confirmation of the gate parity."

- id: EXP-005
  nature: "Does multi-directional (spiral) axial RoPE beat the file/rank-only axial form, from scratch on one self-play round? (Spiral RoPE, ICML 2026: spread the rotary channels over uniformly distributed directions instead of 2 axes.)"
  params: {games: 287527, epochs: 3, samples: 22600000, lr: 2.0e-4, batch: 4096, steps_per_epoch: 1839, base: "random init", pos: "rope2d dirs=4 (0/45/90/135deg), pairs=2 (16 of 32 dims rotated), lambda0=8", precision: "bf16+compile", aux_w: 0.0, gpu: "RTX 5090"}
  loss: [3.7108, 3.0828, 2.8440, 2.7460, 2.6959, 2.6786, 2.6256, 2.6135, 2.6179, 2.5687]
  result: {gate_vs_gen1: {e1_1839: 0.136, e2_3678: 0.360, e3_5517: 0.637}, ckpt: selfplay/hp-spiral-b4096-e1/-e2/-e3.pt, throughput: "0.62 s/step = 6,606 samples/s"}
  note: "0.637 at 3 epochs (22.6M samples): above gen1 (0.500) and level with the fine-tuned lineage (e1 0.616, B1' 0.644) — from random init, no human data. CAVEAT: vs the from-scratch axial control (0.052/0.117/0.291 at dirs=2 pairs=4 lam=16, bs 2048 lr 1e-4) this arm changes TWO things — geometry AND operating point (bs 4096 + lr 2e-4, and lr was earlier the single biggest lever). Data seen is matched (3 passes over the same 7.53M positions). To attribute the gain, re-run the axial control at bs 4096 / lr 2e-4 for 3 epochs; a lam=16-vs-8 arm at the same settings would isolate the ladder too."
```

## Restart on a fresh box

1. Bring up the `chess-train` group; update `SSH_HOST`/`SSH_PORT` in `.env`.
2. `scripts/salad.sh put runs/hp-setup.sh /workspace/` then `bash /workspace/hp-setup.sh`
   — fresh images still bake the legacy `runs/policy_value_net_42k.pt`; this pulls
   Gen1 from HF to `runs/gen1.pt` and repoints `ckpt_a/net.pt` at it.
3. `scripts/salad.sh put scripts/sp_round_train.py /workspace/chess/scripts/`
   — the image's copy predates the ingest fix, the provenance fix and the
   bf16/compile switches. `put chess_ai/net/nn_worker.py
   /workspace/chess/chess_ai/net/` too if the box should gate with the new
   **compiled** eval default (images only get it after the next rebuild).
4. Runs, in order (each: `put` the pair, then `bash /workspace/hp-launch-<tag>.sh`):
   **B1'** `hp-run-b2048-lr1e4-bf16c.sh` — batch 2048 @ lr 1e-4 with bf16 +
   torch.compile on (smoke it first with `hp-smoke-bf16c.sh`, 20 steps) →
   **B2** `hp-run-b256-lr2p5e5.sh` (only if B1 isn't clearly worse) → **C** the
   winning recipe at a full epoch (7,528,579 samples ÷ batch, vs `e1` = 0.616).
   First ingest re-parses the units (~10-13 min; the old box's `packed-cache/` is gone).
   Precision bookkeeping: A1/A2/B1 were trained **fp32** (the flags didn't
   exist yet) and gated with bf16-but-uncompiled eval; from B1' training is
   bf16+compiled. To keep the comparison honest, **re-gate `hp-lr1e4-b1024-e1.pt`
   (HF, = A1 = 0.625) on the same box under the new compiled eval**.

