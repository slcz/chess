# A/B: ACT vs visit-policy self-play data

Testing Grill et al. (ICML 2020) ACT — playing the move sampled from the
regularized policy π̄ instead of from the visit distribution — as a
*data-generation* change. Both arms start from the same checkpoint, differ
only in how their games were generated, and are compared before either is
adopted.

The live pointer stays where it is: nothing here moves
`selfplay-scratch/approved.json`.

## Protocol

| | |
|---|---|
| base checkpoint | gen17 (`selfplay-scratch/gen17.pt`, sha 5249f2f9cb4b) |
| round size | ~50k games per arm |
| training | fresh from the base on that round alone (no replay carry-over), 1 epoch, slow-only (n=300 plies), batch 4096, lr 2e-4, seed 1 |
| decision | head-to-head gate, A vs B, 400 games; adopt only if the winner also clears its own arm-vs-gen17 gate |
| rationale | training each arm on its own round only keeps the two runs symmetric, so the delta is attributable to the data policy. Both nets land near gen17 rather than above it, so the A/B — not the absolute gate — decides. |

## Arm A — visit policy (current recipe)

```yaml
arm: A
policy: sample ∝ visits for the first 10 plies, argmax visits after; target π = π̄ (LEARN only)
image: ghcr.io/slcz/chess-value@sha256:11e0197c522d439b820c3b213e82c63cf4a467454522cb3892169dbf372d40d2
data: selfplay-scratch/units/g17-5249f2f9-i*  (6 units, 46 parts, 49790 games, 8 templates)
ckpt: selfplay/nonact-e1.pt   # pushed by the trainer; NOT adopted
train:
  positions: ~1.43M (slow-only)
  steps: 349
  final: {loss: 2.8353, policy: 1.6851, value: 0.6746, aux: 1.5851}
  acc: {policy: 0.679, value: 0.686}
gate_vs_gen17:
  games: 400
  wdl: [162, 102, 136]
  score: 0.532
  elo: +22.3
  sigma: 17.6
sf_nodes:
  "300": 0.915
  "500": 0.740
  "1000": 0.690
  "1500": 0.480
  "2000": 0.350
  "3000": 0.208
  note: crossover ~1450 nodes at 300 sims (gen17 sat at ~1200-1300)
```

Reading: one extra epoch over a fresh 50k round of gen17's own self-play
data is worth about +22 ± 18 on the gate, and the node suite agrees (every
rung at or above gen17 except 500). That is the control arm's value — the
"do nothing different" number this A/B has to beat.

## Arm B — ACT (π̄ play policy)

```yaml
arm: B
policy: sample π̄ for the first 10 plies, argmax π̄ after; target π̄
image: pending — the ACT image is built but its push and the round launch
       were blocked (missing HF_MODEL_REPO in the container env)
data: pending
gate_vs_A: pending
```
