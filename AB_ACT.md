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
# both arms boot the same pointer, so both are named g17-5249f2f9; the unit
# epoch is the only discriminator (A ends 1792609860, B starts 1792621439)
data: selfplay-scratch/units/g17-5249f2f9-i*  6 units, 46 parts, 49790 games
  units: 1792608236 9633 | 1792608897 8569 | 1792609110 8533
         1792609142 8588 | 1792609427 8349 | 1792609860 6118
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
data: selfplay-scratch/units/g17-b-5249f2f9-i*  6 units, 48 parts, 50796 games
  (generated as g17-5249f2f9-*, renamed server-side so the arm is identifiable
   by path alone; RUN derives from the checkpoint, not the policy)
  units: 1792621439 9662 | 1792622096 9607 | 1792622160 9328
         1792622351 9448 | 1792622431 4166 | 1792622535 8585
train:
  positions: ~1.61M (slow-only) — 394 steps
  final: {loss: 2.8364, policy: 1.8347, value: 0.5604, aux: 1.4709}
  acc: {policy: 0.707, value: 0.740}
gate_vs_gen17:
  games: 400
  wdl: [151, 107, 142]
  score: 0.511
  elo: +7.7
  sigma: 17.6
gate_vs_A:                       # head-to-head, gate seed 9 — the decision
  games: 400
  A_wdl: [122, 108, 170]
  A_score: 0.440
  B_elo_over_A: +42.4
  sigma: 17.5
sf_nodes:
  "300": 0.905
  "500": 0.832
  "1000": 0.580
  "1500": 0.350
  "2000": 0.355
  "3000": 0.110
  note: crossover ~1150; better than A at 500, worse at 1000/3000 — within
        the +-5-7% per-rung noise, so no clear absolute difference
```

## Arm C — SEARCH + ACT + LEARN (all three variants)

```yaml
arm: C
policy: pi-bar is the target (LEARN), the played move (ACT) and the per-node
        selection distribution (SEARCH); no Dirichlet noise
image: ghcr.io/slcz/chess-value@sha256:60ac3983bc7671040aabde4ad5151d28099cfda3b02a2b9f3f80cb8883dcef70
data: selfplay-scratch/units/g17-5249f2f9-c-i*  7 units, 55 parts, 49150 games
ckpt: selfplay/c-e1.pt            # pushed by the trainer; NOT adopted
train:
  steps: 395
  final: {loss: 2.7921, policy: 1.6998, value: 0.6339, aux: 1.5279}
  acc: {policy: 0.673, value: 0.695}
gate_vs_gen17:
  games: 400
  wdl: [184, 101, 115]
  score: 0.586
  elo: +60.4
  sigma: 17.6
sf_nodes:
  "300": 0.930   "500": 0.835   "1000": 0.635
  "1500": 0.435  "2000": 0.350  "3000": 0.220
  note: new baseline — measured under SEARCH, not comparable to earlier ladders
sf_floor:
  "1350": 0.983  "1600": 0.933  "1850": 0.592  "2100": 0.250
  implied_elo: [1854, 2058, 1915, 1909]   # ~1900, rung noise ±50-60
  note: gen17 measured ~1840-1850 under the old search, so the +60 from the
        gate is corroborated by the ladder
```

### Head-to-head matrix (400 games each, gate seed 9)

| A (visits) | B (ACT) | C (ALL) |
|---|---|---|
| — | B over A: **+42 ± 18** | C over A: **+15 ± 17** (n.s.) |
| | — | C over B: **+43 ± 18** |

C vs gen17: **+60 ± 18** (3.4 sigma) | A vs gen17: +22 ± 18 | B vs gen17: +8 ± 18

The matrix is **not transitive**: C > B (2.5σ) and B > A (2.4σ) would imply
C > A by ~85, but C vs A measured +15 (0.8σ). Either one gate is off by ~2σ
or the nets genuinely form a cycle (rock-paper-scissors on the fixed book),
which is plausible for nets trained on differently-shaped self-play data.
C is the best arm on the shared base; its edge over A specifically is not
established at 400 games.

| arm | policy | data | gate vs gen17 |
|---|---|---|---|
| A | visits play, π̄ target | 49,790 | +22 ± 18 |
| B | π̄ play (ACT), π̄ target | 50,796 | +8 ± 18 |
| **C** | **π̄ selection (SEARCH) + π̄ play + π̄ target** | **49,150** | **+60 ± 18** |

C is 3.4σ clear of parity, and its gain matches a normal production
iteration (+59 for gen17 over gen16) — from ~49k fresh games with no replay
carry-over. That is the paper's prediction: the three variants combined beat
each individually.

## Verdict

Across the three arms, only C clears its own noise: **+60 ± 18** against the
shared base, where B was +8 ± 18 and A +22 ± 18. Before C, the reading was
that ACT data beats visit-policy data **head-to-head: B over A by +42 ± 18**
(2.4σ, 400 paired games at gate seed 9). Against the shared base the two arms
are both near parity (A +22 ± 18, B +8 ± 18), and the node ladder sees no
clear absolute difference between them at 100 games/rung.

Honest reading: the three gates are each ±18 and are mildly inconsistent
with each other (A > gen17 by 22 while B > A by 42 would imply B > gen17 by
more than A, which it is not; that is ~1.8σ of spread). So "ACT data is
better per game generated" is *likely*, not established. What would settle
it: the same A-vs-B pair at 800-1000 games, or a second independent pair of
rounds.

Neither checkpoint is adopted: both train on ~50k games with no replay, so
they sit near gen17. The pointer stays at gen17. If ACT is adopted as the
recipe, the next step is a normal round — ACT data plus the standard
1/4-trim buffer — to see whether it clears gen17 by more than the usual
+40-60.
