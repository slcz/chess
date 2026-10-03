# chess_ai/tinygame — Phase 3.5, the tiny-game smoke test

Tic-tac-toe re-instantiates the **entire Phase 3 learn loop** at a scale
where "perfect play" is exactly checkable: Gumbel self-play search →
dense-π′/WDL records → rolling replay window → target-weighted-CE
training → the trained net feeds the next self-play round. The gate
(PLAN §4/§10.5): the deployed player must play value-preservingly in
**100% of all reachable positions** (checked against an exhaustive
minimax oracle), reached **in minutes, locally, before any cloud spend**.

This package deliberately shares no code with the chess play path —
3.5 validates the *learn loop*, not the C engine (PLAN 3.5). What it
reuses from the chess pipeline is the *algorithm* and its pinned
conventions, ported line-for-line:

| Module | Chess counterpart | What carries over |
|---|---|---|
| `game.py` | `chess_ai.engine.Board` (interface subset) | push/pop, legal-move enumeration order as the RNG/tie-break spec, mover-relative encode, WIN≈CHECKMATE / DRAW terminal semantics |
| `game.py` oracle | — (chess has no oracle) | exhaustive negamax → exact value + optimal-move set per position; makes the gate an exact predicate, not a statistic |
| `search.py` | `selfplay/gumbel.py` | the whole GUMBEL_SEARCH: Gumbel top-m candidates, sequential halving (candidate-major, `plays = ceil(n/(rounds·|A|))`), Eq. 14 below the root with the Eq. 10 fallback, σ(q) = (c_visit+max N)·q, π′ targets, sign conventions, F4 handoff, root-level resignation (R4) |
| `search.py` RNG | `selfplay/rng.py` | **imported verbatim** — the same splitmix64/xoshiro256++ stream, seeding, uniform/Gumbel maps, and consumption order |
| `net.py` | `net/model.py` contract | masked log-softmax (illegal slots at exactly −inf), WDL + aux-reply heads behind the same forward signature |
| `net.py` NumpyEvaluator | C→Python evaluator seam | self-play inference on a weight snapshot synced per round |
| `data.py` | `train/selfplay_data.py` | dense-π′ sample schema (first-slot padding, renorm), resign ply contributes no sample, aux reply in our frame, rolling buffer with uniform-with-replacement sampling |
| `loop.py` TinyTrainer | `train/trainer.py` | the dense-π′ branch of `train_step` line-for-line (target-weighted CE, WDL CE, aux CE, Adam 3e-4, clip 1.0) |
| `loop.py` gate | — | perfect-play eval: deterministic Gumbels-off search (`sample_plies=0`, `m=k`) over every reachable position; a resignation only counts if the position is truly lost |

## How to run

```bash
# the official smoke (defaults below; exits 0 iff the gate is met)
python -m chess_ai.tinygame.loop --out runs/tinygame/smoke

# tests (incl. a CI-budget convergence test that must reach 100%)
pytest tests/test_tinygame.py
```

Default configuration (the measured recipe; every knob in
`SmokeConfig`): n=16 Gumbel sims/move in self-play with **c_visit=5 and
∝visit-count move sampling**, 384 games/iteration, 120 training
steps/iteration at batch 256, LR 3e-4 annealed ×0.99, value_weight 3,
whole-run replay window, gate player = Gumbels-off search at 512 sims
with c_visit=5, light eval on a 256-position critical subset every 2
iterations with the full 4520-position gate eval when the light gate is
clean. `c_visit` and the sampling mode are per-game knobs the smoke test
measured (see Results); chess run-1 A/Bs are flagged in PLAN.md.

## Findings (2026-09-29)

1. **The search port is exactly correct independent of any net** — an
   oracle evaluator (exact values, optimal-move priors) driven through
   the same search + gate metric scores 100.0% over all 4520 reachable
   decision positions (`test_oracle_evaluator_search_is_exactly_perfect`).
   Every gate miss is therefore a net-quality fact, never a search bug.
2. **Startpos-only self-play self-confirms on the last ~2%** (measured:
   plateau at 97–98.5% with value-head oracle accuracy stuck ~65%): the
   policy avoids its own blind spots, so those positions never appear in
   the buffer, their deep values never get outcome labels, and π′ there
   stays wrong. Deeper eval search does NOT fix it (98.6 → 99.2% at
   n=2048). Fix: **diverse game initialization** — each self-play game
   starts after a uniformly-random legal prefix of 0..`random_start_plies`
   plies (KataGo's side-position init, adapted). This is the one addition
   over the chess defaults; it lifted value accuracy 61 → 94%+ and play
   to ~99.7%. Chess adoption is a separate decision for the run-1 config.
3. **The plan's Gumbels-off eval variant is load-bearing here**: with
   per-position Gumbel draws at eval, near-tie survivors flip and the
   gate loses ~0.5 pt to measurement noise. `gumbels_off=True` (z = logp,
   no RNG) is the deterministic strength measurement PLAN Phase 3 names.
4. **Apple Accelerate FP flags**: numpy's Accelerate-backed matmul raises
   spurious divide/overflow/invalid flags on this workload; outputs are
   bit-identical to torch's `F.linear` (pinned by test). `NumpyEvaluator`
   masks the flags with `np.errstate`.
5. Resignation (-0.90) occasionally spirals early training when the value
   head is still noisy (false resignations poison z labels). The chess
   R4 design accepted this at chess scale; for the smoke the default
   keeps it ON (it recovers), and the knob is there.
6. **AlphaZero's thesis, measurable**: the gate was met with the raw net
   at policy-only 97.3% and value-head accuracy 70.4% — the deployed
   search (prior + σ(q) over net values at 512 sims) lifts it to 100%
   value-preserving play. The search is doing real work over the net,
   exactly the property the loop is supposed to cultivate.

## Results — GATE MET (2026-09-29)

The official run (`python -m chess_ai.tinygame.loop`, defaults):
**100.00% value-preserving play over all 4520 reachable decision
positions** (critical subset 100.00% of 3191, 0 resign blunders) reached
at **iteration 24, 127 s wall** (~9.2k self-play games) — exit code 0.
Artifacts: `runs/tinygame/smoke/{net.pt,metrics.json}`. A diagnosed
oracle evaluator through the same search scores 100% by construction
(`test_oracle_evaluator_search_is_exactly_perfect`), so the gate number
is a statement about the learned net + search only.

### The bring-up story (what the smoke test actually measured)

Progression of the full-oracle gate across ~15 recipe variants, each
finding is a knob the plan reserved for this test:

| Recipe change | Gate (all positions) |
|---|---|
| startpos-only self-play (chess defaults) | 97–98.5%, hard plateau |
| + diverse game initialization (prefix ≤ 8) | ~99.7%, plateau |
| + keep-all buffer / value_weight 3 / long anneals | 99.7–99.76%, plateau |
| + prior-led self-play (self-play c_visit 5) | 99.95%+ (de-biases labels) |
| + **visit-count sampling** (∝N, Gumbel-paper Fig. 8b mode) | **100.00% @ iter 24** |

The last two knobs were needed for one mechanistic reason, diagnosed by
dumping the net's values along every candidate line of each failing
position: **the outcome label at a drawn position is one-sided** (the
attacker wins or draws, never loses), so the value head learns
outcome-under-current-play and stays pessimistic about the defender
exactly where the defender must find the only defense. With the paper's
c_visit=50, σ(q) amplifies that residual bias (~0.02–0.05 in q) into
3–9-nat logit swings that override the (correct, generalized) policy
prior — and π′ sampling then reproduces the poisoned values in the
targets and the played moves: a closed loop. Volume doesn't break it
(a 16×-volume test went *backwards*); deeper eval search fixes 4/17;
the unwinding came from (a) prior-led self-play so the defender holds
draws and the labels clean up, and (b) ∝visit-count sampling — flat in
the value signal — so the correct defense keeps ~1/k play probability
while the trap is mispriced.
