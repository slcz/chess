# chess_ai/selfplay/ — self-play data path

**Current (branch `search`)**: `engine/build/selfplay` (engine/src/selfplay.c)
generates the data — a pool of games (default 1024) thinks in lockstep on one
`PuctMux`, leaf evals of every game batched on the GPU through the embedded
net. Grill et al. (2020) regularized recipe, all three variants: the descent
samples the improved policy π̄ at every node (SEARCH); the played move and
the stored target are the root π̄ (ACT + LEARN) — sampled for the first
`sample_plys` plies (default 10), argmax after. No root Dirichlet noise and
no PUCT argmax path: π̄'s λ_N schedule carries the exploration and its
decay. Recorded per ply: the played move, the ply's actual search budget,
and the **dense π̄ over every legal move** (zero-visit moves included,
temperature-free), plus the mover-view root value (`outlook`; future
resignation / loss-weighting input). Games end naturally or at a ply cap
(adjudicated draw) — no resignation yet.

**Playout cap randomization** (KataGo, Wu 2020 §4): with `fast_sims` > 0
the driver draws ONE budget per mux step for the whole pool — `fast_sims`
w.p. `fast_p`, else the slow `sims`. Per-step (not per-move) because the
pool is lockstep: a per-move draw would make every step wait on the
slowest game; per-step keeps the KataGo per-ply marginal (each ply
independently slow w.p. 1−fast_p) with full-pool GPU batches. Every ply
records its budget; `selfplay_data.ply_trainable` keeps only slow-cap
plies (sims == game n_sims) as training samples — fast plies are replayed
through (no per-ply FENs are stored) and still serve as aux-reply targets.
`fast_sims=0` (default) is uniform-budget, off.

Output is one work-unit directory per run: `shard-0.spbin` (SPSHARD1 v3 —
byte format authority: engine/include/chess_sp_dump.h; π is dense f16
probs over the legal moves in engine movegen order, no slot ids;
self-delimiting records, so a killed run costs at most a truncated
trailing game) plus `DONE.json`. This package reads that back:
`shard.read_unit()` → `SpGame` records; `chess_ai.train.selfplay_data`
converts records to training samples (dense `pi_slots`/`pi_probs`, width
256 > the 218 legal-move bound, so nothing is truncated) that
`Trainer.train_step`'s dense-π branch consumes (target-weighted CE — with
a one-hot π this is exactly the sparse path). End-to-end contract pinned
by tests/test_selfplay_roundtrip.py. Readers are v3-only — legacy v1/v2
shards must go through scripts/spbin_to_v3.py first.

Deviations from the Gumbel design below (deliberate, branch `search`): the
exact regularized policy π̄ (Grill et al. 2020 — the fixed point the Gumbel
trick approximates) instead of Gumbel top-m candidate subsampling; π̄ keeps
an AGZ-style temperature schedule (sample the first `sample_plys` plies,
argmax after) where Gumbel needs none; startpos-only games. Revisit per
PLAN.md if evidence says otherwise.

**Browse a unit**: open `shard-0.spbin` in the web GUI's Walkthrough tab
(web/, the wasm-serverless branch) — dataset index (plies,
result, ending, opening move; filter box), board stepper (←/→/Home/End),
and a per-ply panel with the search's π bars (played move gold) and root
value. With `--ckpt`, the current position's top-3 moves are heat-colored
on the board (dark = highest π/prior) and hovering a violet slot shows π
visits, the net's prior, and the net's value of the resulting position
(child WDL flipped to the mover's view, terminal children exact) —
machinery reused from supervised/play.py's ValueStrategy. Pure replay
otherwise: no engine search, works offline. Contract pinned by
web/app/shard.js.

## Search algorithm (Phase 3.2 — agreed 2026-09-23, implemented on `main`)

Gumbel AlphaZero (Danihelka et al., ICLR 2022). n = 2–32 simulations per move. **No
Dirichlet noise, no move-selection temperature** — candidate sampling via Gumbel-top-m
and a visit-scaled σ take over both exploration jobs. (Authoritative copy + training
targets + discussion: PLAN.md, Phase 3, "Agreed search design".)

```text
GUMBEL_SEARCH(s, n):
    m      = min(n, 16, #legal)             # candidates; m <= n => all visited in round 1
    g(a)   ~ Gumbel(0,1)  per legal action
    z(a)   = logits(a) + g(a)
    A      = top-m actions by z(a)
    rounds = ceil(log2 m)                   # from the ORIGINAL m — do not recompute
    while |A| > 1:
        plays = ceil(n / (rounds * |A|))    # split the round's slice over survivors
        for a in A: plays x PUCT-descent(child(s,a)); back up, sign-flip every ply
        A = top ceil(|A|/2) of A by [ z(a) + (C_VISIT + max_b N(b)) * C_SCALE * q(a) ]
    return the single survivor              # the move; no visit sampling, no tau
                                            # C_VISIT=50, C_SCALE=1.0
```

Training target for the policy head is NOT the visit distribution (AGZ) but the
improved policy `pi' = softmax(logits + sigma(completedQ))` with
`completedQ(a) = q(a) if N(a)>0 else v_hat(s)` — distilled as KL (i.e. plain
target-weighted cross-entropy). Value target stays the game outcome z.

