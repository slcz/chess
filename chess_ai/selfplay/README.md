# chess_ai/selfplay/ — self-play data path

**Current (branch `puct_mcts`)**: `engine/build/selfplay` (engine/src/selfplay.c)
generates the data — a pool of games (default 1024) thinks in lockstep on one
`PuctMux`, leaf evals of every game batched on the GPU through the embedded
net. AGZ-style exploration: root Dirichlet noise (eps 0.25, alpha 0.3) on;
move choice samples ∝ visits for the first `sample_plys` plies (default 30),
argmax visits after. Recorded per ply: the played move and the **full visit
distribution** (π ∝ N over visited moves, temperature-free), plus the
mover-view root value (`outlook`; future resignation / loss-weighting input).
Games end naturally or at a ply cap (adjudicated draw) — no resignation yet.

Output is one work-unit directory per run: `shard-0.spbin` (SPSHARD1 —
byte format authority: engine/include/chess_sp_dump.h; self-delimiting
records, so a killed run costs at most a truncated trailing game) plus
`DONE.json`. This package reads that back: `shard.read_unit()` →
`SpGame` records; `chess_ai.train.selfplay_data` converts records to
training samples (dense `pi_slots`/`pi_probs`, width 256 > the 218
legal-move bound, so nothing is truncated) that `Trainer.train_step`'s
dense-π branch consumes (target-weighted CE — with a one-hot π this is
exactly the sparse path). End-to-end contract pinned by
tests/test_selfplay_roundtrip.py.

Deviations from the Gumbel design below (deliberate, this branch):
PUCT + Dirichlet instead of Gumbel top-m (no noise-free exploration);
π = visit distribution instead of softmax(logits + σ(completedQ));
startpos-only games. Revisit per PLAN.md if evidence says otherwise.

**Browse a unit**: `scripts/sp_replay.py <unit_dir> [--ckpt net.pt]` serves a
replay browser (replay.py + static/replay.html) — dataset index (plies,
result, ending, opening move; filter box), board stepper (←/→/Home/End),
and a per-ply panel with the search's π bars (played move gold) and root
value. With `--ckpt`, the current position's top-3 moves are heat-colored
on the board (dark = highest π/prior) and hovering a violet slot shows π
visits, the net's prior, and the net's value of the resulting position
(child WDL flipped to the mover's view, terminal children exact) —
machinery reused from supervised/play.py's ValueStrategy. Pure replay
otherwise: no engine search, works offline. Contract pinned by
tests/test_sp_replay.py.

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

