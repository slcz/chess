# chess_ai/selfplay/ — batched self-play

The throughput core (Phase 3): a Python outer loop over N games in lockstep — C advances each game until it needs an NN evaluation and returns packed input planes for pending leaves; Python runs one batched forward pass per round; priors + values return as arrays. Also home to the game recorder, the replay-buffer writer, and the checkpoint-ladder match runner (net@A vs. net@B).

**Architecture (D7, 2026-09-27 — C-orchestrated parallel):** N−1 C MCTS
worker threads + 1 evaluator thread (PyTorch via embedded CPython); all
sync at the batch boundary. Design + review rounds + verification ladder:
`PARALLEL_DESIGN.md`. Status: 3.2a (sequential Python reference) done;
3.1 (C descent core, sequential mode) done — L2-exact vs the reference;
**3.3b (parallel engine) done (2026-09-28)** — with a batch-invariant
evaluator the parallel engine reproduces the sequential games bit-for-bit
(any schedule dependence would be a data race), interleave goldens green,
and the torch net runs behind the embedded-CPython evaluator.

## Layout

- `gumbel.py` — the sequential Python reference (3.2a): the executable spec
  of the search and the comparison oracle for the C engine. Synchronous
  size-1 evals, no queues — literally PLAN's pseudocode plus one
  `Evaluator.evaluate` call.
- `rng.py` — the RNG spec (splitmix64 seeding, xoshiro256++, uniform/gumbel
  mapping, consumption order). Mirrored line for line in C
  (`engine/src/rng.c`); stream parity is verified over the process boundary.
- `csearch.py` — the Python driver over the C search (3.1): runs the same
  game with the same evaluator and exports the same `GameRecord` shape, so
  `tests/test_search_l2.py` can diff the two engines field for field. The
  request/deliver protocol here is the 3.3b worker/evaluator split in
  miniature.
- `spengine.py` — the parallel-engine seam (3.3b): one blocking call runs
  N−1 C workers + the embedded-CPython evaluator over `game_budget` games
  and returns GameRecords; telemetry included. Loads `libchess_sp` (build:
  `make -C engine build/libchess_sp.dylib`).
- `sp_evaluator.py` — the Python side of the batch evaluator: the torch
  net (or batch-invariant fixtures) behind `eval_batch`, fed memoryviews
  over the engine's staging by `engine/src/sp_evaluator.c`.

## Search algorithm (Phase 3.2 — agreed 2026-09-23)

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


