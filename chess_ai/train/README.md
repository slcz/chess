# chess_ai/train/ — data pipeline + training loop (Phase 2, built)

- **`data.py`** —
  - `GameRecord`: start FEN + UCI move list + outcome — the compact
    self-play unit. Tensors are NOT stored (30 KB/position would make a
    500k-position buffer ~15 GB); they are rebuilt on demand by
    replaying the record through the C encoder.
  - Sample schema: tokens `[64,117]`, legal-move mask `[4240]`, policy
    target as a sparse slot index (the Phase-2 stand-in for π′ — Phase 3
    swaps in the real search-improved target without a schema change),
    WDL one-hot from the mover's view, and the KataGo aux-reply target
    (next move + reply mask, both in the current mover's frame; zero
    weight on each game's last ply).
  - `ReplayBuffer`: rolling window of recent games (40–50k at full
    scale), uniform position sampling.
  - **Structured fixtures** (Phase 2 gate, designed with the user
    2026-09-24): `generate_strategy_games` plays e.g.
    **random-vs-greedy** (GreedyMaterialStrategy vs RandomStrategy);
    `build_tensors(..., mover_filter=0)` trains only on the strategist's
    moves — labels become a learnable function of the position instead
    of arbitrary (uniform-random labels are pure rote and stall a 6.8M
    net at the uniform plateau). The gate's value target is
    `material_sign_wdl` — a deterministic per-position label, since the
    *game outcome* under random-vs-greedy carries irreducible noise
    ("mate before the shuffle" depends on future random moves).
  - `generate_random_games`: seeded uniform-random fixtures (kept for
    encoder-parity tests; never real training data — zero-knowledge).
- **`trainer.py`** — the AGZ-shape loss (policy CE to π′ + WDL
  cross-entropy + weighted aux-reply CE), Adam + grad clipping, MPS →
  CUDA → CPU device auto-pick, `fit()` over a fixed tensor set (the
  memorization-gate regime), and **atomic checkpointing** (tmp + rename,
  model + optimizer + step + configs in one file) per the salad
  reliability rules (PLAN §5). `Trainer.load` resumes bit-identically
  (asserted in tests). Since 3.4 the policy CE takes **two batch forms**:
  sparse `pol_idx` (Phase 2 fixtures / supervised) or dense π′ over the
  candidate support (`pi_slots` + `pi_probs` from self-play) — with a
  one-hot π′ the two are identical by construction (tested).
- **`selfplay_data.py`** (Phase 3.4) — self-play records → the training
  format, the trainer-side half of the game recorder:
  - Input = gumbel.py GameRecords, either straight from a run or read
    back from the engine's shard dumps (`read_unit` consumes a work
    unit's `DONE.json` manifest + `shard-w*.spbin`; `ingest_units` unions
    many units, deduped by (seed, game_index)).
  - Sample schema = data.py's plus the **dense π′ target**: support
    slots + renormalized probs, fixed width K = 16 (= the C record
    layout's `SP_REC_CANDIDATES`), padded by repeating the first real
    slot (never a blind slot-0 pad — `policy_logp` is −inf at illegal
    slots and −inf·0 = NaN). The **resign ply contributes no sample**
    (a search happened but no decision completed; the resigner-loses z
    label still reaches every earlier ply), and the ply before a
    resignation loses its aux-reply target.
  - `SelfPlayBuffer`: the replay window over self-play records (stores
    records only, ~9 KB/game; uniform-over-positions sampling with
    replacement, one board walk per picked record, same convention as
    Phase 2's buffer).

Phase 3 replaces `fit()` with streaming batches from the replay buffer;
`train_step` is the shared core. The salad-node binary that *produces*
the units (`engine/build/chess_selfplay`) is documented in
engine/README + PARALLEL_DESIGN §8.
