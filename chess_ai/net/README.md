# chess_ai/net/ — feature encoding + the network (Phase 2, built)

Two modules, shared by every consumer (self-play, training, eval):

- **`features.py`** — the policy move space and the input-tensor spec.
  - **Move space (decision D1, settled 2026-09-24: 64×64 + promotions,
    4240 slots)**: 4096 from-to pairs in mover-relative coordinates
    (covers all normal moves, captures, ep, castling, and queen
    promotions — "pawn arrives on last rank" implies "=Q"), + 72 own
    underpromotion slots (rel-rank-7 pawn, 3 directions × N/B/R), + 72
    mirrored *enemy*-promotion slots (rel-rank-2 pawn moving downward in
    our frame) that exist for the aux opponent-reply head's targets.
  - **Reference encoder**: a pure-Python re-implementation of the C
    encoder from the spec (`engine/include/chess_nn.h`), driven only by
    FENs and the public Board API. `tests/test_nn_features.py` asserts
    C and reference agree tensor-for-tensor over random games.
- **`model.py`** — the net, per the "Agreed net design" block in
  PLAN.md Phase 2 (recipe: arXiv 2409.12272 CF-6M):
  - Encoder-only transformer trunk, **8 blocks × d256 × 8 heads ≈ 6.5M
    params**, pre-LN, GELU FFN (4× width)
  - **Shaw relative position embeddings** (clip k=16) in attention
    scores *and* values — replaces absolute square embeddings (their
    ablation: relative wins for chess). Residual branches are
    zero-initialized (GPT-2-style) so the trunk starts as the input
    embeddings; the 2026-09-24 gate debugging showed standard init makes
    8 layers of near-uniform attention wash out cross-token identity.
  - **Policy head** (`NetConfig.policy_head`, both emit the 4240-slot
    distribution with illegal slots masked to probability **exactly
    zero**):
    - **`pooled` (run-1 default, user decision 2026-09-24)** —
      AlphaZero/Lc0-style: mean-pool the 64 tokens → Linear(256→4240).
      Robust (every move's logit is an independent weight row, no
      representational ties), ~1.1M params.
    - **`attention` (A/B knob)** — CF-paper bilinear from-square ×
      to-square head, ~140k params, geometric sharing. Requires Shaw
      relpos *in the head's scores* (`rel_k`): quiet moves end on empty
      squares, and squares empty for the whole 8-ply window have
      byte-identical tokens, so plain q·k assigns e2e3 and e2e4 exactly
      equal logits — untrainable. (The gate debugging traced this out.)
  - **Value head**: WDL 3-way (scalar v = pW − pL for the search); the
    head itself mean-pools → MLP → 3 logits.
  - **Aux opponent-reply head** (KataGo adoption, `NetConfig.aux_reply`,
    on by default): same head class predicting the enemy's reply policy
    in OUR mover-relative frame (mirrored promotion slots exist for it
    in the move space).
  - Fallback knobs kept out of run 1 per the plan: post-LN + DeepNet +
    Mish recipe, soft policy target, value-error head.

Sanity gate (PLAN Phase 2, redesigned 2026-09-24): the net *learns* a
fixed structured fixture — random-vs-greedy policy labels (greedy side
only) + material-sign W/D/L value labels, bars ≥ 0.65 / ≥ 0.92
(`tests/test_overfit.py`; the docstring tells the full design story,
including why uniform-random labels rote-stall) — and illegal-move
masking is asserted (`tests/test_net_model.py`).
