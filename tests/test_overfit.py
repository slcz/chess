"""
tests/test_overfit.py — THE PHASE 2 GATE (PLAN §4 Phase 2 gate, §10.6),
redesigned 2026-09-24 after the original random-rote fixture stalled.

DESIGN (user-driven): the original gate asked the net to memorize ~100
uniform-random games — but random-move labels have no structure, so
learning is pure rote whose speed scales with exposures-per-position; a
6.8M net sat at the uniform plateau for 1500 steps (the "stall"). The
fix is not capacity (27k positions × ~5 bits ≪ 6.8M params) — it is
*structured labels*:

  - 100 seeded games of **random-vs-greedy** (GreedyMaterialStrategy vs
    RandomStrategy, our own dummy strategies — zero-knowledge intact);
  - **policy target**: the greedy side's actual moves only (mover
    filter). "Take the best capture, else go central" is a learnable
    function of the position; the random opponent supplies diversity.
    Tie-break randomness caps top-1 accuracy (measured rote ceiling
    97.6%), so the bar is not 99%;
  - **value target**: material-sign W/D/L per position
    (data.material_sign_wdl) — a deterministic function of the position,
    populating all three classes. The game *outcome* under random-vs-
    greedy is deliberately NOT used as the value label: whether a game
    ends in mate or shuffles into threefold depends on the random side's
    future moves, i.e. irreducible label noise from any early position.

THRESHOLDS (calibrated 2026-09-24, 400-step run):
  - policy acc ≥ 0.65   (chance ≈ 0.03 for a broken pipeline; healthy
    runs reach ~0.5 by step 200 and keep climbing; ceiling 0.976)
  - value  acc ≥ 0.92   (class prior ≈ 0.86 — a value pathway that never
    learns sits exactly there forever; healthy runs reach 0.94+)
  - illegal-move probability exactly zero after training (the §9.3
    contract), and legal-mass ≈ 1.

Runs the REAL net (8×d256×8h, pooled head) on MPS; a reduced trunk and
fixture on CPU so the gate is runnable anywhere.
"""
from __future__ import annotations

import pytest
import torch

from chess_ai.strategies import GreedyMaterialStrategy, RandomStrategy
from chess_ai.net.features import POLICY_SIZE
from chess_ai.net.model import ChessTransformer, NetConfig
from chess_ai.train import (
    Trainer,
    TrainConfig,
    build_tensors,
    generate_strategy_games,
    material_sign_wdl,
)

GATE_SEED = 20260924          # fixed: a failure reproduces exactly
POLICY_ACC_TARGET = 0.65
VALUE_ACC_TARGET = 0.92
EVAL_EVERY = 100
MAX_STEPS = 1200

MPS = torch.backends.mps.is_available()


@pytest.fixture(scope="module")
def dataset():
    if MPS:
        games, model_cfg, batch = 100, NetConfig(), 128
    else:                       # CPU fallback: smaller fixture and trunk
        games, model_cfg, batch = 40, NetConfig(n_layers=4, d_model=128, n_heads=4, d_ff=512), 96

    records = generate_strategy_games(
        games,
        seed=GATE_SEED,
        strategy_factory=lambda s: GreedyMaterialStrategy(seed=s),
        opponent_factory=lambda s: RandomStrategy(seed=s),
        strategist_side=0,
    )
    tensors = build_tensors(records, mover_filter=0, value_fn=material_sign_wdl)
    n = tensors["tokens"].shape[0]
    print(f"\noverfit gate fixture: {games} games -> {n} strategist positions")
    return tensors, model_cfg, batch


def test_overfit_gate(dataset):
    tensors, model_cfg, batch = dataset
    torch.manual_seed(0)
    trainer = Trainer(
        ChessTransformer(model_cfg),
        TrainConfig(lr=2e-3, batch_size=batch, device="mps" if MPS else "cpu",
                    seed=0, aux_reply_weight=0.0),   # reply labels = random side = noise
    )

    final = {}
    for _ in range(0, MAX_STEPS, EVAL_EVERY):
        trainer.fit(tensors, steps=EVAL_EVERY)
        ev = trainer.evaluate(tensors)
        print(f"  steps {trainer.steps:5d}/{MAX_STEPS}  "
              f"policy acc {ev['policy_acc']:.4f}  value acc {ev['value_acc']:.4f}")
        final = ev
        if (ev["policy_acc"] >= POLICY_ACC_TARGET
                and ev["value_acc"] >= VALUE_ACC_TARGET):
            break

    assert final["policy_acc"] >= POLICY_ACC_TARGET, (
        f"policy learning stalled at {final['policy_acc']:.4f} "
        f"(broken pipeline ≈ 0.03; healthy ≈ 0.5 by step 200)"
    )
    assert final["value_acc"] >= VALUE_ACC_TARGET, (
        f"value learning stalled at {final['value_acc']:.4f} "
        f"(class prior ≈ 0.86 — a dead value pathway sits exactly there)"
    )
    # Gate companion: illegal-move masking holds after training, not just at init.
    assert illegal_probability_is_exactly_zero(trainer, tensors), (
        "trained net assigns nonzero probability to illegal moves (§9.3)"
    )


def illegal_probability_is_exactly_zero(trainer: Trainer, dataset) -> bool:
    """After training: exp(logp) must be exactly 0.0 at every illegal slot
    (the search would otherwise put prior mass on impossible moves)."""
    from chess_ai.train.trainer import tensors_to_torch

    model, device = trainer.model, trainer.device
    model.eval()
    with torch.no_grad():
        for start in range(0, dataset["tokens"].shape[0], 512):
            sl = slice(start, min(start + 512, dataset["tokens"].shape[0]))
            batch = tensors_to_torch({k: v[sl] for k, v in dataset.items()}, device)
            logp = model(batch["tokens"], batch["mask"])["policy_logp"]
            probs = logp.exp()
            if (probs[~batch["mask"]] != 0).any():
                return False
            masses = probs.masked_fill(~batch["mask"], 0).sum(-1)
            if not torch.allclose(masses, torch.ones_like(masses), atol=1e-4):
                return False
    model.train()
    return True
