"""
tests/test_net_model.py — the network's structural contracts (Phase 2).

The big one is masking: illegal moves must carry probability EXACTLY zero
(risk register §9.3 — an unmasked prior silently poisons search). Also:
parameter budget (~6M, the CF-6M recipe), output shapes/ranges, the
aux-head fallback for finished games, one training step end-to-end, and an
MPS smoke test so device bugs surface now, not during the first cloud run.
"""
from __future__ import annotations

import pytest
import torch

from chess_ai.net.features import POLICY_SIZE
from chess_ai.net.model import ChessTransformer, NetConfig, count_parameters


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(0)
    B = 4
    tokens = torch.randn(B, 64, 117)
    mask = torch.zeros(B, POLICY_SIZE, dtype=torch.bool)
    for b in range(B):
        mask[b, torch.randperm(POLICY_SIZE)[:30]] = True
    return tokens, mask


def test_parameter_budget_default_config():
    # Default = transformer trunk (CF-6M recipe, ~6.5M) + pooled policy
    # head + aux (~2×1.1M). The "attention" head variant stays near the
    # original ~6.8M — assert both so a future head change can't silently
    # blow the budget.
    n = count_parameters(ChessTransformer())          # pooled default
    assert 8_000_000 <= n <= 9_500_000, n
    n_attn = count_parameters(ChessTransformer(NetConfig(policy_head="attention")))
    assert 6_500_000 <= n_attn <= 7_200_000, n_attn


def test_forward_shapes_and_ranges(batch):
    tokens, mask = batch
    model = ChessTransformer(NetConfig(n_layers=2))
    model.eval()
    with torch.no_grad():
        out = model(tokens, mask)
    assert out["policy_logp"].shape == (4, POLICY_SIZE)
    assert out["wdl"].shape == (4, 3)
    assert out["value"].shape == (4,)
    assert out["aux_logp"].shape == (4, POLICY_SIZE)
    assert (out["value"].abs() <= 1).all()


def test_illegal_moves_have_exactly_zero_probability(batch):
    tokens, mask = batch
    model = ChessTransformer(NetConfig(n_layers=2))
    model.eval()
    with torch.no_grad():
        out = model(tokens, mask)
    p = out["policy_logp"].exp()
    assert (p[~mask] == 0).all()                    # exact zeros, not ~0
    masses = p.masked_fill(~mask, 0).sum(-1)
    assert torch.allclose(masses, torch.ones(4), atol=1e-5)


def test_aux_head_all_false_mask_stays_finite(batch):
    """Finished-game rows pass an all-False aux mask; the head must not
    produce NaN (their loss weight is zero)."""
    tokens, mask = batch
    model = ChessTransformer(NetConfig(n_layers=2))
    model.eval()
    am = mask.clone()
    am[0] = False
    with torch.no_grad():
        out = model(tokens, mask, aux_mask=am)
    assert torch.isfinite(out["aux_logp"]).all()


def test_aux_head_optional():
    model = ChessTransformer(NetConfig(n_layers=1, aux_reply=False))
    assert model.aux is None
    out = model(torch.randn(2, 64, 117), torch.ones(2, POLICY_SIZE, dtype=torch.bool))
    assert "aux_logp" not in out


def test_one_train_step_is_finite():
    from chess_ai.train import TrainConfig, Trainer, build_tensors, generate_random_games

    games = generate_random_games(2, seed=3, max_plies=30)
    tensors = build_tensors(games)
    model = ChessTransformer(NetConfig(n_layers=1, d_model=64, n_heads=4, d_ff=128))
    trainer = Trainer(model, TrainConfig(lr=1e-3, batch_size=16, device="cpu"))
    from chess_ai.train.trainer import tensors_to_torch
    batch = tensors_to_torch(
        {k: v[:16] for k, v in tensors.items()}, torch.device("cpu")
    )
    metrics = trainer.train_step(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert 0.0 <= metrics["policy_acc"] <= 1.0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_forward_and_step():
    from chess_ai.train import TrainConfig, Trainer, build_tensors, generate_random_games
    from chess_ai.train.trainer import tensors_to_torch

    games = generate_random_games(2, seed=3, max_plies=30)
    tensors = build_tensors(games)
    model = ChessTransformer(NetConfig(n_layers=2, d_model=128, n_heads=4, d_ff=256))
    trainer = Trainer(model, TrainConfig(lr=1e-3, batch_size=16, device="mps"))
    batch = tensors_to_torch(
        {k: v[:16] for k, v in tensors.items()}, torch.device("mps")
    )
    metrics = trainer.train_step(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
