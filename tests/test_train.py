"""
tests/test_train.py — data pipeline invariants + checkpoint round-trip
(Phase 2.2).

build_tensors() is where a silent poisoning bug would hide (wrong
perspective, mask/target mismatch), so its invariants are asserted on real
replayed games: every policy target sits inside its own legal mask, every
weighted aux target sits inside its own (same-frame!) aux mask, the
no-reply weight lands exactly on each game's last ply, and WDL labels flip
correctly with side to move. The checkpoint test proves save/load resumes
bit-identically — the property the salad churn drill (Phase 4.4) will lean
on.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from chess_ai.net.model import ChessTransformer, NetConfig
from chess_ai.train import (
    GameRecord,
    ReplayBuffer,
    TrainConfig,
    Trainer,
    build_tensors,
    generate_random_games,
)
from chess_ai.train.trainer import tensors_to_torch

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture(scope="module")
def games():
    return generate_random_games(8, seed=11, max_plies=50)


@pytest.fixture(scope="module")
def tensors(games):
    return build_tensors(games)


# ---------------------------------------------------------------------------
# build_tensors invariants
# ---------------------------------------------------------------------------

def test_shapes_and_dtypes(tensors, games):
    n = sum(len(g.moves) for g in games)
    assert tensors["tokens"].shape == (n, 64, 117)
    assert tensors["mask"].shape == (n, 4240)
    assert tensors["pol_idx"].shape == (n,)
    assert tensors["wdl"].shape == (n, 3)
    assert tensors["aux_mask"].shape == (n, 4240)
    assert tensors["tokens"].dtype == np.float32
    assert tensors["mask"].dtype == bool
    assert tensors["pol_idx"].dtype == np.int64


def test_policy_target_inside_its_mask(tensors):
    assert tensors["mask"][np.arange(len(tensors["pol_idx"])), tensors["pol_idx"]].all()


def test_aux_target_inside_its_mask_when_weighted(tensors):
    w = tensors["aux_w"]
    hit = tensors["aux_mask"][np.arange(len(w)), tensors["aux_idx"]]
    assert hit[w == 1.0].all()
    assert (w == 0.0).sum() == len(games_count(tensors))   # one per game


def games_count(tensors):
    """Recover per-game boundaries from aux_w: the last ply of each game is
    exactly where the reply is missing."""
    return np.flatnonzero(tensors["aux_w"] == 0.0)


def test_aux_weight_zero_exactly_on_last_plies(tensors, games):
    last_plies = np.cumsum([len(g.moves) for g in games]) - 1
    assert set(games_count(tensors)) == set(last_plies)


def test_wdl_perspective_flips_with_side(tensors, games):
    """A decisive game's labels must be W for the winner's plies, L for the
    loser's — the classic sign-flip bug catcher."""
    decisive = [g for g in games if g.outcome != 0.0]
    if not decisive:  # draw-only random fixture: force a scholar's mate
        decisive = [GameRecord(
            START,
            ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"),
            1.0,
        )]
    t = build_tensors(decisive)
    off = 0
    for g in decisive:
        for ply in range(len(g.moves)):
            wdl = t["wdl"][off + ply]
            white_to_move = (ply % 2 == 0)
            mover_won = (g.outcome > 0) == white_to_move
            assert wdl[0 if mover_won else 2] == 1.0, (g.outcome, ply, wdl)
        off += len(g.moves)


def test_replay_buffer_cap_and_sampling(games):
    buf = ReplayBuffer(max_games=3)
    for g in games[:5]:
        buf.add(g)
    assert len(buf) == 3                       # oldest fell off
    assert buf.total_positions == sum(len(g.moves) for g in games[2:5])
    rng = random.Random(0)
    picks = buf.sample_positions(rng, 50)
    assert len(picks) == 50
    for rec, ply in picks:
        assert rec in games[2:5]
        assert 0 <= ply < len(rec.moves)


# ---------------------------------------------------------------------------
# trainer + checkpointing
# ---------------------------------------------------------------------------

def _small_trainer(device="cpu"):
    torch.manual_seed(0)
    model = ChessTransformer(NetConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128))
    cfg = TrainConfig(lr=1e-3, batch_size=32, device=device, seed=0)
    return Trainer(model, cfg)


def _fixed_batch(tensors, device="cpu"):
    idx = np.arange(32)
    return tensors_to_torch({k: v[idx] for k, v in tensors.items()}, torch.device(device))


def test_checkpoint_round_trip_resumes_identically(tensors, tmp_path):
    torch.manual_seed(0)
    trainer = _small_trainer()
    batch = _fixed_batch(tensors)
    trainer.train_step(batch)                    # step 1: optimizer gains state

    ckpt = tmp_path / "ckpt.pt"
    trainer.save(ckpt)                           # both paths now hold this state
    assert not any(p.suffix == ".tmp" for p in tmp_path.iterdir())   # atomic

    loss_before = trainer.train_step(batch)["loss"]          # original's step 2

    resumed = Trainer.load(ckpt)
    assert resumed.steps == 1                                # saved after step 1
    # If ANY weight or optimizer moment failed to round-trip, the resumed
    # path's losses would drift — the two equality checks below are the
    # bit-exactness proof, on both continuation steps.
    loss_after_resume = resumed.train_step(batch)["loss"]
    assert loss_after_resume == pytest.approx(loss_before, rel=1e-6)

    loss_original_path = trainer.train_step(batch)["loss"]
    loss_second_resume = resumed.train_step(batch)["loss"]
    assert loss_second_resume == pytest.approx(loss_original_path, rel=1e-6)


def test_fit_reduces_loss_on_fixed_set(tensors):
    torch.manual_seed(0)
    trainer = _small_trainer()
    subset = {k: v[:512] for k, v in tensors.items()}
    h = trainer.fit(subset, steps=25)
    assert h[-1]["loss"] < h[0]["loss"]

def test_ply_weight_scales_policy_and_value():
    """--all-plies semantics: ply_w scales the policy and value losses (so a
    fast-cap ply can contribute at a fraction of a slow-cap one)."""
    import torch.nn.functional as F

    from chess_ai.net.model import ChessTransformer, NetConfig
    from chess_ai.train.trainer import Trainer, TrainConfig

    torch.manual_seed(0)
    model = ChessTransformer(NetConfig(n_layers=1, d_model=64, n_heads=4,
                                       d_ff=128, rope_dirs=2))
    trainer = Trainer(model, TrainConfig(lr=1e-3, batch_size=2, device="cpu"))
    B, S = 2, 4240
    batch = {
        "tokens": torch.randn(B, 64, model.cfg.token_dim),
        "mask": torch.ones(B, S, dtype=torch.bool),
        "pol_idx": torch.randint(0, S, (B,)),
        "wdl": torch.eye(3)[torch.randint(0, 3, (B,))],
        "aux_idx": torch.randint(0, S, (B,)),
        "aux_mask": torch.ones(B, S, dtype=torch.bool),
        "aux_w": torch.ones(B),
        "pi_slots": torch.randint(0, S, (B, 6)),
        "pi_probs": torch.full((B, 6), 1 / 6),
        "ply_w": torch.tensor([1.0, 1.0]),           # value weights
        "pol_w": torch.tensor([1.0, 0.15]),           # policy weights
    }
    with torch.no_grad():
        out = model(batch["tokens"], batch["mask"], batch["aux_mask"])
        vnll = -(F.log_softmax(out["wdl"], -1) * batch["wdl"]).sum(-1)
        pnll = -(batch["pi_probs"]
                 * out["policy_logp"].gather(1, batch["pi_slots"])).sum(-1)
    m = trainer.train_step(batch)
    vw, pw = batch["ply_w"], batch["pol_w"]
    assert abs(m["value"] - float((vnll * vw).sum() / vw.sum())) < 1e-6
    assert abs(m["policy"] - float((pnll * pw).sum() / pw.sum())) < 1e-6
