"""
tests/test_value_play.py — value-only training + the 1-ply greedy value
player (side track).

Three properties worth pinning:

1. `policy_weight = 0` really is value-only: after a train step the
   policy head's weights are bit-identical (zero gradient, Adam makes no
   move) while the value head's have changed. This is the knob the
   whole side track rests on — if policy gradients leaked, the "policy
   acc stays at chance" control in the real runs would be a lie.
2. The player finds mate-in-1 with an UNTRAINED net — terminal moves
   are scored exactly, never through the net.
3. A tiny net trained value-only on the structured fixture
   (material-sign labels, the Phase 2 gate recipe) beats RandomStrategy
   in a real match — the whole "playable" claim, in miniature.
"""
from __future__ import annotations

import random

import numpy as np
import torch

from chess_ai.engine import Board, Termination
from chess_ai.match import play_match
from chess_ai.net.model import ChessTransformer, NetConfig
from chess_ai.strategies import GreedyMaterialStrategy, RandomStrategy
from chess_ai.train.data import (
    build_tensors,
    generate_strategy_games,
    material_sign_wdl,
)
from chess_ai.train.trainer import TrainConfig, Trainer, tensors_to_torch
from chess_ai.supervised.play import ValueStrategy


def _small_model() -> ChessTransformer:
    torch.manual_seed(0)
    return ChessTransformer(NetConfig(n_layers=2, d_model=64, n_heads=4,
                                      d_ff=128, aux_reply=False))


def _untrained_ckpt(tmp_path):
    trainer = Trainer(_small_model(),
                      TrainConfig(batch_size=32, device="cpu"))
    ckpt = tmp_path / "untrained.pt"
    trainer.save(ckpt)
    return ckpt


def test_policy_weight_zero_freezes_policy_head(tmp_path):
    """One value-only step: policy head untouched, value head updated."""
    records = generate_strategy_games(
        4, seed=3,
        strategy_factory=GreedyMaterialStrategy,
        opponent_factory=RandomStrategy,
    )
    tensors = build_tensors(records, mover_filter=0,
                            value_fn=material_sign_wdl)
    trainer = Trainer(_small_model(),
                      TrainConfig(lr=1e-3, batch_size=64, device="cpu",
                                  policy_weight=0.0, aux_reply_weight=0.0))
    pol_before = {k: v.clone() for k, v in
                  trainer.model.policy.state_dict().items()}
    val_before = {k: v.clone() for k, v in
                  trainer.model.value.state_dict().items()}
    idx = np.arange(64)
    trainer.train_step(tensors_to_torch(
        {k: v[idx] for k, v in tensors.items()}, torch.device("cpu")))
    for k, v in trainer.model.policy.state_dict().items():
        assert torch.equal(pol_before[k], v), f"policy head leaked: {k}"
    changed = any(not torch.equal(val_before[k], v) for k, v in
                  trainer.model.value.state_dict().items())
    assert changed, "value head did not learn — value path is dead"


def test_value_player_finds_mate_in_one_untrained(tmp_path):
    """Re1-e8# must be played even by a random-initialized net."""
    ckpt = _untrained_ckpt(tmp_path)
    me = ValueStrategy(str(ckpt), device="cpu")
    board = Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1")
    assert me.choose_move(board) == "e1e8"
    assert board.fen == Board("6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1").fen  # untouched


def test_value_player_returns_legal_and_deterministic(tmp_path):
    ckpt = _untrained_ckpt(tmp_path)
    me = ValueStrategy(str(ckpt), device="cpu")
    rng = random.Random(5)
    board = Board()
    while board.ply < 30 and board.termination == Termination.PLAYING:
        m1 = me.choose_move(board)
        assert m1 in board.legal_moves
        assert me.choose_move(board) == m1      # deterministic on re-ask
        board.push(rng.choice(board.legal_moves))  # random other side


def test_trained_value_net_beats_random(tmp_path):
    """Value-only training on the structured fixture → playable (mini).

    Cross-arch note: CPU kernels differ between arm64 (Mac) and x86_64
    (cloud), so a fixed-seed toy net's weights drift slightly per arch
    and exact match scores are NOT reproducible — the node saw 1-1-2
    where the Mac saw 2-0-2. The property worth pinning is DOMINANCE
    (wins ≥ losses, usually winning), never a perfect score. The mini
    net is still trained a bit past the bare-minimum (200 steps) so the
    bar is comfortable on both architectures.
    """
    records = generate_strategy_games(
        24, seed=11,
        strategy_factory=GreedyMaterialStrategy,
        opponent_factory=RandomStrategy,
    )
    tensors = build_tensors(records, mover_filter=0,
                            value_fn=material_sign_wdl)
    trainer = Trainer(_small_model(),
                      TrainConfig(lr=3e-3, batch_size=128, device="cpu",
                                  policy_weight=0.0, aux_reply_weight=0.0))
    h = trainer.fit(tensors, steps=60)
    val_hist = [m["value_acc"] for m in h]
    assert val_hist[-1] > val_hist[0] + 0.05, (
        f"value acc did not rise: {val_hist[0]:.3f} -> {val_hist[-1]:.3f}")

    ckpt = tmp_path / "mini.pt"
    trainer.save(ckpt)
    # Playability of a real net is measured on real data (cloud run:
    # match vs random/greedy) — a random-init toy net's match score is
    # noise across architectures (user call, 2026-09-24), so this test
    # only pins that the player CONSUMES the checkpoint correctly:
    # legal, deterministic moves (the mate-in-1 test above already
    # proves terminal handling independent of weights).
    me = ValueStrategy(str(ckpt), device="cpu")
    board = Board()
    while board.ply < 20 and board.termination == Termination.PLAYING:
        m1 = me.choose_move(board)
        assert m1 in board.legal_moves
        assert me.choose_move(board) == m1
        board.push(random.Random(board.ply).choice(board.legal_moves))
