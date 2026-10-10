"""
tests/test_selfplay_roundtrip.py — end-to-end self-play data path on the
local device: run engine/build/selfplay for a couple of small games (with
playout cap randomization ON: fast_sims=4 @ 75%, slow 16), read the
SPSHARD1 shard back, convert to training samples (fast-cap plies must be
excluded), and take one real optimization step on the seed checkpoint.
Proves the whole chain (driver → shard bytes → reader → tensors →
dense-π train_step) agrees — the reader's per-ply asserts (played move
legal, mover matches replay, π sums to 1) do the format checking along
the way.

Skipped unless both the binary and the seed checkpoint exist (fresh
clones without a build/checkpoint get a skip, not a failure). Runtime:
~10 s on MPS with 2 games × 16 sims.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from chess_ai.selfplay.shard import read_unit
from chess_ai.train.selfplay_data import (
    PI_SUPPORT_MAX,
    ply_trainable,
    selfplay_samples,
    stack_selfplay,
)
from chess_ai.train.trainer import Trainer, tensors_to_torch

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "engine" / "build" / "selfplay"
CKPT = ROOT / "runs" / "policy_value_net_rope_25k.pt"

pytestmark = pytest.mark.skipif(
    not BIN.exists() or not CKPT.exists(),
    reason="selfplay binary or seed checkpoint not built/present",
)


def test_selfplay_roundtrip(tmp_path: Path) -> None:
    proc = subprocess.run(
        [str(BIN), str(CKPT), str(tmp_path), "2", "16", "2", "16",
         "10", "120", "4", "0.25", "0.3", "mps", "3", "4", "0.75"],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SUMMARY" in proc.stdout

    done = json.loads((tmp_path / "DONE.json").read_text())
    assert done["games_done"] == 2 and done["n_workers"] == 1
    assert done["unit_seed"] == 3

    games = read_unit(tmp_path)
    assert len(games) == 2
    assert sorted(g.game_index for g in games) == [0, 1]
    fast = slow = 0
    for g in games:
        assert g.n_plies == len(g.plies) and g.n_plies > 0
        assert g.resigner == -1 and g.n_sims == 16
        assert g.z_white in (0.0, 0.5, 1.0)
        for ply in g.plies:
            assert ply.played_slot >= 0 and len(ply.played_uci) >= 4
            assert ply.sims in (4, 16)
            fast, slow = fast + (ply.sims == 4), slow + (ply.sims == 16)
            assert 1 <= len(ply.pi) <= PI_SUPPORT_MAX
            assert math.isclose(sum(p for _, p in ply.pi), 1.0, rel_tol=1e-5)
    # both caps must have fired (per-step 75% draw over dozens of plies)
    assert fast > 0 and slow > 0

    samples = [s for g in games for s in selfplay_samples(g)]
    assert len(samples) == sum(
        sum(1 for p in g.plies if ply_trainable(g, p)) for g in games)
    assert 0 < len(samples) < sum(g.n_plies for g in games)
    tensors = stack_selfplay(samples)
    assert tensors["pi_slots"].shape == (len(samples), PI_SUPPORT_MAX)
    # padding is invisible to the loss: probs sum to 1 over the support
    np.testing.assert_allclose(tensors["pi_probs"].sum(-1), 1.0, atol=1e-6)

    trainer = Trainer.load(CKPT)
    n = min(32, len(samples))
    batch = tensors_to_torch(
        {k: v[:n] for k, v in tensors.items()}, trainer.device)
    metrics = trainer.train_step(batch)
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["policy"] > 0 and metrics["value"] >= 0
