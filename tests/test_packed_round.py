"""
tests/test_packed_round.py — PackedRound (the trainer's memory-frugal
ingest) must produce samples byte-identical to the reference builders:

  1. packed.sample(row) == selfplay_samples(g) entry-by-entry over the
     round's trainable plies (dtype included), on real engine output;
  2. selfplay_sample(g, ply) == the same reference for EVERY trainable
     ply — this pins the push/pop history bug: tokens must be captured
     BEFORE the aux-reply excursion, or the full 8-ply ring leaves a
     stale/zeroed h=7 frame in the encoded tensors (ply >= 7).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest

from chess_ai.selfplay.shard import read_unit
from chess_ai.train.selfplay_data import (
    PackedRound,
    selfplay_sample,
    selfplay_samples,
)

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "engine" / "build" / "selfplay"
CKPT = ROOT / "runs" / "policy_value_net_rope_25k.pt"

pytestmark = pytest.mark.skipif(
    not BIN.exists() or not CKPT.exists(),
    reason="selfplay binary or seed checkpoint not built/present",
)


def assert_same(a: dict, b: dict) -> None:
    assert set(a) == set(b)
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, np.ndarray):
            assert x.dtype == y.dtype, (k, x.dtype, y.dtype)
            np.testing.assert_array_equal(x, y, err_msg=k)
        else:
            assert x == y, k


@pytest.fixture(scope="module")
def games(tmp_path_factory) -> list:
    tmp = tmp_path_factory.mktemp("unit")
    proc = subprocess.run(
        [str(BIN), str(CKPT), str(tmp), "2", "16", "2", "16",
         "10", "120", "4", "0.25", "0.3", "mps", "3", "4", "0.75"],
        capture_output=True, text=True, timeout=600, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    return read_unit(tmp)


def test_packed_matches_reference(games) -> None:
    packed = PackedRound()
    for g in games:
        packed.add_game(g)
    packed.finalize()

    assert packed.n_games == len(games)
    assert packed.n_recorded_plies == sum(g.n_plies for g in games)
    reference = [s for g in games for s in selfplay_samples(g)]
    assert len(packed) == len(reference) > 0
    for row in range(len(packed)):
        assert_same(packed.sample(row), reference[row])


def test_selfplay_sample_matches_reference(games) -> None:
    # covers plies well past the 8-ply ring capacity (every game here
    # runs dozens of plies), where the push/pop zeroing used to bite
    for g in games:
        ref = selfplay_samples(g)
        rows = [i for i, p in enumerate(g.plies) if p.sims == g.n_sims]
        assert len(rows) == len(ref)
        for i, r in zip(rows, ref):
            assert_same(selfplay_sample(g, i), r)
