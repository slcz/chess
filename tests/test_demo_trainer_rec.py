"""
tests/test_demo_trainer_rec.py — scripts/sp_round_train.py (the readable
demo trainer) end-to-end with REAL engine-built shards: the --buffer
replay path keeps exactly the buffer_keep-predicted games, and the
per-round rec.json carries args, fresh/buffer tallies, metric averages
and the checkpoint list.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from chess_ai.train.selfplay_data import buffer_keep

ROOT = Path(__file__).parent.parent
BIN = ROOT / "engine" / "build" / "selfplay"
BASE = ROOT / "runs" / "scratch-gen1.pt"
TRAINER = ROOT / "scripts" / "sp_round_train.py"

pytestmark = pytest.mark.skipif(
    not BIN.exists() or not BASE.exists(),
    reason="selfplay binary or random-base checkpoint not built/present",
)


def gen_unit(tmp: Path, games: int, seed: int) -> Path:
    d = tmp / f"u{seed}"
    d.mkdir()
    subprocess.run([str(BIN), str(BASE), str(d), str(games), "8", "2", "16",
                    "4", "60", "4", "mps", str(seed), "2", "0.5"],
                   check=True, capture_output=True, text=True, cwd=ROOT,
                   timeout=600)
    assert (d / "DONE.json").exists(), "engine did not finish cleanly"
    return d


def test_buffer_trim_and_rec_json(tmp_path: Path) -> None:
    fresh = gen_unit(tmp_path, 3, 7)
    newer = gen_unit(tmp_path, 4, 9)     # age 1 (newest buffer pool)
    older = gen_unit(tmp_path, 4, 11)    # age 2

    out = tmp_path / "cand"
    subprocess.run([".venv/bin/python", str(TRAINER),
                    "--data-dir", str(fresh),
                    "--buffer", str(newer), "--buffer", str(older),
                    "--trim-seed", "5",
                    "--base-ckpt", str(BASE), "--out", str(out),
                    "--epochs", "1", "--batch", "8", "--slow-only",
                    "--device", "mps"],
                   check=True, capture_output=True, text=True, cwd=ROOT,
                   timeout=600)

    rec = json.loads((tmp_path / "cand-rec.json").read_text())
    # fresh: every game trains
    assert rec["fresh"]["games"] == 3
    assert sum(rec["fresh"]["wdl"]) == 3
    # buffers: (2/3)^age trim, deterministic in (seed, unit_seed, game_index)
    for pool, age in ((rec["buffer"][0], 1), (rec["buffer"][1], 2)):
        useed = json.loads((Path(pool["dir"]) / "DONE.json").read_text())["unit_seed"]
        n_done = json.loads((Path(pool["dir"]) / "DONE.json").read_text())["games_done"]
        want = sum(buffer_keep(useed, i, age, seed=5) for i in range(n_done))
        assert pool["games"] == want, (pool, want)
        assert sum(pool["wdl"]) == pool["games"]
    # positions = all trained rows; one step per epoch at batch 8 (ceil)
    assert rec["positions"] > 0
    assert rec["steps"] == math.ceil(rec["positions"] / 8) * 1
    for row in ("begin", "end"):
        assert "loss" in rec["metrics"][row] and "policy_acc" in rec["metrics"][row]
    assert rec["checkpoints"] == [str(tmp_path / "cand-e1.pt")]
    assert (tmp_path / "cand-e1.pt").exists()
