"""tests/test_selfplay_loop.py — the chess self-play training loop
(Step 1: the whole chess learn cycle, end to end, tiny).

These tests run the REAL loop — the C parallel engine with a torch
evaluator checkpoint, records → SelfPlayBuffer → Trainer.train_step →
latest.pt — at toy scale (tiny net, 4 games/round, 2 train steps, CPU).
They are the integration gate the salad pilot stands on:

- POOL: build_start_pool is deterministic per seed, produces legal,
  non-startpos positions within the prefix-depth bound.
- SMOKE: two rounds complete; each round's unit dir carries shards plus a
  DONE.json the trainer's read_unit consumes; latest.pt round-trips;
  log.jsonl gets one line per round.
- RESUME: a second run_loop call on the same out dir ingests the first
  round's unit instead of replaying it, and continues at round 2.
"""
import json
import random
from pathlib import Path

import pytest
import torch

from chess_ai.engine import Board
from chess_ai.net.model import ChessTransformer, NetConfig
from chess_ai.train.selfplay_data import SelfPlayBuffer, read_unit
from chess_ai.train.selfplay_loop import LoopConfig, build_start_pool, run_loop
from chess_ai.train.trainer import TrainConfig, Trainer, tensors_to_torch


def _tiny_cfg() -> NetConfig:
    return NetConfig(d_model=64, n_layers=1, n_heads=4, d_ff=128)


def _save_seed(path: Path) -> None:
    torch.manual_seed(0)
    net = ChessTransformer(_tiny_cfg())
    Trainer(net, TrainConfig(device="cpu")).save(path)


def _tiny_kw(seed_ckpt: str, out_dir: str, iters: int, **over) -> dict:
    cfg = LoopConfig(seed_ckpt=seed_ckpt, out_dir=out_dir, iters=iters,
                     games_per_iter=4, workers=2, buffer_requests=8,
                     train_steps=2, batch_size=16, n=4, start_plies=3,
                     start_pool_size=4, max_plies=12, eval_device="cpu",
                     save_every=0, device="cpu")
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


@pytest.fixture(scope="module")
def two_round_out(tmp_path_factory):
    """One tiny run of two rounds, shared by the smoke assertions."""
    root = tmp_path_factory.mktemp("loop")
    _save_seed(root / "seed.pt")
    cfg = _tiny_kw(str(root / "seed.pt"), str(root / "run"), iters=2, seed=77)
    run_loop(cfg, log=lambda *_: None)
    return root / "run"


def test_pool_deterministic_and_legal():
    a = build_start_pool(123, 8, 3)
    b = build_start_pool(123, 8, 3)
    c = build_start_pool(124, 8, 3)
    assert a == b
    assert a != c
    start = Board().fen
    for fen in a:
        board = Board(fen)
        assert len(board.legal_moves) > 0
        assert fen != start            # at least one prefix ply was played


def test_pool_survives_terminal_prefixes():
    """A random legal prefix can walk into mate/stalemate (live crash at
    pilot round 26, 2026-09-30: `randrange(0)` on an empty move list).
    Enough samples that the resample branch runs many times; every emitted
    FEN must be a playable, non-start position."""
    start = Board().fen
    seen: set[str] = set()
    for pool_seed in range(60):
        for fen in build_start_pool(pool_seed, 40, 8):
            board = Board(fen)
            assert len(board.legal_moves) > 0      # never terminal
            assert fen != start
            seen.add(fen)
    assert len(seen) > 1000                        # real coverage, not 1 FEN


def test_two_rounds_smoke(two_round_out):
    out = two_round_out
    units = sorted((out / "units").glob("iter-*"))
    assert [d.name for d in units] == ["iter-0001", "iter-0002"]

    # each unit is a complete, trainer-readable work unit
    for d in units:
        manifest = json.loads((d / "DONE.json").read_text())
        assert manifest["format"] == "SPSHARD1"
        assert manifest["games_done"] == 4
        assert len(read_unit(d)) == 4

    # checkpoints + log
    assert (out / "latest.pt").exists()
    lines = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert [m["iter"] for m in lines] == [1, 2]
    assert lines[-1]["steps"] > lines[0]["steps"]
    assert lines[-1]["games_total"] == 8
    for m in lines:
        assert 0.0 <= m["sp"]["draw_rate"] <= 1.0
        assert m["sp"]["games_per_hour"] > 0


def test_resume_continues_at_next_round(tmp_path_factory):
    """Round 2 must ingest round 1's unit (dedup by (seed, game_index)),
    not replay it — the same resume path the cloud driver will use."""
    root = tmp_path_factory.mktemp("resume")
    _save_seed(root / "seed.pt")
    kw = dict(seed_ckpt=str(root / "seed.pt"), out_dir=str(root / "run"))
    run_loop(_tiny_kw(**kw, iters=1, seed=77), log=lambda *_: None)
    steps_after_one = Trainer.load(root / "run" / "latest.pt").steps

    logged: list[str] = []
    run_loop(_tiny_kw(**kw, iters=1, seed=77), log=logged.append)
    steps_after_two = Trainer.load(root / "run" / "latest.pt").steps

    assert steps_after_two == steps_after_one + 2
    assert any("ingested 1 completed unit" in m for m in logged)
    lines = [json.loads(l)
             for l in (root / "run" / "log.jsonl").read_text().splitlines()]
    assert [m["iter"] for m in lines] == [1, 2]   # continued, not restarted


def test_startpos_only_when_disabled(tmp_path_factory):
    """start_plies=0 → no pool → every game from the standard start."""
    root = tmp_path_factory.mktemp("startpos")
    _save_seed(root / "seed.pt")
    cfg = _tiny_kw(str(root / "seed.pt"), str(root / "run"), iters=1,
                   seed=5, start_plies=0, games_per_iter=2, train_steps=1,
                   batch_size=8)
    run_loop(cfg, log=lambda *_: None)
    unit = next((root / "run" / "units").glob("iter-*"))
    records = read_unit(unit)
    assert len(records) == 2
    assert all(r.start_fen == Board().fen for r in records)


def test_eq9_mode_trains_on_survivor(tmp_path_factory):
    """policy_target='eq9': the paper's L_simple = −log π(A_{n+1}).
    Pins the two coupled behaviors: (a) self-play plays the halving
    SURVIVOR (sample_plies forced to 0 — DONE.json + metrics record it),
    (b) train batches drop the dense π′ keys, so Trainer's sparse branch
    (CE on the played move) fires instead of the dense target-weighted
    CE. Value/aux losses unchanged."""
    root = tmp_path_factory.mktemp("eq9")
    _save_seed(root / "seed.pt")
    cfg = _tiny_kw(str(root / "seed.pt"), str(root / "run"), iters=1, seed=31)
    cfg.policy_target = "eq9"
    run_loop(cfg, log=lambda *_: None)

    unit = next((root / "run" / "units").glob("iter-*"))
    manifest = json.loads((unit / "DONE.json").read_text())
    assert manifest["sample_plies"] == 0            # survivor played
    assert manifest["policy_target"] == "eq9"

    lines = (root / "run" / "log.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["policy_target"] == "eq9"

    # the training path: rebuild a batch the way run_round does and verify
    # the sparse branch engages (no dense keys → plain CE on played move)
    recs = read_unit(unit)
    buf = SelfPlayBuffer(100)
    buf.add_all(recs)
    batch = buf.sample_tensors(random.Random(0), 8)
    batch.pop("pi_slots")
    batch.pop("pi_probs")
    trainer = Trainer.load(root / "run" / "latest.pt")
    m = trainer.train_step(tensors_to_torch(batch, trainer.device))
    assert set(m) == {"loss", "policy", "value", "aux",
                      "policy_acc", "value_acc"}   # same report, sparse path


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
