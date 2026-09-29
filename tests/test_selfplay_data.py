"""tests/test_selfplay_data.py — Phase 3.4 (trainer side): self-play
records → the dense-π′ training format.

The recorder half of 3.4 shipped with 3.3b (collect-time shard dumps);
these tests gate the OTHER half (chess_ai/train/selfplay_data.py):

- schema: dense π′ arrays (support legal, renormalized, NaN-safe padding),
  masks, played-move indices, WDL perspective, aux-reply stitching;
- resignation semantics: the resign ply contributes no sample, the ply
  before it loses its aux target, and the outcome label is
  resigner-loses everywhere;
- the replay window: eviction, playable-position accounting, deterministic
  sampling;
- the loss seam: dense target-weighted CE == the Phase 2 sparse CE when
  π′ is one-hot (Trainer accepts both forms), and a real train_step learns
  on self-play tensors end to end;
- unit ingestion: DONE.json manifests, (seed, game_index) dedup, and the
  truncated-tail detection the shard format promises.
"""
import copy
import json
import random
import struct

import numpy as np
import pytest

from chess_ai.net.features import uci_to_index
from chess_ai.selfplay.gumbel import SelfPlayConfig
from chess_ai.selfplay.spengine import SpRun, read_records, shard_path
from chess_ai.train.selfplay_data import (
    PI_SUPPORT_MAX,
    SelfPlayBuffer,
    build_selfplay_tensors,
    ingest_units,
    playable_plies,
    read_unit,
    samples_at,
    selfplay_samples,
)


def play_records(tmp_path, *, eval_spec="fixture", seed=500, budget=6,
                 workers=2, max_plies=10):
    """A tiny parallel run through the real engine (fixture evaluator)."""
    cfg = SelfPlayConfig()
    run = SpRun(cfg, unit_seed=seed, game_budget=budget, n_workers=workers,
                buffer_requests=8, max_plies=max_plies, eval_spec=eval_spec,
                record_dir=tmp_path)
    try:
        return run.play()
    finally:
        run.close()


@pytest.fixture(scope="module")
def natural_records(tmp_path_factory):
    """Adjudicated (max_plies) games — every ply has a played move."""
    return play_records(tmp_path_factory.mktemp("nat"), seed=500)


@pytest.fixture(scope="module")
def resign_records(tmp_path_factory):
    """fixture:white_wins: black's first search sees outlook −1 and resigns
    immediately — records are [white's move, black's resign ply]."""
    return play_records(tmp_path_factory.mktemp("res"),
                        eval_spec="fixture:white_wins", seed=501, budget=4)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_dense_pi_schema(natural_records):
    tensors = build_selfplay_tensors(natural_records)
    n = tensors["tokens"].shape[0]
    assert tensors["tokens"].shape[1:] == (64, 117)
    assert tensors["pi_slots"].shape == (n, PI_SUPPORT_MAX)
    assert tensors["pi_slots"].dtype == np.int64
    assert tensors["pi_probs"].dtype == np.float32
    for i in range(n):
        probs = tensors["pi_probs"][i]
        assert abs(probs.sum() - 1.0) < 1e-6          # renormalized exactly
        k = int((probs > 0).sum())                    # support from PROBS:
        assert 0 < k <= PI_SUPPORT_MAX                 # an f32-rounded tail
        assert (probs[k:] == 0).all()                  # may legally be 0
        support = tensors["pi_slots"][i][:k]
        # every supported slot is legal in this position, and the played
        # move (the metrics stand-in) is legal too
        assert tensors["mask"][i][support].all()
        assert tensors["mask"][i][tensors["pol_idx"][i]]
        # padding repeats the first REAL slot with prob 0 — never a blind
        # slot-0 pad that could gather −inf · 0 = NaN from policy_logp
        if k < PI_SUPPORT_MAX:
            assert (tensors["pi_slots"][i][k:] == support[0]).all()


def test_wdl_perspective_over_all_outcomes(natural_records):
    """The label transformation is pure: z_white × mover → the mover's W/D/L
    one-hot. Exercise every (z, mover) cell by patching z on real records —
    the fixtures alone only ever produce z = 0 and white-mover z = +1."""
    rec = natural_records[0]
    white_ply, black_ply = rec.moves[0], rec.moves[1]
    assert white_ply.mover == 0 and black_ply.mover == 1
    expected = {   # (z_white, mover) -> (pW, pD, pL) from the mover's view
        (1.0, 0): (1, 0, 0), (1.0, 1): (0, 0, 1),
        (-1.0, 0): (0, 0, 1), (-1.0, 1): (1, 0, 0),
        (0.0, 0): (0, 1, 0), (0.0, 1): (0, 1, 0),
    }
    for z in (1.0, -1.0, 0.0):
        patched = copy.deepcopy(rec)
        patched.z_white = z
        samples = selfplay_samples(patched)
        for ply in (0, 1):
            assert tuple(samples[ply]["wdl"]) == expected[(z, rec.moves[ply].mover)]


def test_aux_targets_follow_phase2_convention(natural_records):
    """The reply at ply i+1, in OUR mover-relative frame — the same
    convention as data.replay_samples (KataGo aux head)."""
    for rec in natural_records:
        for i, s in enumerate(selfplay_samples(rec)):
            if i + 1 < len(rec.moves):
                assert s["aux_w"] == 1.0
                assert s["aux_idx"] == uci_to_index(rec.moves[i + 1].played_uci,
                                                    rec.moves[i].mover)
                assert s["aux_mask"][s["aux_idx"]]
            else:
                assert s["aux_w"] == 0.0    # last ply: no reply exists


def test_samples_at_picks_plies(natural_records):
    rec = natural_records[0]
    full = selfplay_samples(rec)
    picked = samples_at(rec, [0, len(full) - 1, 2])   # walk order: 0, 2, last
    assert len(picked) == 3
    assert (picked[0]["tokens"] == full[0]["tokens"]).all()
    assert (picked[1]["pi_probs"] == full[2]["pi_probs"]).all()
    assert (picked[2]["tokens"] == full[-1]["tokens"]).all()


# ---------------------------------------------------------------------------
# Resignation semantics
# ---------------------------------------------------------------------------

def test_resign_game_shape_and_labels(resign_records):
    assert resign_records
    for rec in resign_records:
        assert rec.outcome_kind == "resign"
        assert rec.resigner == 1                     # black gives up at ply 1
        assert rec.z_white == 1.0                    # resigner loses
        assert len(rec.moves) == 2                   # white's move + resign ply
        assert rec.moves[-1].played_slot == -1
        samples = selfplay_samples(rec)
        # the resign ply itself contributes NO sample
        assert len(samples) == playable_plies(rec) == 1
        # ...and the ply before it has no reply target
        assert samples[0]["aux_w"] == 0.0
        # white's own label: it won (the opponent resigned)
        assert tuple(samples[0]["wdl"]) == (1.0, 0.0, 0.0)
        # π′ of the played (non-resign) ply still round-trips
        assert abs(samples[0]["pi_probs"].sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# The replay window
# ---------------------------------------------------------------------------

def test_buffer_window_and_sampling(natural_records, resign_records):
    records = natural_records + resign_records
    buf = SelfPlayBuffer(max_games=6)
    buf.add_all(records)
    assert len(buf) == 6                              # evicted to the window
    assert buf.total_positions == sum(playable_plies(r)
                                      for r in records[-6:])
    t = buf.sample_tensors(random.Random(7), 32)
    assert t["tokens"].shape == (32, 64, 117)
    assert t["pi_slots"].shape == (32, PI_SUPPORT_MAX)
    assert t["mask"].dtype == bool
    # determinism: same window + same seed replays the same batch
    buf2 = SelfPlayBuffer(max_games=6)
    buf2.add_all(records)
    t2 = buf2.sample_tensors(random.Random(7), 32)
    for k in t:
        assert (t[k] == t2[k]).all()


def test_buffer_rejects_empty():
    with pytest.raises(RuntimeError):
        SelfPlayBuffer(max_games=4).sample_tensors(random.Random(0), 8)


# ---------------------------------------------------------------------------
# Dense ≡ sparse at the loss (Trainer accepts both batch forms)
# ---------------------------------------------------------------------------

def _tiny_net():
    from chess_ai.net.model import ChessTransformer, NetConfig
    return ChessTransformer(NetConfig(d_model=64, n_layers=2, n_heads=4,
                                      d_ff=128))


def test_dense_onehot_equals_sparse_ce(natural_records):
    """π′ one-hot ⇒ target-weighted CE is exactly the Phase 2 sparse CE."""
    pytest.importorskip("torch")
    import torch

    from chess_ai.train.trainer import TrainConfig, Trainer, tensors_to_torch

    tensors = build_selfplay_tensors(natural_records[:2])
    dense = tensors_to_torch(tensors, torch.device("cpu"))
    sparse = {k: v for k, v in dense.items()          # the Phase 2 form:
              if k not in ("pi_slots", "pi_probs")}   # pol_idx only
    oh_slots = dense["pol_idx"][:, None].repeat(1, PI_SUPPORT_MAX)
    oh_probs = torch.zeros_like(dense["pi_probs"])
    oh_probs[:, 0] = 1.0
    onehot = dict(sparse, pi_slots=oh_slots, pi_probs=oh_probs)

    torch.manual_seed(0)      # identical init on both sides (the Trainer's
    tr_sparse = Trainer(_tiny_net(), TrainConfig(lr=0.0, device="cpu"))  # own
    m_sparse = tr_sparse.train_step(sparse)                              # seed
    torch.manual_seed(0)      # lands AFTER the net is built — re-seed here)
    tr_onehot = Trainer(_tiny_net(), TrainConfig(lr=0.0, device="cpu"))
    m_onehot = tr_onehot.train_step(onehot)
    assert abs(m_sparse["policy"] - m_onehot["policy"]) < 1e-6
    assert abs(m_sparse["value"] - m_onehot["value"]) < 1e-9


def test_train_step_learns_on_selfplay_tensors(natural_records):
    """The whole 3.4 seam: shards → records → dense tensors → train_step
    runs, is finite, and the loss comes down on a fixed small set."""
    pytest.importorskip("torch")
    import torch

    from chess_ai.train.trainer import TrainConfig, Trainer, tensors_to_torch

    trainer = Trainer(_tiny_net(), TrainConfig(lr=3e-4, batch_size=16,
                                               device="cpu", seed=1))
    tensors = build_selfplay_tensors(natural_records)
    n = tensors["tokens"].shape[0]
    rng = np.random.default_rng(0)
    first = last = None
    for step in range(12):
        idx = rng.integers(0, n, size=16)
        m = trainer.train_step(
            tensors_to_torch({k: v[idx] for k, v in tensors.items()},
                             torch.device("cpu")))
        assert all(np.isfinite(v) for v in m.values())
        if step == 0:
            first = m["loss"]
        last = m["loss"]
    assert last < first


# ---------------------------------------------------------------------------
# Unit ingestion
# ---------------------------------------------------------------------------

def _write_manifest(unit, seed, games):
    (unit / "DONE.json").write_text(json.dumps({
        "format": "SPSHARD1", "version": 1, "unit_seed": seed,
        "game_budget": games, "n_workers": 2, "games_done": games,
        "eval_spec": "fixture"}))


def test_read_unit_manifest(natural_records, tmp_path):
    """A SpRun-produced dir has no manifest — write one like the binary's
    and check the cross-count assertion + reader round trip."""
    unit = tmp_path / "unit-a"
    unit.mkdir()
    play_records(unit, seed=500, budget=len(natural_records))
    _write_manifest(unit, 500, len(natural_records))
    recs = read_unit(unit)
    assert len(recs) == len(natural_records)
    assert [r.game_index for r in recs] == list(range(len(natural_records)))
    # a lying manifest fails loudly
    (unit / "DONE.json").write_text(json.dumps({
        "unit_seed": 500, "n_workers": 2, "games_done": 999}))
    with pytest.raises(AssertionError):
        read_unit(unit)


def test_ingest_units_dedups(tmp_path):
    """A re-issued unit's later copy supersedes; (seed, game_index) keys
    stay unique across units."""
    units = []
    for u, seed in enumerate((900, 901)):
        unit = tmp_path / f"unit-{u}"
        unit.mkdir()
        play_records(unit, seed=seed, budget=3)
        _write_manifest(unit, seed, 3)
        units.append(unit)
    # a stale duplicate of unit-0 (the partial copy a kill leaves behind)
    dup = tmp_path / "unit-0-copy"
    dup.mkdir()
    for f in units[0].glob("shard-w*.spbin"):
        (dup / f.name).write_bytes(f.read_bytes())
    _write_manifest(dup, 900, 3)
    recs = ingest_units([dup, *units])
    assert len(recs) == 6
    assert {(r.seed, r.game_index) for r in recs} == {
        (900, i) for i in range(3)} | {(901, i) for i in range(3)}


def test_truncated_shard_tail_detected(tmp_path):
    """The format's contract: a killed run leaves only a truncated LAST
    record, which the reader must detect (short read), not accept."""
    unit = tmp_path / "unit-trunc"
    unit.mkdir()
    play_records(unit, seed=321, budget=4)
    shard = shard_path(unit, 0)
    data = shard.read_bytes()
    assert len(data) > 64
    shard.write_bytes(data[: len(data) - 20])   # tear into the last record
    with pytest.raises((struct.error, ValueError, IndexError)):
        read_records(unit, 2, 321)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
