"""
tests/test_pusher_streaming.py — the pusher must slice byte-exact parts
in O(1) memory: one 5090 unit now carries 30+ parts / 30K+ games, and the
old per-poll list(scan_shard) materialized every parsed game and
OOM-died past ~10-11K (SALAD_OPS.md §4). shard.game_extents walks record
headers only; these tests pin its boundaries against scan_shard and run
the real pusher loop end-to-end (HF upload mocked): parts must be
byte-identical to the slices the old algorithm cut.
"""
from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

from chess_ai.selfplay.shard import game_extents, read_shard, scan_shard

SHARD_HDR = struct.Struct("<8sIIQ")
GAME_HDR = struct.Struct("<IIBbId96s")
PLY_V3 = struct.Struct("<B8sHdH")            # mover, uci, n_pi, outlook, sims


def _game(idx: int, pi_widths=(2, 3)) -> bytes:
    plies = b"".join(PLY_V3.pack(0, b"e2e4", w, -0.3, 600) + b"\0\0" * w
                     for w in pi_widths)
    return GAME_HDR.pack(idx, len(pi_widths), 1, -1, 600, 1.0,
                         b"fen" + b"\0" * 93) + plies


def _shard(games, seed: int = 11, trailer: bytes = b"") -> bytes:
    return SHARD_HDR.pack(b"SPSHARD1", 3, 0, seed) + b"".join(games) + trailer


def test_extents_match_scan_shard_boundaries(tmp_path) -> None:
    games = [_game(0), _game(1, (1, 30, 5)), _game(2, (40,))]
    blob = _shard(games) + _game(3)[:14]          # truncated trailing record
    p = tmp_path / "shard-0.spbin"
    p.write_bytes(blob)
    assert list(game_extents(p)) == [(s, e) for _, s, e in scan_shard(p)]
    assert len(read_shard(p)) == 3                 # parser agrees: 3 complete


def test_extents_resume_from_offset(tmp_path) -> None:
    p = tmp_path / "shard-0.spbin"
    p.write_bytes(_shard([_game(i) for i in range(4)]))
    full = list(game_extents(p))
    assert len(full) == 4
    assert list(game_extents(p, start=full[1][0])) == full[1:]


def _load_pusher():
    spec = importlib.util.spec_from_file_location(
        "sp_pusher", Path(__file__).parent.parent / "cloud" / "sp_pusher.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_pusher_slices_byte_exact_and_finishes(tmp_path, monkeypatch) -> None:
    hh = pytest.importorskip("huggingface_hub")
    pusher = _load_pusher()
    uploads = []

    class FakeApi:                                # records calls, uploads nothing
        def __init__(self, token=None):
            pass

        def create_repo(self, *a, **k):
            pass

        def upload_folder(self, *, folder_path, path_in_repo, **k):
            uploads.append((Path(folder_path), path_in_repo))

    monkeypatch.setattr(hh, "HfApi", FakeApi)

    run = tmp_path / "run"
    run.mkdir()
    shard = run / "shard-0.spbin"
    games = [_game(i) for i in range(7)]
    shard.write_bytes(_shard(games[:4]))          # 4 games landed mid-run

    def poll():
        monkeypatch.setattr(
            sys, "argv", ["sp_pusher", str(run), "ds", "units/u",
                          "--every", "4", "--interval", "0", "--once"])
        pusher.main()

    poll()                                        # 4 new ≥ 4: part-0000
    shard.write_bytes(_shard(games))              # 3 more landed
    poll()                                        # 3 new < 4: hold them
    assert len(uploads) == 1
    (run / "DONE.json").write_text(
        json.dumps({"n_workers": 1, "unit_seed": 11, "games_done": 7}))
    poll()                                        # DONE: final tail part
    assert json.loads((run / "pushed.json").read_text()) == \
        {"pushed": 7, "parts": 2}

    # byte-exact regression: each part = header + the exact record ranges
    # the old list(scan_shard) algorithm would have cut
    raw = shard.read_bytes()
    extents = [(s, e) for _, s, e in scan_shard(shard)]
    for k, (count, span) in enumerate(zip((4, 3), (extents[0:4], extents[4:7]))):
        part = run / f"part-{k:04d}"
        a, b = span[0][0], span[-1][1]
        assert (part / "shard-0.spbin").read_bytes() == raw[:24] + raw[a:b]
        assert json.loads((part / "DONE.json").read_text()) == \
            {"n_workers": 1, "unit_seed": 11, "games_done": count}
    assert [u[1] for u in uploads] == ["units/u/part-0000", "units/u/part-0001"]
