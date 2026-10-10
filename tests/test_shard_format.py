"""
tests/test_shard_format.py — SPSHARD1 reader format gates, no engine
binary or checkpoint needed. Hand-builds shard bytes (the layout is
spelled out in engine/include/chess_sp_dump.h):

  v1  pre-PCR layout — must still parse, every ply.sims filled with the
      game's n_sims so all plies look trainable (the back-compat promise
      to the units already on HuggingFace);
  v2  the current layout — per-ply sims round-trips;
  v3  unknown version — must raise, not guess.
"""
from __future__ import annotations

import struct

import pytest

from chess_ai.selfplay.shard import read_shard
from chess_ai.train.selfplay_data import ply_trainable

SHARD_HDR = struct.Struct("<8sIIQ")
GAME_HDR = struct.Struct("<IIBbId96s")
PLY_V1 = struct.Struct("<Bh8sHd")
PLY_V2 = struct.Struct("<Bh8sHdH")
PAIR = struct.Struct("<Hf")


def shard_bytes(version: int) -> bytes:
    v1_ply = PLY_V1.pack(0, 100, b"e2e4", 1, -0.3) + PAIR.pack(100, 1.0)
    v2_ply = (PLY_V2.pack(0, 100, b"e2e4", 1, -0.3, 50)
              + PAIR.pack(100, 1.0))
    plies = (v1_ply + v1_ply) if version == 1 else (v2_ply + v2_ply)
    game = GAME_HDR.pack(7, 2, 1, -1, 600, 1.0, b"fen" + b"\0" * 93) + plies
    return SHARD_HDR.pack(b"SPSHARD1", version, 0, 11) + game


def write_shard(tmp_path, version: int):
    p = tmp_path / f"v{version}.spbin"
    p.write_bytes(shard_bytes(version))
    return p


def test_v1_parses_all_plies_trainable(tmp_path) -> None:
    (g,) = read_shard(write_shard(tmp_path, 1))
    assert g.game_index == 7 and g.n_sims == 600 and g.n_plies == 2
    for p in g.plies:
        assert p.sims == 600 and ply_trainable(g, p)


def test_v2_per_ply_sims(tmp_path) -> None:
    (g,) = read_shard(write_shard(tmp_path, 2))
    for p in g.plies:
        assert p.sims == 50 and not ply_trainable(g, p)


def test_unknown_version_raises(tmp_path) -> None:
    with pytest.raises(ValueError, match="version"):
        read_shard(write_shard(tmp_path, 3))
