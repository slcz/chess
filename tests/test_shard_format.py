"""
tests/test_shard_format.py — SPSHARD1 v3 reader format gates, no engine
binary or checkpoint needed. Hand-builds shard bytes (the layout is
spelled out in engine/include/chess_sp_dump.h):

  v3  the only supported layout — dense f16 π probs over the legal moves
      in engine movegen order, per-ply sims round-trips, no played_slot
      (move identities come from replaying played_uci);
  v1/v2  legacy layouts — must raise, not guess (convert first with
      scripts/spbin_to_v3.py).
"""
from __future__ import annotations

import struct

import pytest

from chess_ai.selfplay.shard import read_shard
from chess_ai.train.selfplay_data import ply_trainable

SHARD_HDR = struct.Struct("<8sIIQ")
GAME_HDR = struct.Struct("<IIBbId96s")
PLY_V3 = struct.Struct("<B8sHdH")            # mover, uci, n_pi, outlook, sims


def shard_bytes(version: int = 3, sims: int = 50) -> bytes:
    ply = (PLY_V3.pack(0, b"e2e4", 2, -0.3, sims)
           + struct.pack("<2e", 0.25, 0.75))
    game = GAME_HDR.pack(7, 2, 1, -1, 600, 1.0, b"fen" + b"\0" * 93) \
        + ply + ply
    return SHARD_HDR.pack(b"SPSHARD1", version, 0, 11) + game


def write_shard(tmp_path, version: int = 3, sims: int = 50):
    name = f"v{version}-{sims}.spbin"
    p = tmp_path / name
    p.write_bytes(shard_bytes(version, sims))
    return p


def test_v3_dense_pi_per_ply_sims(tmp_path) -> None:
    (g,) = read_shard(write_shard(tmp_path))
    assert g.game_index == 7 and g.n_sims == 600 and g.n_plies == 2
    for p in g.plies:
        assert not hasattr(p, "played_slot")   # slot ids are gone in v3
        assert p.played_uci == "e2e4" and p.outlook == -0.3
        assert p.sims == 50 and not ply_trainable(g, p)   # fast cap
        # 0.25/0.75 are f16-exact; π is bare probs, engine movegen order
        assert p.pi == (0.25, 0.75)


def test_v3_slow_cap_ply_trainable(tmp_path) -> None:
    (g,) = read_shard(write_shard(tmp_path, sims=600))
    for p in g.plies:
        assert p.sims == 600 and ply_trainable(g, p)


@pytest.mark.parametrize("version", [1, 2, 4])
def test_unsupported_version_raises(tmp_path, version) -> None:
    with pytest.raises(ValueError, match="version"):
        read_shard(write_shard(tmp_path, version))
