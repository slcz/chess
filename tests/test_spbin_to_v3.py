"""
tests/test_spbin_to_v3.py — the v1/v2 → v3 converter (scripts/
spbin_to_v3.py, the only legacy reader left in the repo). Builds a small
synthetic v2 shard against REAL positions (startpos, two plies): a dense
π ply (every legal move recorded — self-play shape) and a sparse one
(two visited moves — gauntlet shape), converts, and checks the v3
output: dense probs in engine movegen order, zeros where the source had
no pair, f16 rounding, all other fields verbatim, and the byte savings
of the new layout.

Needs the cffi engine (Board replay derives the legal-move order).
"""
from __future__ import annotations

import importlib.util
import struct
from pathlib import Path

import pytest

from chess_ai.engine import Board
from chess_ai.net.features import uci_to_index
from chess_ai.selfplay.shard import read_shard

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(
    not (ROOT / "engine/build/libchess.dylib").exists()
    and not (ROOT / "engine/build/libchess.so").exists(),
    reason="cffi engine library not built (make -C engine)",
)

_spec = importlib.util.spec_from_file_location(
    "spbin_to_v3", ROOT / "scripts" / "spbin_to_v3.py")
spbin_to_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(spbin_to_v3)

SHARD_HDR = struct.Struct("<8sIIQ")
GAME_HDR = struct.Struct("<IIBbId96s")
PLY_V2 = struct.Struct("<Bh8sHdH")
PAIR = struct.Struct("<Hf")
START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def build_v2() -> bytes:
    """One 2-ply v2 game. Ply 0 (White): dense π, uniform over every
    legal move. Ply 1 (Black): sparse π — legal[0] 0.7, legal[3] 0.3,
    the rest unrecorded (the gauntlet shape that must become zeros)."""
    board = Board(START)
    bodies = []
    for i, (uci, sims) in enumerate((("e2e4", 600), ("e7e5", 100))):
        legal = board.legal_moves
        slots = [uci_to_index(m, board.side_to_move) for m in legal]
        if i == 0:
            pairs = [(s, 1.0 / len(slots)) for s in slots]
        else:
            pairs = [(slots[0], 0.7), (slots[3], 0.3)]
        body = PLY_V2.pack(i, 5, uci.encode(), len(pairs), -0.25 + i, sims)
        body += b"".join(PAIR.pack(s, p) for s, p in pairs)
        bodies.append(body)
        board.push(uci)
    fen96 = START.encode() + b"\0" * (96 - len(START))
    game = GAME_HDR.pack(7, 2, 1, -1, 600, 0.5, fen96)
    return SHARD_HDR.pack(b"SPSHARD1", 2, 0, 11) + game \
        + bodies[0] + bodies[1]


def test_convert_v2_to_v3(tmp_path: Path) -> None:
    src = tmp_path / "shard-0.spbin"
    src.write_bytes(build_v2())
    dst = tmp_path / "out.spbin"
    spbin_to_v3.convert_file(src, dst)          # convert_file runs verify()

    (g,) = read_shard(dst)
    assert g.game_index == 7 and g.n_sims == 600 and g.z_white == 0.5
    assert g.start_fen == START
    p0, p1 = g.plies

    board = Board(START)
    legal0 = board.legal_moves
    assert (p0.mover, p0.played_uci, p0.sims, p0.outlook) == \
        (0, "e2e4", 600, -0.25)
    assert len(p0.pi) == len(legal0)            # dense: every legal move
    for q in p0.pi:
        assert q == pytest.approx(1.0 / len(legal0), rel=1e-3)  # f16 err
    assert sum(p0.pi) == pytest.approx(1.0, abs=2e-3)

    board.push("e2e4")
    assert (p1.mover, p1.played_uci, p1.sims, p1.outlook) == \
        (1, "e7e5", 100, 0.75)
    assert len(p1.pi) == len(board.legal_moves)
    assert p1.pi[0] == pytest.approx(0.7, rel=1e-3)
    assert p1.pi[3] == pytest.approx(0.3, rel=1e-3)
    unrecorded = p1.pi[1:3] + p1.pi[4:]
    assert sum(unrecorded) == 0.0               # sparse tail → exact zeros

    assert len(dst.read_bytes()) < len(src.read_bytes())


def test_v3_input_rejected(tmp_path: Path) -> None:
    p = tmp_path / "x.spbin"
    p.write_bytes(SHARD_HDR.pack(b"SPSHARD1", 3, 0, 11))
    with pytest.raises(ValueError, match="legacy"):
        spbin_to_v3.convert(p.read_bytes(), str(p))
