"""
chess_ai/selfplay/shard.py — read SPSHARD1 v3 shards written by
engine/build/selfplay (engine/include/chess_sp_dump.h is the byte-format
authority). Deliberately minimal: struct parsing plus a light record type,
nothing else — replaying positions for tensors happens in
chess_ai.train.selfplay_data.

v3 only: π is dense f16 probs over the legal moves in engine movegen
order (no slot ids — they are derivable by replaying the game; the shard
does not depend on the policy-head layout). Legacy v1/v2 shards (sparse
(slot, f32) pairs) must go through scripts/spbin_to_v3.py first.

Kill-safety: records are self-delimiting, so a run killed mid-write has at
most one truncated LAST record — detected as a short read and dropped with
a warning, never silently ingested.
"""
from __future__ import annotations

import json
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

_SHARD_HDR = struct.Struct("<8sIIQ")          # magic, version, worker, seed
_GAME_HDR = struct.Struct("<IIBbId96s")       # idx, plies, end, resigner, sims, z, fen
_PLY_HDR = struct.Struct("<B8sHdH")           # mover, uci, n_pi, outlook, sims
MAGIC = b"SPSHARD1"
VERSIONS = (3,)


@dataclass(frozen=True)
class SpPly:
    mover: int                 # 0 white / 1 black
    played_uci: str
    outlook: float             # mover-view root value
    sims: int                  # this ply's search budget
    pi: Tuple[float, ...]      # dense probs over legal moves, engine
                               # movegen order; sums ~1 (f16 rounding)


@dataclass(frozen=True)
class SpGame:
    game_index: int
    n_plies: int
    end_kind: int              # 1 natural, 3 ply-cap adjudicated
    resigner: int              # always -1 from this driver
    n_sims: int
    z_white: float             # 1 / 0.5 / 0
    start_fen: str
    plies: Tuple[SpPly, ...]


def scan_shard(path: str | Path):
    """Yield (SpGame, start_offset, end_offset) per complete game — the
    offsets let a slicer cut byte-exact parts out of a shard that is
    still being written (the trailing partial record simply never
    yields). Raises on a bad header."""
    data = Path(path).read_bytes()
    if len(data) < _SHARD_HDR.size:
        raise ValueError(f"{path}: shard shorter than its header")
    magic, version, _worker, _seed = _SHARD_HDR.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}")
    if version not in VERSIONS:
        raise ValueError(f"{path}: unsupported shard version {version} "
                         f"(supported: {VERSIONS}; convert v1/v2 with "
                         f"scripts/spbin_to_v3.py)")
    off = _SHARD_HDR.size
    while off + _GAME_HDR.size <= len(data):
        start = off
        (gidx, n_plies, end_kind, resigner, n_sims, z_white,
         fen_raw) = _GAME_HDR.unpack_from(data, off)
        off += _GAME_HDR.size
        plies: List[SpPly] = []
        ok = True
        for _ in range(n_plies):
            if off + _PLY_HDR.size > len(data):
                ok = False
                break
            mover, uci_raw, n_pi, outlook, sims = \
                _PLY_HDR.unpack_from(data, off)
            off += _PLY_HDR.size
            if off + n_pi * 2 > len(data):
                ok = False
                break
            pi = struct.unpack_from(f"<{n_pi}e", data, off)
            off += n_pi * 2
            plies.append(SpPly(mover,
                               uci_raw.split(b"\0", 1)[0].decode(),
                               outlook, sims, pi))
        if not ok:
            break
        yield (SpGame(gidx, n_plies, end_kind, resigner, n_sims,
                      z_white, fen_raw.split(b"\0", 1)[0].decode(),
                      tuple(plies)), start, off)


def read_shard(path: str | Path) -> List[SpGame]:
    """One shard file → game records. Raises on a bad header; drops a
    truncated trailing record (the kill-mid-write case) with a warning."""
    games = []
    consumed = _SHARD_HDR.size
    for g, _, end in scan_shard(path):
        games.append(g)
        consumed = end
    if consumed != Path(path).stat().st_size:
        warnings.warn(f"{path}: truncated trailing game "
                      f"(got {len(games)} complete) — dropped")
    return games


def read_unit(unit_dir: str | Path) -> List[SpGame]:
    """One work-unit directory → its games. DONE.json gates completeness
    (a unit without it is an interrupted run, not data); games dedup by
    (unit_seed, game_index) across shards."""
    unit_dir = Path(unit_dir)
    done = json.loads((unit_dir / "DONE.json").read_text())
    records: List[SpGame] = []
    seen = set()
    for w in range(done["n_workers"]):
        for g in read_shard(unit_dir / f"shard-{w}.spbin"):
            key = (done["unit_seed"], g.game_index)
            if key not in seen:
                seen.add(key)
                records.append(g)
    assert len(records) == done["games_done"], \
        f"{unit_dir}: DONE says {done['games_done']} games, read {len(records)}"
    return records
