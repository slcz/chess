#!/usr/bin/env python3
"""spbin_to_v3.py — convert legacy SPSHARD1 shards (v1/v2) to v3.

v3 replaced the sparse π encoding — n_pi × (u16 policy slot, f32 prob)
pairs, plus an i16 played_slot per ply — with dense f16 probs over the
legal moves in ENGINE MOVEGEN order and no slot ids at all (~59% smaller
shards; engine/include/chess_sp_dump.h is the format authority). All
readers in this repo (trainer, pusher, web GUI) are v3-only, so this
script is the ONLY v1/v2 reader left: it carries its own legacy parser.

How the dense vector is rebuilt: each game is replayed from start_fen +
played_uci through the engine; at every ply the ply's legal moves (engine
order) are mapped to policy slots with uci_to_index (mover frame — the
same mapping chess_policy_slot used when writing), and each recorded
(slot, prob) pair lands on its move's index. Legal moves with no recorded
pair get 0.0 — correct for sparse legacy π (the gauntlet recorder skipped
unvisited moves); self-play v2 shards cover every legal move, so their
sum is preserved exactly. probs round f32 → f16 (round-to-nearest-even);
that quantization is the point of v3 and is what training will see.

    python scripts/spbin_to_v3.py IN OUT

IN/OUT are each a shard file or a unit directory. Directory mode
converts every *.spbin inside (recursively) keeping relative names, and
copies DONE.json sidecars. Every output is re-parsed and checked against
its source (game count, per-game plies, per-ply fields) before the
script reports success.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from chess_ai.engine import Board
    from chess_ai.net.features import uci_to_index
    from chess_ai.selfplay.shard import scan_shard as scan_v3
except Exception as e:  # the converter replays through the engine — no
    raise SystemExit(   # way to derive legal-move order without it
        f"cannot import the engine stack ({e}); build the cffi engine first")

# ---- legacy v1/v2 parsing (this script is its only remaining home) ----
_SHARD_HDR = struct.Struct("<8sIIQ")          # magic, version, worker, seed
_GAME_HDR = struct.Struct("<IIBbId96s")       # idx, plies, end, resigner, sims, z, fen
_PLY_HDR = {1: struct.Struct("<Bh8sHd"),      # mover, slot, uci, n_pi, outlook
            2: struct.Struct("<Bh8sHdH")}     # ... + this ply's sims
_PAIR = struct.Struct("<Hf")                  # slot, prob
_PLY_V3 = struct.Struct("<B8sHdH")            # mover, uci, n_pi, outlook, sims


def legacy_games(data: bytes, path: str):
    """Yield parsed v1/v2 games, keeping the raw fen/uci bytes so game
    heads round-trip verbatim. Raises on anything unexpected — a
    converter must not guess."""
    if len(data) < _SHARD_HDR.size:
        raise ValueError(f"{path}: shorter than its header")
    magic, version, _worker, _seed = _SHARD_HDR.unpack_from(data, 0)
    if magic != b"SPSHARD1":
        raise ValueError(f"{path}: bad magic {magic!r}")
    if version not in (1, 2):
        raise ValueError(f"{path}: version {version} is not a legacy (v1/v2) "
                         f"shard — nothing to convert")
    ply_hdr, off = _PLY_HDR[version], _SHARD_HDR.size
    while off + _GAME_HDR.size <= len(data):
        (gidx, n_plies, end_kind, resigner, n_sims, z_white,
         fen_raw) = _GAME_HDR.unpack_from(data, off)
        off += _GAME_HDR.size
        plies = []
        for _ in range(n_plies):
            if off + ply_hdr.size > len(data):
                raise ValueError(f"{path}: truncated game {gidx} — convert a "
                                 f"complete shard (the writer's DONE.json "
                                 f"gates this), not a kill-mid-write file")
            mover, _slot, uci_raw, n_pi, outlook, *extra = \
                ply_hdr.unpack_from(data, off)
            off += ply_hdr.size
            pairs = [_PAIR.unpack_from(data, off + i * _PAIR.size)
                     for i in range(n_pi)]
            off += n_pi * _PAIR.size
            sims = extra[0] if extra else n_sims   # v1: every ply ≡ game cap
            plies.append((mover, uci_raw, outlook, sims, pairs))
        yield (gidx, n_plies, end_kind, resigner, n_sims, z_white,
               fen_raw, plies)


def convert(data: bytes, path: str) -> bytes:
    """One legacy shard's bytes → v3 bytes."""
    magic, _version, worker, seed = _SHARD_HDR.unpack_from(data, 0)
    out = bytearray(_SHARD_HDR.pack(magic, 3, worker, seed))
    for (gidx, n_plies, end_kind, resigner, n_sims, z_white,
         fen_raw, plies) in legacy_games(data, path):
        out += _GAME_HDR.pack(gidx, n_plies, end_kind, resigner, n_sims,
                              z_white, fen_raw)
        board = Board(fen_raw.split(b"\0", 1)[0].decode())
        for mover, uci_raw, outlook, sims, pairs in plies:
            uci = uci_raw.split(b"\0", 1)[0].decode()
            side = board.side_to_move
            if mover != side:
                raise ValueError(f"{path}: game {gidx}: recorded mover "
                                 f"{mover} disagrees with replay ({side})")
            legal = board.legal_moves
            slot_of = [uci_to_index(m, side) for m in legal]
            unknown = {s for s, _ in pairs} - set(slot_of)
            if unknown:
                bad = sorted(unknown)[:4]
                raise ValueError(f"{path}: game {gidx}: π slots {bad}... are "
                                 f"not legal moves at ply {board.ply} — "
                                 f"corrupt shard?")
            by_slot = dict(pairs)
            dense = [by_slot.get(s, 0.0) for s in slot_of]
            out += _PLY_V3.pack(mover, uci_raw, len(legal), outlook, sims)
            out += struct.pack(f"<{len(dense)}e", *dense)
            board.push(uci)
    return bytes(out)


def verify(src: bytes, dst: Path) -> None:
    """Re-parse the v3 output and check it against the source fields."""
    games = [g for g, _, _ in scan_v3(dst)]
    legacy = list(legacy_games(src, str(dst)))
    assert len(games) == len(legacy), \
        f"{dst}: wrote {len(games)} games, source has {len(legacy)}"
    for new, (gidx, n_plies, _e, _r, n_sims, z, _f, old_plies) in \
            zip(games, legacy):
        assert (new.game_index, new.n_plies, new.n_sims) == \
            (gidx, n_plies, n_sims), f"{dst}: game {gidx} head mismatch"
        assert abs(new.z_white - z) < 1e-12, f"{dst}: game {gidx} z mismatch"
        for p, (mover, uci_raw, outlook, sims, _pairs) in \
                zip(new.plies, old_plies):
            assert (p.mover, p.played_uci, p.sims) == \
                (mover, uci_raw.split(b"\0", 1)[0].decode(), sims), \
                f"{dst}: game {gidx} ply fields mismatch"
            assert abs(p.outlook - outlook) < 1e-12


def convert_file(src: Path, dst: Path) -> tuple:
    out = convert(src.read_bytes(), str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(out)
    verify(src.read_bytes(), dst)
    return src.stat().st_size, len(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="legacy .spbin file, or a directory tree "
                                "of them")
    ap.add_argument("dst", help="output .spbin file, or directory tree")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    if src.is_file():
        pairs = [(src, dst)]
    else:
        files = sorted(src.rglob("*.spbin"))
        if not files:
            raise SystemExit(f"no .spbin under {src}")
        pairs = [(f, dst / f.relative_to(src)) for f in files]
        for done in src.rglob("DONE.json"):   # unit marker: copy verbatim
            out = dst / done.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(done.read_bytes())

    in_b = out_b = 0
    for s, d in pairs:
        si, so = convert_file(s, d)
        in_b, out_b = in_b + si, out_b + so
        print(f"[spbin_to_v3] {s} -> {d}  {si:,} -> {so:,} B "
              f"({(so - si) / si:+.1%})", flush=True)
    if len(pairs) > 1:
        print(f"[spbin_to_v3] total {in_b:,} -> {out_b:,} B "
              f"({(out_b - in_b) / in_b:+.1%})", flush=True)


if __name__ == "__main__":
    main()
