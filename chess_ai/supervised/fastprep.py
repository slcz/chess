"""
chess_ai/supervised/fastprep.py — parallel pack generation (side track).

The serial preprocess spends ~all its time on SAN→UCI replay of games
that header filters then discard (~70% at min-Elo 1600 on 2016-era
dumps), in one thread. Two stacked fixes:

1. header-only filters run BEFORE movetext parsing (pgn.iter_pgn_games'
   `header_filter` hook) — the parse budget drops to the kept games.
2. the decompressed dump is sharded across worker processes at EXACT
   game boundaries: a single C-speed byte scan for b"\\n[Event " finds
   every game-start offset (~2 s for 2.5 GB), so shards never split a
   game and the ordered concatenation of shards is byte-identical to
   the serial pack (asserted in tests).

Expected on the 5090 node: 600k-game pack ~15 min -> ~2 min (the zstd
decompress dominates). Same filters, same stats json, same pack bytes.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from chess_ai.supervised.dataset import RESULT_SCORES

_DEF_WORKERS = min(16, os.cpu_count() or 4)


# ---------------------------------------------------------------------------
# Stats plumbing (plain dicts — forked workers must return picklables)
# ---------------------------------------------------------------------------

def _empty_stats() -> Dict:
    return {"seen": 0, "kept": 0, "positions": 0,
            "outcomes": {"white": 0, "draw": 0, "black": 0},
            "elo_sum": 0, "elo_n": 0, "elo_min": 9999, "elo_max": 0,
            "dropped": {"parse_error": 0, "no_result": 0, "unrated": 0,
                        "low_elo": 0, "time_control": 0, "ply_range": 0,
                        "time_forfeit": 0, "custom_start": 0}}


def _merge(a: Dict, b: Dict) -> Dict:
    a["seen"] += b["seen"]
    a["kept"] += b["kept"]
    a["positions"] += b["positions"]
    for k in a["outcomes"]:
        a["outcomes"][k] += b["outcomes"][k]
    a["elo_sum"] += b["elo_sum"]
    a["elo_n"] += b["elo_n"]
    a["elo_min"] = min(a["elo_min"], b["elo_min"])
    a["elo_max"] = max(a["elo_max"], b["elo_max"])
    for k in a["dropped"]:
        a["dropped"][k] += b["dropped"][k]
    return a


def _finalize(s: Dict) -> Dict:
    d = dict(s)
    d["avg_elo"] = round(s["elo_sum"] / max(1, s["elo_n"]), 1)
    d.pop("elo_sum")
    d.pop("elo_n")
    return d


# ---------------------------------------------------------------------------
# Plain-file staging + boundary scan
# ---------------------------------------------------------------------------

def _to_plain(pgn_path: str, workdir: Path) -> Path:
    """Decompress .zst/.gz to a plain file (plain paths pass through)."""
    plain = workdir / (Path(pgn_path).name.split(".pgn")[0] + ".plain.pgn")
    if plain.exists():
        return plain
    if pgn_path.endswith(".zst"):
        with open(plain, "wb") as out:
            subprocess.run(["zstd", "-dc", pgn_path], stdout=out, check=True)
    elif pgn_path.endswith(".gz"):
        import gzip
        import shutil
        with gzip.open(pgn_path, "rb") as src, open(plain, "wb") as out:
            shutil.copyfileobj(src, out)
    else:
        return Path(pgn_path)
    return plain


def _game_cuts(plain: Path, n: int) -> List[int]:
    """Byte offsets of n game starts, evenly spread + file end.

    Lichess dumps lead every header block with [Event; b"\\n[Event " is
    an exact game boundary. Returns n+1 offsets → ranges [cuts[i],
    cuts[i+1]). We sample every len(markers)//n-th marker so shards are
    balanced by game count, not byte count (parse cost is per game).
    """
    size = plain.stat().st_size
    markers: List[int] = []
    with open(plain, "rb") as f:
        pos = 0
        marker = b"\n[Event "
        while True:
            f.seek(pos)
            chunk = f.read(64 << 20)
            if not chunk:
                break
            at = 0
            while True:
                at = chunk.find(marker, at)
                if at < 0:
                    break
                markers.append(pos + at + 1)
                at += 1
            if len(chunk) <= len(marker):
                break                # tail chunk: scanned, marker can't span
            pos += len(chunk) - len(marker) + 1
    if not markers:
        return [0, size]
    step = max(1, len(markers) // n)
    cuts = [0] + markers[step - 1::step][:n - 1] + [size]
    return sorted(set(cuts))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _parse_shard(args) -> Dict:
    """Parse one [start, end) byte range of the plain dump (forked proc).

    Reads BINARY (text-mode seek only accepts tell() cookies; shard
    offsets are raw bytes) and decodes lines on the way into the same
    iter_pgn_games framing the serial path uses.
    """
    (plain_path, start, end, shard_path, filters) = args
    from chess_ai.supervised.pgn import iter_pgn_games

    stats = _empty_stats()

    def _elo(tag, default=None):
        try:
            return int(tag)
        except (TypeError, ValueError):
            return default

    def header_filter(headers: Dict[str, str]) -> bool:
        # Header-only drops happen BEFORE movetext parsing — the speed.
        result = headers.get("Result")
        if result not in RESULT_SCORES:
            stats["dropped"]["no_result"] += 1
            return False
        if headers.get("SetUp") == "1" or headers.get("FEN"):
            stats["dropped"]["custom_start"] += 1
            return False
        we = _elo(headers.get("WhiteElo"))
        be = _elo(headers.get("BlackElo"))
        if we is None or be is None:
            stats["dropped"]["unrated"] += 1
            return False
        if we < filters["min_elo"] or be < filters["min_elo"]:
            stats["dropped"]["low_elo"] += 1
            return False
        base = headers.get("TimeControl", "?").split("+")[0]
        if not base.isdigit() or int(base) < filters["min_base_sec"]:
            stats["dropped"]["time_control"] += 1
            return False
        if filters["drop_time_forfeits"] and headers.get("Termination") != "Normal":
            stats["dropped"]["time_forfeit"] += 1
            return False
        return True

    with open(shard_path, "w", encoding="utf-8") as out, \
            open(plain_path, "rb") as f:
        f.seek(start)
        pos = start
        while pos < end:
            lines: List[str] = []
            for raw in f:
                pos += len(raw)
                lines.append(raw.decode("utf-8", "replace"))
                if pos >= end:
                    break
            # `end` sits on a game boundary; if the final line belongs
            # to the NEXT game, its headers accumulate but never finish
            # inside this shard — dropped here, owned by the next worker.
            for headers, moves in iter_pgn_games(lines, header_filter=header_filter):
                stats["seen"] += 1
                if not (filters["min_plies"] <= len(moves) <= filters["max_plies"]):
                    stats["dropped"]["ply_range"] += 1
                    continue
                score = RESULT_SCORES[headers["Result"]]
                token = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}[score]
                out.write(f"{token}\t{' '.join(moves)}\n")
                stats["kept"] += 1
                stats["positions"] += len(moves)
                stats["outcomes"]["white" if score > 0
                                  else "black" if score < 0 else "draw"] += 1
                for tag in ("WhiteElo", "BlackElo"):
                    v = _elo(headers.get(tag), 0)
                    stats["elo_sum"] += v
                    stats["elo_n"] += 1
                    stats["elo_min"] = min(stats["elo_min"], v)
                    stats["elo_max"] = max(stats["elo_max"], v)
                cap = filters["max_games"] or filters["max_positions"]
                if cap and (stats["kept"] >= filters["max_games"]
                            or stats["positions"] >= filters["max_positions"]):
                    return stats
    return stats


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def preprocess_parallel(
    pgn_path: str,
    out_path: str,
    workers: int = _DEF_WORKERS,
    min_elo: int = 1600,
    min_base_sec: int = 180,
    min_plies: int = 10,
    max_plies: int = 300,
    drop_time_forfeits: bool = False,
    max_games: int = 0,
    max_positions: int = 0,
    keep_plain: bool = False,
) -> dict:
    """Parallel equivalent of dataset.preprocess — same pack, same stats.

    Note: with max_games/max_positions, workers stop at per-shard caps
    (total ≈ the requested cap, not exact — serial preprocess for
    exactness; the fast path is for the big harvests)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    filters = dict(min_elo=min_elo, min_base_sec=min_base_sec,
                   min_plies=min_plies, max_plies=max_plies,
                   drop_time_forfeits=drop_time_forfeits,
                   max_games=max_games // workers if max_games else 0,
                   max_positions=max_positions // workers if max_positions else 0)

    workdir = Path(pgn_path).parent
    plain = _to_plain(pgn_path, workdir)
    cuts = _game_cuts(plain, workers)
    n = len(cuts) - 1
    shard_paths = [f"{out_path}.shard{i}" for i in range(n)]
    jobs = [(str(plain), cuts[i], cuts[i + 1], shard_paths[i], filters)
            for i in range(n)]

    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
        shard_stats = list(pool.map(_parse_shard, jobs))

    total = _empty_stats()
    for s in shard_stats:
        _merge(total, s)

    with open(out_path, "w", encoding="utf-8") as out:
        for sp in shard_paths:
            with open(sp, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
    for sp in shard_paths:
        os.remove(sp)
    if not keep_plain and str(plain) != pgn_path:
        os.remove(plain)

    meta = {
        "source": pgn_path, "workers": n,
        "filters": {"min_elo": min_elo, "min_base_sec": min_base_sec,
                    "min_plies": min_plies, "max_plies": max_plies,
                    "drop_time_forfeits": drop_time_forfeits},
        "stats": _finalize(total),
    }
    Path(str(out_path) + ".stats.json").write_text(json.dumps(meta, indent=2))
    print(f"pack written: {out_path}  workers={n}  "
          f"kept={total['kept']} positions={total['positions']}")
    return meta
