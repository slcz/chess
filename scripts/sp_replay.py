#!/usr/bin/env python3
"""
scripts/sp_replay.py — browse a self-play unit (shard + DONE.json) in the
browser: pick a game from the dataset index (plies, result, ending,
opening move), step through it on the board with the search's per-ply
visit distribution (π) and root value alongside. Read-only — pure replay
from the shard, no engine search, no checkpoint.

    .venv/bin/python scripts/sp_replay.py /tmp/spbench [--port 8467]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_ai.selfplay.replay import serve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("unit_dir", help="selfplay output dir (shard + DONE.json)")
    ap.add_argument("--port", type=int, default=8467)
    args = ap.parse_args()
    serve(args.unit_dir, args.port)


if __name__ == "__main__":
    main()
