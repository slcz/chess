#!/usr/bin/env python3
"""
scripts/sp_replay.py — browse a self-play unit (shard + DONE.json) in the
browser: pick a game from the dataset index (plies, result, ending,
opening move), step through it on the board with the search's per-ply
visit distribution (π) and root value alongside. Read-only replay from
the shard; with --ckpt the current position's top-3 moves are also
heat-colored on the board and a hover tooltip shows each move's π
visits, net prior, and the net's value of the resulting position.

    .venv/bin/python scripts/sp_replay.py /tmp/spbench [--port 8467]
        [--ckpt runs/policy_value_net_42k.pt] [--device auto]
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
    ap.add_argument("--ckpt", help="checkpoint for the live-net overlay: "
                    "top-3 moves heat-colored per position, hover shows "
                    "π visits, net prior, and the child's net value")
    ap.add_argument("--device", help="torch device for --ckpt (auto → MPS/CUDA/CPU)")
    args = ap.parse_args()
    serve(args.unit_dir, args.port, args.ckpt, args.device)


if __name__ == "__main__":
    main()
