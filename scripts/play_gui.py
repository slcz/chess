#!/usr/bin/env python3
"""
scripts/play_gui.py — launch the local chess GUI (chess_ai.gui).

    .venv/bin/python scripts/play_gui.py                          # you vs greedy
    .venv/bin/python scripts/play_gui.py --black random           # you vs random
    .venv/bin/python scripts/play_gui.py --white random --black greedy   # watch
    ENGINE_PATH=/opt/homebrew/bin/stockfish \
        .venv/bin/python scripts/play_gui.py --black engine --movetime 300

Play the trained value net (supervised side track; needs a checkpoint
from chess_ai.supervised.train_supervised train):
    .venv/bin/python scripts/play_gui.py --black value \\
        --value-ckpt runs/value_pilot_2013/final.pt

Watch the self-play nets play each other WITH SEARCH (Gumbel MCTS,
--sims per move; first --sample-plies plies sampled ∝ π′, then the
deterministic survivor; resignation off) — here the supervised baseline
seed (white) vs the latest v3 self-play net (black):
    .venv/bin/python scripts/play_gui.py --white net --black net \\
        --white-ckpt runs/policy_value_net_rope_25k.pt \\
        --black-ckpt /tmp/v3_r70/pi_prime_fix/net-iter-0070.pt \\
        --sims 100

Engines are opponents for play/calibration only (PLAN §10.3) — they never
touch training data, so zero-knowledge is intact.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_ai.gui import CONTROLLERS, serve  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--white", choices=CONTROLLERS, default="human")
    p.add_argument("--black", choices=CONTROLLERS, default="greedy")
    p.add_argument("--port", type=int, default=8466)
    p.add_argument("--seed", type=int, default=None,
                   help="strategy RNG seed (reproducible dummy games)")
    p.add_argument("--movetime", type=int, default=200,
                   help="UCI engine per-move budget in ms")
    p.add_argument("--value-ckpt", default=None,
                   help="checkpoint for the 'value' controller (else $VALUE_CKPT)")
    p.add_argument("--white-ckpt", default=None,
                   help="checkpoint for the white 'net' controller "
                        "(else $NET_CKPT_WHITE / $NET_CKPT)")
    p.add_argument("--black-ckpt", default=None,
                   help="checkpoint for the black 'net' controller "
                        "(else $NET_CKPT_BLACK / $NET_CKPT)")
    p.add_argument("--sims", type=int, default=100,
                   help="'net' controller: Gumbel MCTS simulations per move")
    p.add_argument("--sample-plies", type=int, default=10,
                   help="'net' controller: sample ∝ π′ for this many plies, "
                        "then play the deterministic survivor")
    p.add_argument("--no-open", action="store_true", help="don't open a browser")
    args = p.parse_args()

    if args.value_ckpt:
        os.environ["VALUE_CKPT"] = args.value_ckpt
    if args.white_ckpt:
        os.environ["NET_CKPT_WHITE"] = args.white_ckpt
    if args.black_ckpt:
        os.environ["NET_CKPT_BLACK"] = args.black_ckpt
    os.environ["NET_SIMS"] = str(args.sims)
    os.environ["NET_SAMPLE_PLIES"] = str(args.sample_plies)

    try:
        serve(white=args.white, black=args.black, port=args.port,
              seed=args.seed, movetime_ms=args.movetime,
              open_browser=not args.no_open)
    except RuntimeError as e:  # e.g. engine requested but not installed
        sys.exit(str(e))


if __name__ == "__main__":
    main()
