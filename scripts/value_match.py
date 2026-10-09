#!/usr/bin/env python3
"""
scripts/value_match.py — a VALUE-HEAD tournament: both sides play 1-ply
on their value head alone (no policy head, no search) — every legal
move's resulting position is evaluated, and the move is chosen from the
value distribution. Isolates the value head: if a fine-tune wins here
but loses at search, the damage lives in the search x value coupling,
not the head itself.

    python3 scripts/value_match.py <ckpt_a> <ckpt_b> [games=200]
        [temperature=0.20] [max_plys=300] [device=auto]

temperature > 0 samples moves ∝ softmax(value / T) (argmax would
carbon-copy every game from the startpos); colors alternate per game;
engine termination rules adjudicate (threefold auto-draw included), ply
cap = draw. Output: one TOTAL line, score from A's perspective, plus a
per-100 progress line.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_ai.engine import Board, Termination  # noqa: E402
from chess_ai.supervised.play import ValueStrategy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt_a")
    ap.add_argument("ckpt_b")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.20)
    ap.add_argument("--max-plys", type=int, default=300)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    a = ValueStrategy(args.ckpt_a, device=args.device,
                      temperature=args.temperature, seed=11)
    b = ValueStrategy(args.ckpt_b, device=args.device,
                      temperature=args.temperature, seed=22)

    w = d = l = 0
    t0 = time.time()
    for i in range(args.games):
        a_white = (i % 2 == 0)
        board = Board()
        while (board.termination == Termination.PLAYING
               and board.ply < args.max_plys):
            mover_white = board.side_to_move == 0
            strat = a if mover_white == a_white else b
            board.push(strat.choose_move(board))
        term = board.termination
        if term == Termination.CHECKMATE:
            a_won = (board.side_to_move == 1) == a_white   # mover is mated
            sc = 1.0 if a_won else 0.0
        else:
            sc = 0.5
        w += sc == 1.0
        d += sc == 0.5
        l += sc == 0.0
        if (i + 1) % 100 == 0:
            print(f"# {i + 1}/{args.games} score_a={(w + 0.5 * d) / (i + 1):.3f}",
                  flush=True)
    n = args.games
    print(f"TOTAL games={n} w={w} d={d} l={l} "
          f"score_a={(w + 0.5 * d) / n:.3f} T={args.temperature} "
          f"elapsed={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
