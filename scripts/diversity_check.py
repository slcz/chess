#!/usr/bin/env python3
"""diversity_check.py — are sampled games actually diversified?

Re-runs a few match_puct games per config, captures the full move list and
reports, per config:
  - distinct first moves / distinct first-12-ply openings
  - mean pairwise move agreement over the first 30 plies (1.0 = same game)
  - number of fully identical games

A sample_plys=0 control is included: with sampling off every game must be
byte-identical, which validates that the diversity above comes from the
sampling knob and not from nondeterminism elsewhere (MPS reduction order
etc. could otherwise silently randomize play).
"""
import argparse
import concurrent.futures as cf
import subprocess
from collections import Counter
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "engine/build/match_puct"


def moves_of(job):
    ckpt, w, b, sample_plys, seed = job
    r = subprocess.run(
        [str(ENGINE), str(ckpt), str(w), str(b),
         str(sample_plys), str(seed), "250"],
        capture_output=True, text=True, timeout=1800, check=True)
    moves = r.stdout.splitlines()[0].split()
    return moves


def agreement(games, k):
    pairs = same = total = 0
    for i in range(len(games)):
        for j in range(i + 1, len(games)):
            a, b = games[i][:k], games[j][:k]
            n = min(len(a), len(b))
            if n == 0:
                continue
            same += sum(x == y for x, y in zip(a, b))
            total += n
            pairs += 1
    return same / total if total else float("nan"), pairs


def report(name, games, k_open=12, k_agree=30):
    print(f"\n{name}  ({len(games)} games)")
    print(f"  first moves: {dict(Counter(g[0] for g in games))}")
    print(f"  distinct first-{k_open}-ply openings: "
          f"{len({tuple(g[:k_open]) for g in games})}/{len(games)}")
    ag, pairs = agreement(games, k_agree)
    print(f"  mean pairwise agreement over first {k_agree} plys: "
          f"{ag:.3f}  ({pairs} pairs)")
    print(f"  fully identical games: {len(games) - len({tuple(g) for g in games})}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=900000)
    args = ap.parse_args()
    ckpt = str(Path(args.ckpt).resolve())

    blocks = [
        ("MCTS(256) vs raw, sample_plys=30",
         [(ckpt, 256, 1, 30, args.seed_base + i) for i in range(args.games)]),
        ("MCTS(64) vs raw, sample_plys=30",
         [(ckpt, 64, 1, 30, args.seed_base + 100 + i) for i in range(args.games)]),
        ("control: raw vs raw, sample_plys=0 (must be identical)",
         [(ckpt, 1, 1, 0, args.seed_base + 200 + i) for i in range(3)]),
    ]
    jobs = [j for _, js in blocks for j in js]
    with cf.ThreadPoolExecutor(4) as ex:
        all_moves = list(ex.map(moves_of, jobs))

    at = 0
    for name, js in blocks:
        report(name, all_moves[at:at + len(js)])
        at += len(js)


if __name__ == "__main__":
    main()
