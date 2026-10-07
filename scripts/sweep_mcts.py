#!/usr/bin/env python3
"""sweep_mcts.py — MCTS sims sweep vs the raw-policy baseline.

For each n in {32, 64, 128, 256} plays G games of PUCT-MCTS(n) against the
raw policy net (sims=1: one root eval, move from the policy softmax alone),
colors alternating. An n=1 vs n=1 control checks the harness for bias.

Both sides sample their moves for the first 30 plies (search player from
visit counts, policy player from the priors), then argmax — see
engine/src/match_puct.c. Draws count 1/2. Runs GAMES in parallel processes
(MPS is shared; each game is an independent process embedding the net).
"""
import argparse
import concurrent.futures as cf
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent / "engine/build/match_puct"


def play(job):
    n, color, seed, ckpt, sample_plys, max_plys, device = job
    white, black = (n, 1) if color == "w" else (1, n)
    cmd = [str(ENGINE), str(ckpt), str(white), str(black),
           str(sample_plys), str(seed), str(max_plys), str(device)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        return (n, color, seed, "ERROR", r.stderr.strip()[-200:])
    result = next(l for l in r.stdout.splitlines() if l.startswith("RESULT"))
    _, outcome, reason, plies = result.split()
    return (n, color, seed, outcome, f"{reason} {plies}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/policy_value_net_42k.pt")
    ap.add_argument("--games", type=int, default=16,
                    help="games per n, half as white (control uses 8)")
    ap.add_argument("--sample-plys", type=int, default=30)
    ap.add_argument("--max-plys", type=int, default=250)
    ap.add_argument("--sims", type=int, nargs="+",
                    default=[32, 64, 128, 256])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=2000,
                    help="opening-sampling seeds are seed-base + 100*n + i")
    ap.add_argument("--device", default="mps",
                    help="net device for the embedded evaluator")
    args = ap.parse_args()

    ckpt = str(Path(args.ckpt).resolve())
    jobs = [(1, c, 1000 + i, ckpt, args.sample_plys, args.max_plys,
             args.device)
            for i, c in enumerate(["w", "b"] * 4)]
    for n in args.sims:
        for i in range(args.games):
            c = "w" if i % 2 == 0 else "b"
            jobs.append((n, c, args.seed_base + 100 * n + i, ckpt,
                         args.sample_plys, args.max_plys, args.device))

    print(f"{len(jobs)} games, {args.workers} workers, ckpt={ckpt}",
          flush=True)
    rows = []
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for row in ex.map(play, jobs):
            rows.append(row)
            print(f"  n={row[0]:>3} {row[1]} seed={row[2]}: "
                  f"{row[3]} ({row[4]})", flush=True)

    print("\n=== sweep: MCTS(n) vs raw policy (score = (W + D/2)/games) ===")
    print(f"{'n':>4} {'games':>5} {'W':>3} {'D':>3} {'L':>3} {'score':>6}")
    for n in [1] + args.sims:
        sub = [r for r in rows if r[0] == n and r[3] != "ERROR"]
        if not sub:
            continue
        full = {"w": "white", "b": "black"}
        pts = sum(1.0 if r[3] == f"{full[r[1]]}_win" else
                  0.5 if r[3] == "draw" else 0.0 for r in sub)
        w = sum(r[3] == f"{full[r[1]]}_win" for r in sub)
        d = sum(r[3] == "draw" for r in sub)
        print(f"{n:>4} {len(sub):>5} {w:>3} {d:>3} "
              f"{len(sub) - w - d:>3} {pts / len(sub):>6.3f}")
    errs = [r for r in rows if r[3] == "ERROR"]
    for r in errs:
        print(f"ERROR n={r[0]} seed={r[2]}: {r[4]}")


if __name__ == "__main__":
    main()
