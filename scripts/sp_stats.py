#!/usr/bin/env python3
"""
scripts/sp_stats.py — distribution report over a self-play work unit
(engine/build/selfplay output directory: shard-*.spbin + DONE.json).

    python3 scripts/sp_stats.py out/units/iter-0001

Reports what the training loop needs to watch per round: outcome split
and endings, game-length shape, the policy target's quality (support
size, top-1 mass, entropy — collapse toward one-hot or mush toward
uniform are both bad), the mover-view root value's distribution (the
KataGo downweight fraction), per-ply-band evolution, and the root value's
sign agreement with the final result (value-head sanity). All stats are
read-only; nothing touches the shard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_ai.selfplay.shard import read_unit  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("unit_dir", help="selfplay output dir (shard + DONE.json)")
    args = ap.parse_args()

    games = read_unit(args.unit_dir)
    print(f"games: {len(games)}  positions: {sum(g.n_plies for g in games)}")

    zw = np.array([g.z_white for g in games])
    ends = np.array([g.end_kind for g in games])
    print("\n== outcomes ==")
    print(f"white win {np.mean(zw == 1.0):.3f} | draw {np.mean(zw == 0.5):.3f} "
          f"| black win {np.mean(zw == 0.0):.3f}")
    print(f"natural {(ends == 1).mean():.3f} | ply-cap adjudicated "
          f"{(ends == 3).mean():.3f}")

    plies = np.array([g.n_plies for g in games])
    print("\n== game length (plies) ==")
    print(f"mean {plies.mean():.1f} median {np.median(plies):.0f} "
          f"p10 {np.percentile(plies, 10):.0f} p90 {np.percentile(plies, 90):.0f}")
    h, e = np.histogram(plies, bins=[0, 20, 40, 60, 80, 100, 150, 200, 250, 301])
    for i in range(len(h)):
        bar = "#" * int(h[i] / max(h) * 40)
        print(f"  {e[i]:>3.0f}-{e[i + 1]:>3.0f}: {h[i]:4d} "
              f"{h[i] / len(games) * 100:4.1f}%  {bar}")

    all_pi = [p for g in games for p in g.plies]
    supp = np.array([len(p.pi) for p in all_pi])
    top1 = np.array([max(pr for _, pr in p.pi) for p in all_pi])
    outlook = np.array([p.outlook for p in all_pi])
    plyno, ent = [], []
    for g in games:
        for i, p in enumerate(g.plies):
            plyno.append(i)
            pr = np.array([q for _, q in p.pi])
            ent.append(-(pr * np.log(pr + 1e-12)).sum())
    plyno, ent = np.array(plyno), np.array(ent)

    print(f"\n== π (policy target) over {len(all_pi)} positions ==")
    print(f"support size: mean {supp.mean():.1f} median {np.median(supp):.0f} "
          f"p90 {np.percentile(supp, 90):.0f} max {supp.max()}")
    h, _ = np.histogram(supp, bins=[0, 2, 4, 8, 16, 32, 64, 128, 257])
    for i, lbl in enumerate(["1", "2-3", "4-7", "8-15", "16-31",
                             "32-63", "64-127", "128+"]):
        print(f"  support {lbl:>7}: {h[i]:6d} {h[i] / len(supp) * 100:5.1f}%")
    print(f"top-1 mass: mean {top1.mean():.3f} median {np.median(top1):.3f} "
          f"| <0.5: {np.mean(top1 < 0.5):.3f} | >0.9: {np.mean(top1 > 0.9):.3f}")
    print(f"π entropy (nats): mean {ent.mean():.2f} median {np.median(ent):.2f}")

    print("\n== outlook (mover-view root value) ==")
    h, e = np.histogram(outlook, bins=[-1.01, -0.9, -0.6, -0.3, 0, 0.3, 0.6, 0.9, 1.01])
    for i in range(len(h)):
        bar = "#" * int(h[i] / max(h) * 40)
        print(f"  [{e[i]:5.2f},{e[i + 1]:5.2f}): {h[i]:6d} "
              f"{h[i] / len(outlook) * 100:5.1f}%  {bar}")
    print(f"KataGo downweight zone (< -0.9): {np.mean(outlook < -0.9):.3f}")

    print("\n== by ply band ==")
    print(f"{'band':>8} {'n':>7} {'supp':>5} {'top1':>5} {'ent':>5} "
          f"{'|out|':>5} {'top1<0.5':>8}")
    for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 100), (100, 160), (160, 300)]:
        m = (plyno >= lo) & (plyno < hi)
        if m.sum() == 0:
            continue
        print(f"{lo:>4}-{hi:<4} {m.sum():7d} {supp[m].mean():5.1f} "
              f"{top1[m].mean():5.3f} {ent[m].mean():5.2f} "
              f"{np.abs(outlook[m]).mean():5.3f} {np.mean(top1[m] < 0.5):8.3f}")

    agree = n = 0
    for g in games:
        if g.z_white == 0.5:
            continue
        for p in g.plies:
            v = p.outlook if p.mover == 0 else -p.outlook   # white-view
            n += 1
            if (v > 0) == (g.z_white == 1.0):
                agree += 1
    if n:
        print(f"\nroot-value sign agrees with final z (decisive games): "
              f"{agree / n:.3f} (n={n})")


if __name__ == "__main__":
    main()
