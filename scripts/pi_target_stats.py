#!/usr/bin/env python3
"""Compare the dumped policy targets of two rounds ply-by-ply.

Built for the PI_TARGET=visits vs PI_TARGET=learn comparison: run
selfplay twice over the same distribution (same SIMS/FAST_SIMS/seed
family) and point this at the two unit directories. Prints, per trainable
ply, the target's support size / max mass / entropy and the KL divergence
between the two, plus how often each target's argmax is the played move.

    scripts/pi_target_stats.py /tmp/pi-visits /tmp/pi-learn --max-games 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chess_ai.selfplay.shard import read_unit      # noqa: E402


def _targets(game):
    """[(dense target, played slot)] over the game's trainable plies."""
    out = []
    for p in game.plies:
        if p.sims != game.n_sims:
            continue
        t = np.zeros(4240, dtype=np.float64)
        for slot, prob in p.pi:
            t[slot] = prob
        out.append((t / t.sum(), p.played_slot))
    return out


def stats(t):
    support = int((t > 0).sum())
    ent = float(-(t[t > 0] * np.log(t[t > 0])).sum())
    return support, float(t.max()), ent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")                      # e.g. PI_TARGET=visits
    ap.add_argument("dir_b")                      # e.g. PI_TARGET=learn
    ap.add_argument("--max-games", type=int, default=0)
    args = ap.parse_args()

    ga = read_unit(args.dir_a)
    gb = read_unit(args.dir_b)
    if args.max_games:
        ga, gb = ga[: args.max_games], gb[: args.max_games]
    n = min(len(ga), len(gb))
    rows, kls, agree, played_a, played_b = [], [], [], [], []
    for i in range(n):
        ta, tb = _targets(ga[i]), _targets(gb[i])
        for (a, played), (b, _) in zip(ta, tb):
            sa, ma, ea = stats(a)
            sb, mb, eb = stats(b)
            rows.append((sa, ma, ea, sb, mb, eb))
            kls.append(float((a[a > 0] * np.log(a[a > 0] / np.maximum(b[a > 0], 1e-12))).sum()))
            agree.append(float(a.argmax() == b.argmax()))
            played_a.append(float(a.argmax() == played))
            played_b.append(float(b.argmax() == played))
    if not rows:
        print("no trainable plies found")
        return 1
    r = np.array(rows)
    print(f"ply pairs: {len(rows)} over {n} games")
    print(f"{'':18s} {'support':>8s} {'max mass':>9s} {'entropy':>8s}")
    print(f"{'A (visits)':18s} {r[:,0].mean():8.2f} {r[:,1].mean():9.3f} {r[:,2].mean():8.3f}")
    print(f"{'B (learn)':18s} {r[:,3].mean():8.2f} {r[:,4].mean():9.3f} {r[:,5].mean():8.3f}")
    print(f"\nKL(A||B) mean {np.mean(kls):.3f}   top-1 agreement {np.mean(agree):.3f}")
    print(f"argmax == played move:  A {np.mean(played_a):.3f}   B {np.mean(played_b):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
