#!/usr/bin/env python3
"""
scripts/sp_stats.py — distribution report over a self-play work unit
(engine/build/selfplay output directory: shard-*.spbin + DONE.json).

    python3 scripts/sp_stats.py out/units/iter-0001
    python3 scripts/sp_stats.py out/units/iter-0001 --plot dup.png

Reports what the training loop needs to watch per round: outcome split
and endings, game-length shape, the policy target's quality (support
size, top-1 mass, entropy — collapse toward one-hot or mush toward
uniform are both bad), the mover-view root value's distribution (the
KataGo downweight fraction), per-ply-band evolution, the root value's
sign agreement with the final result (value-head sanity), and exact
sample duplication (blake2b of the full 64x117 tokens — placement plus
8-ply history, so a hash match is a byte-identical training input; the
label-variance table shows whether duplicates carry identical or
genuinely different z). All stats are read-only; nothing touches the
shard. --plot writes a 3-panel duplication figure (matplotlib).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_ai.engine import Board  # noqa: E402
from chess_ai.net.features import index_to_uci  # noqa: E402
from chess_ai.selfplay.shard import read_unit  # noqa: E402


def dup_report(games, plot: str | None = None) -> None:
    """Exact-sample duplication: hash every replayed position's tokens."""
    hashes, plys, gidx, zs = [], [], [], []
    amoves, atops = [], []          # per-instance π argmax slot, top-1 prob
    movers = []
    for g in games:
        b = Board(g.start_fen)
        for ply in range(g.n_plies):
            hashes.append(
                hashlib.blake2b(b.encode().tobytes(), digest_size=16).digest())
            plys.append(ply)
            gidx.append(g.game_index)
            zs.append(g.z_white)
            movers.append(g.plies[ply].mover)
            pi = g.plies[ply].pi        # tuple of (slot, prob)
            amoves.append(max(pi, key=lambda kv: kv[1])[0])
            atops.append(max(p for _, p in pi))
            b.push(g.plies[ply].played_uci)
    hashes_n = np.array([int.from_bytes(h, "little") for h in hashes])
    plys = np.array(plys)
    counts = Counter(hashes)
    mult = np.array([counts[h] for h in hashes])
    n_total, n_uniq = len(hashes), len(counts)

    print(f"\n== duplication (hash of full tokens) ==")
    print(f"total samples {n_total} | unique {n_uniq} "
          f"({n_uniq / n_total * 100:.1f}%) | duplicated samples "
          f"{(mult > 1).sum()} ({(mult > 1).mean() * 100:.1f}%)")
    print(f"effective unique positions per game: {n_uniq / len(games):.1f}")

    inst = defaultdict(list)
    for i, h in enumerate(hashes):
        inst[h].append(i)
    within = sum(1 for idxs in inst.values()
                 if len(idxs) > 1 and len({gidx[i] for i in idxs}) == 1)
    cross = sum(1 for idxs in inst.values()
                if len(idxs) > 1 and len({gidx[i] for i in idxs}) > 1)
    print(f"duplicated distinct positions: {cross} cross-game, "
          f"{within} within-game only")

    dist = Counter(counts.values())
    print("multiplicity -> #distinct positions (top rows):")
    for k in sorted(dist, reverse=True)[:8]:
        print(f"  x{k:<5d} {dist[k]:6d}")
    print(f"  (bulk: x1 {dist[1]} distinct, "
          f"x2..x5 {sum(dist[k] for k in range(2, 6))} distinct)")

    print("per-ply duplicated fraction (every 4th ply, first 40):")
    for p in range(0, 40, 4):
        m = plys == p
        if m.sum() == 0:
            break
        bar = "#" * int((mult[m] > 1).mean() * 40)
        print(f"  ply {p:>2}: n={m.sum():5d} dup={(mult[m] > 1).mean() * 100:5.1f}% {bar}")

    print("label variance of the most-duplicated positions "
          "(z varies = value signal, not redundancy):")
    print(f"{'count':>5} {'ply':>3}  {'z W/D/L':>13}  {'π argmax':>8}  "
          f"{'stable':>7}  {'top1':>5}")
    for h, idxs in sorted(inst.items(), key=lambda kv: -len(kv[1]))[:6]:
        zs_i = np.array([zs[i] for i in idxs])
        am = Counter(amoves[i] for i in idxs).most_common(1)[0]
        uci = index_to_uci(am[0], movers[idxs[0]])
        print(f"{len(idxs):5d} {plys[idxs[0]]:3d}  "
              f"{np.mean(zs_i == 1):.2f}/{np.mean(zs_i == .5):.2f}/"
              f"{np.mean(zs_i == 0):.2f}  {uci:>8}  {am[1]}/{len(idxs):<5d} "
              f"{np.mean([atops[i] for i in idxs]):5.3f}")

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        ks = sorted(dist)
        axes[0].bar([str(k) for k in ks], [dist[k] for k in ks], color="#4477aa")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("multiplicity")
        axes[0].set_ylabel("# distinct positions (log)")
        axes[0].set_title(f"{n_total} samples -> {n_uniq} unique "
                          f"({n_uniq / n_total * 100:.0f}%)")
        xs = [p for p in range(60) if (plys == p).sum()]
        axes[1].bar(xs, [(mult[plys == p] > 1).mean() for p in xs], color="#ee6677")
        axes[1].set_xlabel("ply")
        axes[1].set_ylabel("fraction duplicated")
        axes[1].set_title("duplication rate by ply")
        axes[1].set_ylim(0, 1)
        axes[2].bar(xs, [np.log2(mult[plys == p]).mean() for p in xs], color="#228833")
        axes[2].set_xlabel("ply")
        axes[2].set_ylabel("mean log2 multiplicity")
        axes[2].set_title("mean effective copies by ply")
        fig.suptitle("self-play unit: exact-sample duplication "
                     f"({len(games)} games, blake2b of 64x117 tokens)", y=1.03)
        fig.tight_layout()
        fig.savefig(plot, dpi=120, bbox_inches="tight")
        print(f"plot saved: {plot}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("unit_dir", help="selfplay output dir (shard + DONE.json)")
    ap.add_argument("--plot", metavar="PNG",
                    help="also write the 3-panel duplication figure")
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

    dup_report(games, plot=args.plot)


if __name__ == "__main__":
    main()
