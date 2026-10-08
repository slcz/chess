#!/usr/bin/env python3
"""n_sweep_agreement.py — find the optimal sim count n by search convergence.

Protocol (agreed with the user):
  1. Positions: fresh self-play games (MCTS-256 vs raw policy, sampled
     openings — the sweep config), replayed ply by ply, sampled every 8th
     ply (stratified over opening/middlegame/endgame).
  2. For each position, a FRESH search at each level n in {0, 32, ..., 512}
     (new tree per level — levels are independent); n = 0 is the raw-policy
     argmax. Records the argmax-visits move per (position, level).
  3. Metrics:
       - adjacent agreement A(n-32, n): how often stepping n up changes
         nothing. Rises toward 1 as search saturates.
       - reference agreement A(n, 512): how often the level-n move matches
         the largest-n answer (robust to flip-flopping between levels).
  The optimal n is where adjacent agreement flattens (~0.95), balanced
  against cost, which is linear in n.

Outputs: runs/n_sweep_agreement.csv (raw rows) and
         runs/n_sweep_agreement.png (chart).
"""
import argparse
import concurrent.futures as cf
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chess_ai.engine import Board

REPO = Path(__file__).resolve().parent.parent
MATCH = REPO / "engine/build/match_puct"
PROBE = REPO / "engine/build/puct_probe"
LEVELS = list(range(0, 513, 32))


def collect_positions(args):
    """Play games, replay move lists, return ply-stratified FENs."""
    jobs = [(args.ckpt, 256, 1, 30, args.seed_base + i)
            for i in range(args.games)]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        game_moves = list(ex.map(play_moves, jobs))

    fens = []
    for moves in game_moves:
        b = Board()
        plies = []
        for u in moves:
            if b.ply >= 8 and b.ply % 8 == 0:
                plies.append(b.fen)
            b.push(u)
        if len(plies) > args.per_game:      # evenly spread across the game
            idx = [round(i * (len(plies) - 1) / (args.per_game - 1))
                   for i in range(args.per_game)]
            plies = [plies[i] for i in sorted(set(idx))]
        fens.extend(plies)
    print(f"{len(game_moves)} games -> {len(fens)} probe positions "
          f"(every 8th ply, <= {args.per_game}/game)", flush=True)
    return fens


def play_moves(job):
    ckpt, w, b_, sample_plys, seed = job
    r = subprocess.run(
        [str(MATCH), str(ckpt), str(w), str(b_), str(sample_plys),
         str(seed), "250"],
        capture_output=True, text=True, timeout=1800, check=True)
    return r.stdout.splitlines()[0].split()


def probe_chunk(job):
    fens, ckpt, device, levels = job
    r = subprocess.run(
        [str(PROBE), str(ckpt), ",".join(map(str, levels)), device],
        input="\n".join(fens) + "\n",
        capture_output=True, text=True, timeout=7200, check=True)
    rows = []
    for line in r.stdout.splitlines():
        if line.startswith("MOVE"):
            _, pos, n, uci, visits, val = line.split()
            rows.append((int(pos), int(n), uci, int(visits), float(val)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--per-game", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=910000)
    args = ap.parse_args()
    ckpt = str((REPO / args.ckpt).resolve()
               if not args.ckpt.startswith("/") else Path(args.ckpt))

    fens = collect_positions(args)
    chunks = [fens[i::args.workers] for i in range(args.workers)]
    jobs = [(c, ckpt, args.device, LEVELS) for c in chunks if c]
    with cf.ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(probe_chunk, jobs))

    # chunk ci got fens[ci::workers]; its rows number positions locally,
    # so re-key to global indices by walking chunks in order
    moves = defaultdict(dict)          # global pos -> n -> uci
    offset = 0
    for chunk, rows in zip([c for c in chunks if c], results):
        for pos, n, uci, _, _ in rows:
            moves[offset + pos][n] = uci
        offset += len(chunk)

    out_csv = REPO / "runs/n_sweep_agreement.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pos", "n", "move"])
        for pos in sorted(moves):
            for n in LEVELS:
                w.writerow([pos, n, moves[pos].get(n, "")])

    def agree(n_a, n_b):
        common = [p for p in moves if n_a in moves[p] and n_b in moves[p]]
        return (sum(moves[p][n_a] == moves[p][n_b] for p in common)
                / len(common) if common else float("nan"))

    adj = [(n, agree(n - 32, n)) for n in LEVELS[1:]]
    top = LEVELS[-1]
    ref = [(n, agree(n, top)) for n in LEVELS if n < top]
    ref.append((0, agree(0, top)))

    print(f"\nn      A(n-32,n)   A(n,{top})")
    for n, a in adj:
        print(f"{n:>4}  {a:>9.3f}   {dict(ref).get(n, float('nan')):>9.3f}")
    print(f"raw policy (n=0) agrees with n={top}: "
          f"{dict(ref)[0]:.3f}")

    out_png = REPO / "runs/n_sweep_agreement.png"
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot([n for n, _ in adj], [a for _, a in adj], "o-",
            label="adjacent agreement A(n−32, n)")
    ax.plot([n for n, _ in ref], [a for _, a in ref], "s--",
            label=f"agreement with n={top} reference")
    ax.axhline(0.95, color="grey", lw=0.8, ls=":",
               label="0.95 guide")
    ax.set_xlabel("sims per move n")
    ax.set_ylabel("move agreement (fraction of positions)")
    ax.set_title("Search convergence: does more search change the move?\n"
                 f"{len(moves)} self-play positions, {args.ckpt}")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"\nwrote {out_csv}\nwrote {out_png}")


if __name__ == "__main__":
    main()
