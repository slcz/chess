#!/usr/bin/env python3
"""gauntlet_sf.py — node-limited Stockfish gauntlet for the PUCT engine.

Plays our UCI adapter (engine/build/uci_puct, PUCT sims fixed by --sims)
against Stockfish limited to a fixed node budget per move, sweeping several
budgets to locate the crossover (where we score ~50%). Colors alternate;
our engine samples from visit counts for the first --sample-plys plies
(variety without a book; set 0 for fully deterministic play). Games are
adjudicated by python-chess with claimable draws (50-move, threefold) plus
a --max-plys cap counted as a draw.

Scores are from OUR engine's perspective: 1 win, 1/2 draw, 0 loss.
"""
import argparse
import concurrent.futures as cf
import os
import time
from pathlib import Path

import chess
import chess.engine

REPO = Path(__file__).resolve().parent.parent
UCI_PUCT = REPO / "engine/build/uci_puct"


def play_game(puct, sf, our_color, sims, sf_nodes, max_plys):
    board = chess.Board()
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_plys:
        if board.turn == our_color:
            lim = chess.engine.Limit(nodes=sims)
            eng = puct
        else:
            lim = chess.engine.Limit(nodes=sf_nodes)
            eng = sf
        board.push(eng.play(board, lim, ponder=False).move)
        plies += 1
    over = board.is_game_over(claim_draw=True)
    res = board.result(claim_draw=True) if over else "1/2-1/2"
    if res == "1/2-1/2":
        return 0.5, res, plies
    we_won = (res == "1-0") == (our_color == chess.WHITE)
    return (1.0 if we_won else 0.0), res, plies


def run_level(job):
    sf_nodes, games, ckpt, device, sims, sample_plys, max_plys = job
    env = dict(os.environ, PUCT_CKPT=ckpt, PUCT_DEVICE=device,
               PUCT_SIMS=str(sims), PUCT_SAMPLE_PLYS=str(sample_plys))
    pts = w = d = l = 0
    with chess.engine.SimpleEngine.popen_uci(
            [str(UCI_PUCT)], env=env, setpgrp=True) as puct, \
         chess.engine.SimpleEngine.popen_uci(
            ["stockfish"], setpgrp=True) as sf:
        sf.configure({"Threads": 1, "Hash": 16})
        for i in range(games):
            our_color = chess.WHITE if i % 2 == 0 else chess.BLACK
            t0 = time.time()
            try:
                p, res, plies = play_game(puct, sf, our_color, sims,
                                          sf_nodes, max_plys)
            except chess.engine.EngineError as e:
                print(f"  sf={sf_nodes} game {i}: ENGINE ERROR {e}",
                      flush=True)
                continue
            pts += p
            w, d, l = w + (p == 1), d + (p == 0.5), l + (p == 0)
            print(f"  sf_nodes={sf_nodes:>6} game {i + 1}/{games}: "
                  f"{'W' if p == 1 else 'D' if p == 0.5 else 'L'} {res} "
                  f"{plies} plys {time.time() - t0:.0f}s", flush=True)
    return sf_nodes, pts, w, d, l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/policy_value_net_42k.pt")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--sims", type=int, default=256)
    ap.add_argument("--sample-plys", type=int, default=10)
    ap.add_argument("--max-plys", type=int, default=300)
    ap.add_argument("--games", type=int, default=8, help="per node level")
    ap.add_argument("--sf-nodes", type=int, nargs="+",
                    default=[100, 1000, 10000, 100000])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ckpt = args.ckpt if args.ckpt.startswith("/") else str(REPO / args.ckpt)
    jobs = [(n, args.games, ckpt, args.device, args.sims,
             args.sample_plys, args.max_plys) for n in args.sf_nodes]

    print(f"gauntlet: sims={args.sims} sample_plys={args.sample_plys} "
          f"games/level={args.games} ckpt={ckpt}", flush=True)
    with cf.ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(run_level, jobs))

    print(f"\n=== gauntlet: PUCT(sims={args.sims}) vs node-limited "
          "Stockfish ===")
    print(f"{'sf_nodes':>8} {'games':>5} {'W':>3} {'D':>3} {'L':>3} "
          f"{'score':>6}")
    for sf_nodes, pts, w, d, l in sorted(results):
        games = w + d + l
        print(f"{sf_nodes:>8} {games:>5} {w:>3} {d:>3} {l:>3} "
              f"{pts / games if games else float('nan'):>6.3f}")


if __name__ == "__main__":
    main()
