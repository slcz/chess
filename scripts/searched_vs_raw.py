"""
scripts/searched_vs_raw.py — one checkpoint, searched seat vs raw-policy
seat (the strength gap the search itself adds, measured directly).

Seats, both from the SAME checkpoint:
  searched   the R5 gate protocol: Gumbel n=160 via SelfPlayGame
             (10 sampled opening plies ∝ π′, then the deterministic
             halving survivor), resignation OFF;
  raw        the policy_match protocol, NO search: 10 sampled opening
             plies ∝ the net's own policy (per-game RNG), then argmax
             policy.

Every seed is played twice with colors reversed (the ladder's pairing),
so first-move advantage cancels. Games adjudicate as draws at --max-plies.

    python -m scripts.searched_vs_raw --ckpt runs/n64_eq9_node_latest.pt \
        --games 3 --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from chess_ai.engine import Termination
from chess_ai.net.features import uci_to_index
from chess_ai.selfplay.gumbel import SelfPlayConfig, SelfPlayGame
from chess_ai.selfplay.ladder import NetEvaluator


def _raw_move(ev, game, rng, ply: int, sample_plies: int):
    """One raw-policy move pushed directly on the game's board."""
    board = game.board
    side = board.side_to_move
    idx = [uci_to_index(u, side) for u in board.legal_moves]
    logp, _ = ev.evaluate(board.encode(), idx, side)
    moves = board.legal_moves
    if ply < sample_plies:
        probs = [math.exp(logp[i]) for i in idx]
        total = sum(probs)
        r, acc, mv = rng.random() * total, 0.0, moves[-1]
        for u, p in zip(moves, probs):
            acc += p
            if r < acc:
                mv = u
                break
    else:
        mv = max(moves, key=lambda u: logp[uci_to_index(u, side)])
    board.push(mv)


def play_game(ev, a_white: bool, seed: int, cfg: SelfPlayConfig,
              sample_plies: int, max_plies: int) -> dict:
    """Searched seat = A; raw seat = B. Returns the game summary."""
    game = SelfPlayGame(cfg, seed=0, game_index=seed)
    rng = random.Random((seed << 1) | (0 if a_white else 1))
    ply = 0
    t0 = time.perf_counter()
    while game.board.termination == Termination.PLAYING and ply < max_plies:
        searched_to_move = (game.board.side_to_move == 0) == a_white
        if searched_to_move:
            game._play_one_move(ev, None)     # fresh tree each move (no F4)
        else:
            _raw_move(ev, game, rng, ply, sample_plies)
        ply += 1
    kind = ("adjudicated" if game.board.termination == Termination.PLAYING
            else "natural")
    winner = game.board.winner
    a_score = 0.5 if winner is None else (1.0 if (winner == 0) == a_white
                                          else 0.0)
    return {"seed": seed, "a_white": a_white, "z_white": winner,
            "plies": ply, "outcome_kind": kind, "a_score": a_score,
            "s": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m scripts.searched_vs_raw",
        description="Searched (Gumbel n=160) vs raw-policy seat, same ckpt.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seeds", type=int, default=3,
                   help="seeds per color pairing (games = 2 × seeds)")
    p.add_argument("--n", type=int, default=160, help="searched sims/move")
    p.add_argument("--sample-plies", type=int, default=10)
    p.add_argument("--max-plies", type=int, default=256)
    p.add_argument("--device", default="mps")
    p.add_argument("--jsonl", default=None)
    a = p.parse_args(argv)

    ev = NetEvaluator(a.ckpt, device=a.device)
    cfg = SelfPlayConfig(n=a.n, sample_plies=a.sample_plies,
                         resign_value=-2.0)   # off: play to a real result
    print(f"searched (n={a.n}) vs raw policy, ckpt {a.ckpt}  "
          f"({2 * a.seeds} games, resignation off, cap {a.max_plies})",
          flush=True)
    games, t0 = [], time.time()
    for seed in range(a.seeds):
        for a_white in (True, False):
            g = play_game(ev, a_white, seed, cfg, a.sample_plies, a.max_plies)
            games.append(g)
            print(f"  game {len(games):2d}/{2 * a.seeds}  seed {g['seed']}  "
                  f"A={'white' if g['a_white'] else 'black'}  "
                  f"z={g['z_white']} ({g['outcome_kind']}, {g['plies']} plies)"
                  f"  A-score {g['a_score']:.1f}  ({g['s']} s)", flush=True)
    s = sum(g["a_score"] for g in games) / len(games)
    s_c = min(max(s, 0.5 / len(games)), 1 - 0.5 / len(games))
    summary = {"ckpt": a.ckpt, "mode": "searched-vs-raw", "n": a.n,
               "games": len(games), "score_searched": round(s, 4),
               "elo_searched_minus_raw":
                   round(400 * math.log10(s_c / (1 - s_c)), 1),
               "draws": sum(1 for g in games if g["a_score"] == 0.5),
               "s": round(time.time() - t0, 1)}
    print(json.dumps(summary), flush=True)
    if a.jsonl:
        out = Path(a.jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                **summary}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
