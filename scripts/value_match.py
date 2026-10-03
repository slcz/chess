"""
scripts/value_match.py — value-head-only match between two checkpoints:
NO search, NO policy head. Each seat plays 1-ply value-greedy — evaluate
EVERY legal move's child position with the WDL head and play the move
that is best for itself. The direct A/B of value-head health: raw policy
play never consults the value head, searched play leans on it everywhere
(σ in π′, the Eq. 10 fallback) — this match isolates it completely.

Protocol twin of policy_match: seeds × 2 color-paired games, per-game RNG;
the first 10 plies sample ∝ softmax(score/T) (opening diversity), then
deterministic value-argmax. Games adjudicate as draws at --max-plies.
The value head reads each CHILD in the child-mover's frame (the encoder is
mover-relative), so the mover's score for a move is −v(child) = pL − pW
of the child — the search's own sign convention (child value negated).

    python -m scripts.value_match --a net.pt --b seed.pt --seeds 32 --device mps
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from chess_ai.engine import Board, Termination
from chess_ai.net.features import legal_move_indices, policy_mask
from chess_ai.selfplay.ladder import NetEvaluator


def _value_scores(ev, board) -> tuple[list[str], list[float]]:
    """Legal moves with the mover's value for playing each (one batched
    WDL forward over all candidate children)."""
    moves = board.legal_moves
    xs, masks = [], []
    for u in moves:
        board.push(u)
        xs.append(board.encode().reshape(64, 117))
        masks.append(policy_mask(legal_move_indices(board)))
        board.pop()
    x = torch.from_numpy(np.stack(xs).astype(np.float32)).to(ev._device)
    m = torch.from_numpy(np.stack(masks)).to(ev._device)
    with torch.no_grad():
        out = ev.model(x, m)
    p = torch.softmax(out["wdl"].float(), dim=-1).cpu().numpy()
    return moves, [float(p[2] - p[0]) for p in p]   # −v(child), mover's view


def play_game(eva, evb, a_white: bool, seed: int, sample_plies: int,
              max_plies: int, temp: float) -> dict:
    board = Board()
    rng = random.Random((seed << 1) | (0 if a_white else 1))
    ply = 0
    t0 = time.perf_counter()
    while board.termination == Termination.PLAYING and ply < max_plies:
        ev = eva if (board.side_to_move == 0) == a_white else evb
        moves, scores = _value_scores(ev, board)
        if ply < sample_plies:
            probs = [math.exp(s / temp) for s in scores]
            total = sum(probs)
            r, acc, mv = rng.random() * total, 0.0, moves[-1]
            for u, w in zip(moves, probs):
                acc += w
                if r < acc:
                    mv = u
                    break
        else:
            mv = moves[max(range(len(moves)), key=lambda i: scores[i])]
        board.push(mv)
        ply += 1
    winner = board.winner
    return {"seed": seed, "a_white": a_white,
            "winner": winner, "plies": ply,
            "kind": "adjudicated" if winner is None and ply >= max_plies
                    else "natural",
            "a_score": 0.5 if winner is None
                       else (1.0 if (winner == 0) == a_white else 0.0),
            "s": round(time.perf_counter() - t0, 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m scripts.value_match",
        description="Value-head-only match: 1-ply value-greedy seats, "
                    "no search, no policy.")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--sample-plies", type=int, default=10)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--temp", type=float, default=0.5,
                   help="opening-sampling temperature on the value scores")
    p.add_argument("--device", default="mps")
    p.add_argument("--jsonl", default=None)
    a = p.parse_args(argv)

    eva = NetEvaluator(a.a, device=a.device)
    evb = NetEvaluator(a.b, device=a.device)
    print(f"value-head match (no search, no policy): {a.a} vs {a.b}  "
          f"({2 * a.seeds} games, {a.sample_plies} sampled plies at "
          f"T={a.temp})", flush=True)
    games, t0 = [], time.time()
    for seed in range(a.seeds):
        for a_white in (True, False):
            g = play_game(eva, evb, a_white, seed, a.sample_plies,
                          a.max_plies, a.temp)
            games.append(g)
            print(f"  game {len(games):3d}/{2 * a.seeds}  seed {g['seed']:2d}  "
                  f"A={'white' if g['a_white'] else 'black'}  "
                  f"winner={g['winner']} ({g['kind']}, {g['plies']} plies)  "
                  f"A-score {g['a_score']:.1f}  ({g['s']} s)", flush=True)
    s = sum(g["a_score"] for g in games) / len(games)
    s_c = min(max(s, 0.5 / len(games)), 1 - 0.5 / len(games))
    summary = {"a": Path(a.a).stem, "b": Path(a.b).stem,
               "mode": "value-head-only", "games": len(games),
               "score_a": round(s, 4),
               "elo_a_minus_b": round(400 * math.log10(s_c / (1 - s_c)), 1),
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
