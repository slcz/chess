"""
scripts/policy_match.py — raw-policy match between two checkpoints, NO
search (the diagnostic that separates policy damage from search-interaction
damage when a ladder point surprises).

Protocol twin of the ladder's R5: 10 sampled opening plies ∝ the net's own
policy (per-game RNG from the seed), then argmax policy play to a natural
end (ply-capped, cap = draw), every seed played twice with colors
reversed. Both nets run at the same precision (bf16 both seats on CUDA).

    python -m scripts.policy_match --a net.pt --b seed.pt \
        --seeds 32 --device cuda --bf16
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

from chess_ai.engine import Board, Termination
from chess_ai.net.features import uci_to_index
from chess_ai.selfplay.ladder import NetEvaluator


def play_game(eva, evb, a_white: bool, seed: int, sample_plies: int,
              max_plies: int):
    board = Board()
    rng = random.Random((seed << 1) | (0 if a_white else 1))
    ply = 0
    while board.termination == Termination.PLAYING and ply < max_plies:
        ev = eva if (board.side_to_move == 0) == a_white else evb
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
        ply += 1
    if board.winner is None:
        return 0.5
    return 1.0 if (board.winner == 0) == a_white else 0.0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scripts.policy_match")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--seeds", type=int, default=32)
    p.add_argument("--sample-plies", type=int, default=10)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--device", default="cpu")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--jsonl", default=None)
    a = p.parse_args(argv)

    eva = NetEvaluator(a.a, device=a.device, bf16=a.bf16, compile=a.compile)
    evb = NetEvaluator(a.b, device=a.device, bf16=a.bf16, compile=a.compile)
    print(f"policy match (no search): {a.a} vs {a.b}  "
          f"({2 * a.seeds} games, {a.sample_plies} sampled plies)")
    t0 = time.time()
    scores = []
    for seed in range(a.seeds):
        scores.append(play_game(eva, evb, True, seed, a.sample_plies,
                                a.max_plies))
        scores.append(play_game(eva, evb, False, seed, a.sample_plies,
                                a.max_plies))
        if (seed + 1) % 8 == 0:
            print(f"  {len(scores):3d}/{2 * a.seeds} games  "
                  f"score {sum(scores) / len(scores):.3f}  "
                  f"({time.time() - t0:.0f} s)")
    s = sum(scores) / len(scores)
    s_clamped = min(max(s, 0.5 / len(scores)), 1 - 0.5 / len(scores))
    summary = {
        "a": Path(a.a).stem, "b": Path(a.b).stem, "mode": "policy-no-search",
        "games": len(scores), "score_a": round(s, 4),
        "elo_a_minus_b": round(400 * math.log10(s_clamped / (1 - s_clamped)), 1),
        "draws": scores.count(0.5),
    }
    print(json.dumps(summary))
    if a.jsonl:
        out = Path(a.jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
