"""
scripts/gumbel_py_match.py — the PYTHON Gumbel reference (chess_ai/
selfplay/gumbel.py, the 3.2a oracle) as an evaluation engine.

Same R5 protocol as the C ladder (n sims per move, 10-ply ∝π′ seed
phase, deterministic survivor thereafter, resignation off, natural
termination) but ZERO C search: pure-Python tree, batch-1 MPS evals.
Purpose: the C-vs-Python cross-check on a full match — the L2 gate
pins byte equality; this verifies it empirically at the matchup level
(specifically: does py-Gumbel at c_scale=1.0 reproduce the C ladder's
0.25 for v3-r70 vs seed?).

    python -m scripts.gumbel_py_match --a CKPT --b CKPT --games 2
    (parallel-friendly: --seed-start, one process per core like az_eval)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from chess_ai.selfplay.gumbel import SelfPlayConfig, SelfPlayGame
from chess_ai.selfplay.ladder import NetEvaluator, SeatEvaluator


def play_game(eva, evb, a_white: bool, seed: int, cfg, max_plies) -> dict:
    white = eva if a_white else evb
    black = evb if a_white else eva
    game = SelfPlayGame(cfg, seed=0, game_index=seed)
    t0 = time.perf_counter()
    rec = game.play(SeatEvaluator(white=white, black=black),
                    max_plies=max_plies)
    z = rec.z_white
    return {"seed": seed, "a_white": a_white, "z_white": z,
            "kind": rec.outcome_kind, "plies": len(rec.moves),
            "a_score": 0.5 if z == 0 else (1.0 if (z > 0) == a_white else 0.0),
            "s": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m scripts.gumbel_py_match")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--seeds", type=int, default=1,
                    help="seed-color pairs this process plays (2 games each)")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--c-scale", type=float, default=1.0)
    ap.add_argument("--sample-plies", type=int, default=10)
    ap.add_argument("--max-plies", type=int, default=256)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--jsonl", default="runs/eval/gumbel_py.jsonl")
    a = ap.parse_args(argv)

    eva = NetEvaluator(a.a, device=a.device)
    evb = NetEvaluator(a.b, device=a.device)
    cfg = SelfPlayConfig(n=a.n, c_scale=a.c_scale,
                         sample_plies=a.sample_plies, resign_value=-2.0)
    print(f"py-Gumbel match: {a.a} vs {a.b}  (n={a.n}, c_scale={a.c_scale}, "
          f"{2 * a.seeds} games)", flush=True)
    games = []
    for seed in range(a.seed_start, a.seed_start + a.seeds):
        for a_white in (True, False):
            g = play_game(eva, evb, a_white, seed, cfg, a.max_plies)
            games.append(g)
            print(f"  game {len(games):2d}/{2 * a.seeds}  seed {g['seed']}  "
                  f"A={'white' if g['a_white'] else 'black'}  z={g['z_white']:+.0f} "
                  f"({g['kind']}, {g['plies']} plies)  A-score {g['a_score']:.1f}  "
                  f"({g['s']} s)", flush=True)
    s = sum(g["a_score"] for g in games) / len(games)
    summary = {"a": Path(a.a).stem, "b": Path(a.b).stem,
               "mode": "gumbel-py", "n": a.n, "c_scale": a.c_scale,
               "games": len(games), "score_a": round(s, 4),
               "draws": sum(1 for g in games if g["a_score"] == 0.5)}
    print(json.dumps(summary), flush=True)
    out = Path(a.jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            **summary}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
