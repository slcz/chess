"""
scripts/target_quality.py — how good is each n's self-play move as a
training target? (the pre-check for the deeper-search experiment)

For a sample of REAL positions (drawn from the run's own self-play units),
run the same net's search at several n values in eval mode (the halving
survivor — exactly the Eq. 9 target) and score each against the deepest n
in the list as the reference. Agreement-with-deep-search is the proxy for
target quality: if 64 agrees with the reference far more often than 16/32,
deeper self-play buys better labels; if it saturates early, it doesn't.

    python -m scripts.target_quality --ckpt net.pt --n 16 32 64 256 \
        --positions 120 --units runs/eq9_lr2e5/units --device cuda --bf16
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from chess_ai.engine import Board
from chess_ai.selfplay.csearch import CSpGame
from chess_ai.selfplay.gumbel import SelfPlayConfig
from chess_ai.selfplay.spengine import read_records
from chess_ai.train.selfplay_data import read_unit


def sample_positions(unit_dir: Path, count: int, seed: int = 0) -> list[str]:
    """Random (position, not opening-biased) FENs from completed units:
    replay each record to a uniformly drawn ply."""
    units = sorted(unit_dir.glob("iter-*"))
    records = []
    for d in units:
        records.extend(read_unit(d))
    rng = random.Random(seed)
    rng.shuffle(records)
    fens: list[str] = []
    for rec in records:
        if len(fens) >= count:
            break
        board = Board(rec.start_fen)
        ply = rng.randrange(len(rec.moves))
        for mv in rec.moves[:ply]:
            board.push(mv.played_uci)
        if board.legal_moves:
            fens.append(board.fen)
    return fens


def survivor_at(ev, fen: str, n: int, seed: int) -> str | None:
    """The eval-mode move (halving survivor) from `fen` at n sims."""
    cfg = SelfPlayConfig(n=n, sample_plies=0, resign_value=-2.0)
    game = CSpGame(cfg, seed=seed, game_index=0, start_fen=fen)
    try:
        rec = game.play(ev, max_plies=1)
    finally:
        try:
            game.close()
        except BufferError:
            # CUDA/autocast paths can retain an export of the slot mmap
            # (under investigation); leaking one 2 MiB virtual mapping per
            # call is harmless at bench scale.
            game.g = None
            game._view = None
    if not rec.moves:
        return None
    return rec.moves[0].played_uci


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scripts.target_quality")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--units", default="runs/eq9_lr2e5/units")
    p.add_argument("--n", type=int, nargs="+", default=[16, 32, 64, 256])
    p.add_argument("--positions", type=int, default=120)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    from chess_ai.selfplay.ladder import NetEvaluator
    ev = NetEvaluator(a.ckpt, device=a.device, bf16=a.bf16)

    fens = sample_positions(Path(a.units), a.positions, a.seed)
    print(f"{len(fens)} positions from {a.units}; reference = n={a.n[-1]}")
    moves: dict[int, list[str | None]] = {n: [] for n in a.n}
    for i, fen in enumerate(fens):
        for n in a.n:
            moves[n].append(survivor_at(ev, fen, n, a.seed + i))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(fens)} positions searched")

    ref_n = a.n[-1]
    ref = moves[ref_n]
    print(json.dumps({
        "ckpt": Path(a.ckpt).stem, "positions": len(fens), "reference_n": ref_n,
        "agreement": {
            str(n): round(sum(1 for m, r in zip(moves[n], ref) if m == r)
                          / max(1, sum(1 for r in ref if r)), 4)
            for n in a.n[:-1]
        },
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
