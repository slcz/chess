#!/usr/bin/env python3
"""
scripts/play_demo.py — play around with the engine end-to-end (the "dummy
strategy" demo): C rules → termination → FFI → strategies → match loop,
and optionally a real UCI engine opponent.

Usage (from the repo root):
    .venv/bin/python scripts/play_demo.py                     # greedy vs random
    ENGINE_PATH=/opt/homebrew/bin/stockfish \
        .venv/bin/python scripts/play_demo.py                 # + engine game

The engine part looks for ENGINE_PATH, then `stockfish` on PATH. Engines
are *opponents/calibration only* (PLAN §1, §10.3) — they never touch
training data, so zero-knowledge is intact.
"""
from __future__ import annotations  # 3.9: allow PEP 604 type annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_ai.engine import Board, Termination          # noqa: E402
from chess_ai.match import play_game, play_match        # noqa: E402
from chess_ai.strategies import GreedyMaterialStrategy, RandomStrategy  # noqa: E402


def demo_rules_and_ffi():
    """A 60-second tour of the FFI surface the strategies sit on."""
    print("== FFI tour ==")
    b = Board()
    print("startpos, 20 legal moves:", len(b.legal_moves) == 20)
    for mv in ("e2e4", "e7e5", "g1f3"):
        b.push(mv)
    print("after 1.e4 e5 2.Nf3:", b.fen)
    b.pop()
    print("after pop (back to 1.e4 e5):", b.fen)
    print("perft(4) through FFI:", b.perft(4),
          "(matches the C CLI: 728887 from the 1.e4 e5 position)")

    m = Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    print("fool's mate position:", m.termination.name,
          "— winner:", "black" if m.winner == 1 else "?")
    print()


def demo_dummy_match():
    print("== greedy vs random, 6 games (colors alternate) ==")
    score = play_match(GreedyMaterialStrategy(seed=1), RandomStrategy(seed=2),
                       n_games=6, verbose=True)
    print(f"greedy: {score['a_wins']}  random: {score['b_wins']}  "
          f"draws: {score['draws']}")
    print()


def demo_ucf_engine_game(engine_path: str):
    from chess_ai.uci import UCIStrategy
    print(f"== UCI engine ({os.path.basename(engine_path)}, 100ms/move) "
          f"vs greedy, 1 game ==")
    engine = UCIStrategy(engine_path, movetime_ms=100)
    try:
        result = play_game(engine, GreedyMaterialStrategy(seed=3))
    finally:
        engine.close()
    moves = result.moves
    pretty = " ".join(
        f"{i // 2 + 1}.{m}" if i % 2 == 0 else m for i, m in enumerate(moves))
    print(pretty or "(no moves)")
    print("result:", result.describe(), f"({result.plies} plies)")
    print()


def find_engine() -> str | None:
    env = os.environ.get("ENGINE_PATH")
    if env and Path(env).exists():
        return env
    return shutil.which("stockfish")


def main() -> None:
    demo_rules_and_ffi()
    demo_dummy_match()
    engine = find_engine()
    if engine:
        demo_ucf_engine_game(engine)
    else:
        print("(no UCI engine found — set ENGINE_PATH or `brew install "
              "stockfish` to add the engine game)")


if __name__ == "__main__":
    main()
