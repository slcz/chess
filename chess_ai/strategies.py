"""
chess_ai/strategies.py — dummy players over the C engine via FFI.

These exist to exercise the whole stack (rules → termination → FFI → game
loop) before any search/NN exists, and to give the self-play pipeline its
first opponents. Real strength comes later (MCTS + network, Phase 2–3);
here we only need players that (a) always return legal moves and (b) are
simple enough to reason about when a match result looks wrong.

A Strategy is any callable `choose_move(board) -> uci_str` given a
chess_ai.engine.Board; it must return one of board.legal_moves.
"""
from __future__ import annotations

import random

from chess_ai.engine import Board, Termination

# Piece codes on the C side: color*6 + type with PAWN=0..KING=5, EMPTY=12.
PIECE_TYPE = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]

# Classic "point" values, in centipawn-ish units. Only used by the greedy
# dummy for capture preferences — nothing statistical rides on them.
PIECE_VALUES = {"p": 100, "n": 320, "b": 330, "r": 500, "q": 900, "k": 0}
TYPE_LETTER = "pnbrqk"


def _square(name: str) -> int:
    """UCI square name ("e4") -> engine square index (a1=0 … h8=63)."""
    return (ord(name[1]) - ord("1")) * 8 + (ord(name[0]) - ord("a"))


def _center_bonus(sq_name: str) -> int:
    """Tiny bonus for moving toward the board center (distance in files
    and ranks from the middle, scaled). Purely heuristic garnish."""
    f, r = ord(sq_name[0]) - ord("a"), ord(sq_name[1]) - ord("1")
    return 7 - (abs(2 * f - 7) + abs(2 * r - 7))


class RandomStrategy:
    """Uniform random legal move. The floor: any trained net must beat it."""

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: Board) -> str:
        return self.rng.choice(board.legal_moves)


class GreedyMaterialStrategy:
    """A one-ply materialist: take the best capture, else shuffle toward the
    center. It also takes an immediate checkmate when it sees one (a free
    one-ply search that makes demo games finish like real games).

    Deliberately dumb in known ways — it cannot see that a capture is
    recaptured, that walking into check loses, nothing about king safety.
    That makes it a useful fixed reference opponent for gauges like
    "does the net at checkpoint N beat greedy by more than checkpoint M?".
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def choose_move(self, board: Board) -> str:
        moves = board.legal_moves
        us = board.side_to_move
        them = 1 - us

        def victim_value(uci: str) -> int:
            pc = board.piece_at(_square(uci[2:4]))
            if pc >= 12:
                return 0  # quiet move (en-passant victims read as quiet here;
                          # the dummy just doesn't prioritize ep — fine)
            return PIECE_VALUES[TYPE_LETTER[pc % 6]] if pc // 6 == them else 0

        # 1. Immediate checkmates win the game on the spot.
        for uci in moves:
            board.push(uci)
            mate = board.termination == Termination.CHECKMATE
            board.pop()
            if mate:
                return uci

        # 2. Best capture (most valuable victim), ties broken randomly.
        best = max(victim_value(m) for m in moves)
        if best > 0:
            picks = [m for m in moves if victim_value(m) == best]
            return self.rng.choice(picks)

        # 3. No capture: prefer central squares, break ties randomly.
        top = max(_center_bonus(m[2:4]) for m in moves)
        picks = [m for m in moves if _center_bonus(m[2:4]) == top]
        return self.rng.choice(picks)
