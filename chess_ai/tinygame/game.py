"""chess_ai/tinygame/game.py — tic-tac-toe: the game + the perfect-play oracle.

The game class mirrors the subset of the chess `Board` interface the Phase 3
loop actually consumes (push/pop, legal_moves, termination, winner, ply,
side_to_move, mover-relative encode) so the search port (search.py) keeps the
same shape as `chess_ai.selfplay.gumbel`:

    chess Board                         TicTacToe
    -------------------------------     --------------------------------
    side_to_move 0 white / 1 black      0 = X (first) / 1 = O
    legal_moves (FFI order)             empty cells, ASCENDING cell order —
                                        this is the pinned "movegen order"
                                        the RNG spec and every tie-break run
                                        over (gumbel.py uses FFI order; here
                                        the natural order IS deterministic)
    push(uci) / pop()                   push(cell) / pop()
    Termination.CHECKMATE               WIN (the side to move just lost —
                                        same semantics: mover is mated)
    Termination.STALEMATE / draws       DRAW (board full; ttt has no other
                                        draw kinds)
    encode() 64×117, mover-relative     encode() 18, mover-relative:
                                        [my 9 one-hots | opp 9 one-hots]
                                        ("ours first", same convention as
                                        features.py's piece ordering). ttt
                                        needs no history/flags: the position
                                        is a pure function of the cells.

The ORACLE is exhaustive minimax over the reachable-position graph (the
classic count: 5478 reachable positions). `value(cells)` is the game value
from the side to move's view (+1 mover wins / 0 draw / −1 mover loses with
perfect play both sides); `optimal_moves` are the value-preserving moves.
This is what makes "perfect play" in the 3.5 gate an exact predicate, not a
statistic.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

PLAYING = "PLAYING"
WIN = "WIN"        # terminal: the side that just moved made a line
DRAW = "DRAW"      # terminal: board full, no line

EMPTY, X, O = 0, 1, 2
NN_INPUT_DIM = 18

# The 8 winning lines as cell triples (row-major 0..8).
LINES: Tuple[Tuple[int, int, int], ...] = tuple(
    [(r * 3, r * 3 + 1, r * 3 + 2) for r in range(3)]
    + [(c, 3 + c, 6 + c) for c in range(3)]
    + [(0, 4, 8), (2, 4, 6)]
)

_TO_CHAR = {EMPTY: ".", X: "x", O: "o"}


class TicTacToe:
    """One position + the make/unmake machinery. Zero dynamic state beyond
    the undo stack, mirroring the C engine's fixed-undo design."""

    __slots__ = ("cells", "_undo")

    def __init__(self, cells: Optional[Tuple[int, ...]] = None):
        if cells is None:
            cells = (EMPTY,) * 9
        assert len(cells) == 9 and all(c in (EMPTY, X, O) for c in cells)
        nx, no = cells.count(X), cells.count(O)
        assert nx == no or nx == no + 1, "unreachable cell counts"
        self.cells = tuple(cells)
        self._undo: List[int] = []          # pushed cell indices, in order

    # -- the Board-shaped interface -----------------------------------------

    @property
    def side_to_move(self) -> int:
        """0 = X, 1 = O — X moves on even plies."""
        return (self.cells.count(X) - self.cells.count(O)) % 2

    @property
    def ply(self) -> int:
        return self.cells.count(X) + self.cells.count(O)

    @property
    def legal_moves(self) -> Tuple[int, ...]:
        """Empty cells, ascending — the pinned enumeration order."""
        if self.termination != PLAYING:
            return ()
        return tuple(i for i, c in enumerate(self.cells) if c == EMPTY)

    @property
    def termination(self) -> str:
        if _line_holder(self.cells) != EMPTY:
            return WIN
        if EMPTY not in self.cells:
            return DRAW
        return PLAYING

    @property
    def winner(self) -> Optional[int]:
        """X/O if the game is a WIN; None otherwise (mirrors Board.winner)."""
        holder = _line_holder(self.cells)
        return holder if holder != EMPTY else None

    def push(self, cell: int) -> None:
        assert self.cells[cell] == EMPTY, "push onto an occupied cell"
        piece = X if self.side_to_move == 0 else O
        self.cells = self.cells[:cell] + (piece,) + self.cells[cell + 1:]
        self._undo.append(cell)

    def pop(self) -> None:
        assert self._undo, "pop on an empty undo stack"
        cell = self._undo.pop()
        lst = list(self.cells)
        lst[cell] = EMPTY
        self.cells = tuple(lst)

    def encode(self) -> np.ndarray:
        """[18] f32, current mover's view: [my pieces | opponent's]."""
        me = X if self.side_to_move == 0 else O
        t = np.zeros(NN_INPUT_DIM, dtype=np.float32)
        for i, c in enumerate(self.cells):
            if c != EMPTY:
                t[i if c == me else 9 + i] = 1.0
        return t

    def fen(self) -> str:
        """Position id for records/tests: "xx.oo../w"-style, 10 chars."""
        return "".join(_TO_CHAR[c] for c in self.cells) + "/xo"[self.side_to_move]


def _line_holder(cells: Tuple[int, ...]) -> int:
    """X/O if some line is fully that player's, else EMPTY."""
    for a, b, c in LINES:
        if cells[a] != EMPTY and cells[a] == cells[b] == cells[c]:
            return cells[a]
    return EMPTY


# ---------------------------------------------------------------------------
# The oracle: exhaustive minimax over reachable positions
# ---------------------------------------------------------------------------

def reachable_positions() -> List[Tuple[int, ...]]:
    """Every non-terminal position reachable in real play (BFS from empty,
    not descending into terminals — the game stops there). 5478 positions."""
    seen = {(EMPTY,) * 9}
    frontier = [(EMPTY,) * 9]
    while frontier:
        nxt = []
        for cells in frontier:
            board = TicTacToe(cells)
            for m in board.legal_moves:
                board.push(m)
                child = board.cells
                if child not in seen:
                    seen.add(child)
                    if TicTacToe(child).termination == PLAYING:
                        nxt.append(child)
                board.pop()
        frontier = nxt
    return sorted(seen)


@lru_cache(maxsize=None)
def value(cells: Tuple[int, ...]) -> int:
    """Game value from the SIDE TO MOVE's view: +1 / 0 / −1, perfect play.

    Standard negamax: max over legal moves of −value(child). A non-terminal
    position always has a legal move in ttt (not full, no line)."""
    assert TicTacToe(cells).termination == PLAYING, "value of a terminal"
    board = TicTacToe(cells)
    best = -2
    for m in board.legal_moves:
        board.push(m)
        if board.termination == WIN:
            v = 1                          # opponent (now to move) is mated
        elif board.termination == DRAW:
            v = 0
        else:
            v = -value(board.cells)
        board.pop()
        best = max(best, v)
    return best


def optimal_moves(cells: Tuple[int, ...]) -> Tuple[int, ...]:
    """Value-preserving moves: exactly the perfect player's choice set."""
    board = TicTacToe(cells)
    out = []
    for m in board.legal_moves:
        board.push(m)
        if board.termination == WIN:
            v = 1
        elif board.termination == DRAW:
            v = 0
        else:
            v = -value(board.cells)
        board.pop()
        if v == value(cells):
            out.append(m)
    return tuple(out)


def is_critical(cells: Tuple[int, ...]) -> bool:
    """A position where a MISTAKE is possible: some legal move is not
    value-preserving. (E.g. the empty board is NOT critical — every first
    move preserves tic-tac-toe's drawn value; the gate lives here.)"""
    return len(optimal_moves(cells)) < len(TicTacToe(cells).legal_moves)


def wdl_of_value(v: int) -> Tuple[float, float, float]:
    """Oracle value → WDL triple from the mover's view (diagnostic metric)."""
    return {1: (1.0, 0.0, 0.0), 0: (0.0, 1.0, 0.0), -1: (0.0, 0.0, 1.0)}[v]
