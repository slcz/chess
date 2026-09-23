"""
chess_ai/san.py — UCI → SAN (Standard Algebraic Notation) conversion.

SAN is the human notation ("Nf3", "exd5", "O-O", "Qh5#"); the engine speaks
UCI ("g1f3") everywhere internally. SAN exists only so human-facing output
(the GUI move list, future PGN recording) reads like a chess book.

Built on the FFI Board alone (python-chess stays test-only): piece_at gives
piece letters and capture detection, legal_moves gives disambiguation, and
push/pop gives the check/mate suffix. Display utility — nothing downstream
consumes it programmatically.
"""
from __future__ import annotations

from chess_ai.engine import Board

# Index = engine piece type (PAWN=0 … KING=5). Pawns have no letter; the
# king is never ambiguous so it needs no disambiguation logic.
_PIECE_LETTER = ["", "N", "B", "R", "Q", "K"]
_EMPTY = 12


def _sq(name: str) -> int:
    """UCI square name ("e4") → engine square index (a1=0 … h8=63)."""
    return (ord(name[1]) - ord("1")) * 8 + (ord(name[0]) - ord("a"))


def uci_to_san(board: Board, uci: str) -> str:
    """SAN for playing `uci` in `board`'s current position.

    The board is left exactly as found (the probe move is popped).
    """
    if uci not in board.legal_moves:
        raise ValueError(f"{uci!r} is not legal in {board.fen}")

    frm, to = uci[0:2], uci[2:4]
    promo = uci[4:].upper()  # "" or "Q"/"R"/"B"/"N"
    pc = board.piece_at(_sq(frm))
    ptype = pc % 6

    # A pawn moving diagonally onto an empty square is an en-passant capture.
    is_capture = board.piece_at(_sq(to)) != _EMPTY or (ptype == 0 and frm[0] != to[0])

    if ptype == 5 and abs(ord(to[0]) - ord(frm[0])) == 2:
        # King moves two files: castling (our UCI encodes it as e1g1/e1c1).
        core = "O-O" if to[0] > frm[0] else "O-O-O"
    elif ptype == 0:
        core = (frm[0] + "x" if is_capture else "") + to
        if promo:
            core += "=" + promo
    else:
        # Disambiguation: another identical piece that could also legally
        # reach `to`. SAN picks the minimal distinguishing part — file,
        # else rank, else both (e.g. R1a3 vs Rdf8 vs Qed4).
        others = {m[0:2] for m in board.legal_moves
                  if m[2:4] == to and m[0:2] != frm
                  and board.piece_at(_sq(m[0:2])) == pc}
        if not others:
            dis = ""
        elif frm[0] not in {s[0] for s in others}:
            dis = frm[0]
        elif frm[1] not in {s[1] for s in others}:
            dis = frm[1]
        else:
            dis = frm
        core = _PIECE_LETTER[ptype] + dis + ("x" if is_capture else "") + to

    # Check/mate suffix: look at the position AFTER the move.
    from chess_ai.engine import Termination
    board.push(uci)
    try:
        if board.termination == Termination.CHECKMATE:
            core += "#"
        elif board.is_check:
            core += "+"
    finally:
        board.pop()
    return core
