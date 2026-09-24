"""
chess_ai/net/features.py — feature tensor + policy move space (Phase 2.1).

Two jobs, both shared by every part of the pipeline (self-play, training,
eval):

1. THE MOVE SPACE (decision D1, settled here: **64×64 + promotions**)
   ---------------------------------------------------------------------------
   The policy heads emit one distribution over 4240 slots, always from the
   CURRENT MOVER's perspective (matching the input tokens):

     [0, 4096)    from-square × 64 + to-square, both in mover-relative
                  coordinates (a1..h8 as seen by the mover; the board is
                  vertically flipped when the mover is Black). This single
                  region covers ALL normal moves, captures, en passant,
                  castling (e1g1-style king two-stepper), and QUEEN
                  promotions (a pawn's arrival square implies "=Q").
     [4096, 4168) OUR underpromotions: 8 pawn squares (relative rank 7) ×
                  3 directions (capture-left +7, push +8, capture-right +9)
                  × 3 pieces (N, B, R).
     [4168, 4240) THE ENEMY's underpromotions, mirrored: relative rank 2 ×
                  3 downward directions (−7, −8, −9) × 3 pieces. These
                  slots exist for the aux opponent-reply head (KataGo
                  adoption): its target is the enemy's move expressed in
                  OUR frame, and an enemy pawn promoting "downward" in our
                  orientation is otherwise unencodable (its from-to pair
                  would collide across all four promo pieces). Our own
                  moves never use this region — our pawns promote upward
                  by construction.

   Why this over LC0's 4672-flat (PLAN D1): the transformer's bilinear
   policy head produces 64×64 = from×to attention scores *natively* — the
   from-to region needs no decode table — and promotions are 3.4% of the
   space instead of 12%. Illegal slots are masked to probability
   exactly zero before the softmax (see net.model).

2. A PYTHON REFERENCE ENCODER of the input tensor
   ---------------------------------------------------------------------------
   `reference_tokens()` re-implements the C encoder (engine/nn_encode.c)
   from scratch — different code, same spec (engine/include/chess_nn.h) —
   so the tests can compare the C output tensor-for-tensor. It deliberately
   uses only observable engine state (FENs + Board.is_repetition), never
   the C internals, so agreement is evidence and not a tautology.

   Layout (identical to chess_nn.h): 64 tokens × 117 dims, token-major;
   token = mover-relative square; dims
     [0, 96)   8-ply history × 12 piece one-hots (h=0 now, "ours" first)
     [96, 104) per-ply already-repeated flag
     [104, 112) ep target file one-hot (current position)
     [112, 116) castling [mover K, mover Q, enemy K, enemy Q]
     [116]     halfmove clock / 100
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from chess_ai.engine import Board

# ---------------------------------------------------------------------------
# Input-tensor constants (mirror of engine/include/chess_nn.h)
# ---------------------------------------------------------------------------

NN_TOKENS = 64
NN_TOKEN_DIM = 117
NN_HISTORY = 8

# Dims, per token.
DIM_PIECES = 0        # 8 × 12 one-hots
DIM_REPFLAG = 96      # 8 flags
DIM_EP = 104          # 8 file one-hots
DIM_CASTLE = 112      # 4 rights
DIM_HALFMOVE = 116    # 1 scalar

# ---------------------------------------------------------------------------
# Move space (D1)
# ---------------------------------------------------------------------------

POLICY_SIZE = 4240            # 4096 from-to + 72 own + 72 enemy promos
FT_SIZE = 4096
UNDERPROMO_SIZE = 144

_PROMO_LETTERS = {"n": 0, "b": 1, "r": 2}     # slot offset per promo piece
# Mover-relative directions from OUR rank-7 pawn (own promotion, upward),
# in slot order; and the mirrored downward set for the ENEMY's promoting
# pawn as seen in our frame (aux-reply target region).
_PROMO_DIRS = (7, 8, 9)                        # capture-left, push, capture-right
_PROMO_DIRS_DOWN = (-7, -8, -9)
_PROMO_FROM_BASE = 48                          # own: relative rank 7
_PROMO_FROM_BASE_DOWN = 8                      # enemy: relative rank 2


def rel_square(sq: int, side: int) -> int:
    """Absolute square (a1=0…h8=63) → mover-relative square.

    The mover always "sits at the bottom": for Black the board flips
    vertically (rank ^ 56). Same convention as the input tokens.
    """
    return sq if side == 0 else sq ^ 56


def abs_square(sq: int, side: int) -> int:
    """Mover-relative → absolute (self-inverse inverse of rel_square)."""
    return rel_square(sq, side)


def uci_to_index(uci: str, side: int) -> int:
    """UCI move string + side to move → policy slot.

    `side` is the frame: whose perspective the from/to squares are relative
    to. Normally that is the mover (own promotion → upward slots), but the
    aux-reply target calls this with the CURRENT mover's side for the
    ENEMY's move — then a promotion maps to the mirrored downward slots.
    Caller must pass a *legal* move: the mapping is pure arithmetic, it
    does not know the position.
    """
    files = "abcdefgh"
    from_abs = files.index(uci[0]) + (int(uci[1]) - 1) * 8
    to_abs = files.index(uci[2]) + (int(uci[3]) - 1) * 8
    frm, to = rel_square(from_abs, side), rel_square(to_abs, side)
    if len(uci) == 5:
        promo = uci[4]
        if promo == "q":
            pass                     # queen promo lives in the from-to region
        elif promo in _PROMO_LETTERS:
            direction = to - frm     # ±7 / ±8 / ±9 by construction
            if direction in _PROMO_DIRS:                       # own promo
                return (FT_SIZE
                        + (frm - _PROMO_FROM_BASE) * 9
                        + _PROMO_DIRS.index(direction) * 3
                        + _PROMO_LETTERS[promo])
            if direction in _PROMO_DIRS_DOWN:                  # enemy promo
                return (FT_SIZE + 72
                        + (frm - _PROMO_FROM_BASE_DOWN) * 9
                        + _PROMO_DIRS_DOWN.index(direction) * 3
                        + _PROMO_LETTERS[promo])
            raise ValueError(f"promotion with impossible geometry: {uci!r}")
        else:
            raise ValueError(f"bad promotion letter in {uci!r}")
    return frm * 64 + to


def index_to_uci(idx: int, side: int) -> str:
    """Policy slot → UCI string (inverse of uci_to_index).

    Slots are stored mover-relative, so `side` is needed to flip back to
    absolute coordinates. For from-to slots the move's UCI is fully
    determined ONLY up to promotion: a pawn moving to the last rank
    promotes, and the from-to region means queen. (Whether the from piece
    is a pawn at all is a property of the position, not of the slot.)
    """
    files = "abcdefgh"

    def uci_sq(rel_sq: int) -> str:
        sq = abs_square(rel_sq, side)          # mover-relative → absolute
        return files[sq % 8] + str(sq // 8 + 1)

    if idx < FT_SIZE:
        frm, to = idx // 64, idx % 64
        uci = uci_sq(frm) + uci_sq(to)
        # A from-to pair crossing into the last rank is promotion-implied
        # ("=Q") in either direction: our pawn promotes upward (rel rank
        # 7 → 8), the enemy's downward in our frame (rel rank 2 → 1) —
        # that's the aux-target case. Non-pawn moves on those squares
        # share the slot cosmetically; within a position they can never
        # collide with a pawn's move (one square holds one piece).
        up_promo = 48 <= frm <= 55 and 56 <= to <= 63
        down_promo = 8 <= frm <= 15 and 0 <= to <= 7
        if up_promo or down_promo:
            uci += "q"
        return uci
    if idx < FT_SIZE + 72:                        # own underpromotion
        k = idx - FT_SIZE
        frm = _PROMO_FROM_BASE + k // 9
        direction = _PROMO_DIRS[(k // 3) % 3]
    else:                                         # enemy underpromotion
        k = idx - FT_SIZE - 72
        frm = _PROMO_FROM_BASE_DOWN + k // 9
        direction = _PROMO_DIRS_DOWN[(k // 3) % 3]
    promo = "nbr"[k % 3]
    to = frm + direction
    return uci_sq(frm) + uci_sq(to) + promo


def legal_move_indices(board: Board) -> List[int]:
    """Policy slots of every legal move in `board`'s position."""
    side = board.side_to_move
    return [uci_to_index(u, side) for u in board.legal_moves]


def policy_mask(indices: Sequence[int]) -> np.ndarray:
    """Boolean [POLICY_SIZE] mask: True = legal. Everything else will be
    masked to probability exactly zero before the softmax."""
    m = np.zeros(POLICY_SIZE, dtype=bool)
    for i in indices:
        m[i] = True
    return m


# ---------------------------------------------------------------------------
# Value targets (WDL)
# ---------------------------------------------------------------------------

def wdl_onehot(outcome_white_pov: float, side: int) -> np.ndarray:
    """Game outcome as a 3-way Win/Draw/Loss one-hot from `side`'s view.

    outcome_white_pov: +1 white win, -1 black win, 0 draw. Order: [W, D, L].
    """
    z = outcome_white_pov if side == 0 else -outcome_white_pov
    v = np.zeros(3, dtype=np.float32)
    v[0 if z > 0 else (1 if z == 0 else 2)] = 1.0
    return v


# ---------------------------------------------------------------------------
# Reference encoder (test oracle for the C encoder)
# ---------------------------------------------------------------------------

def _parse_fen(fen: str):
    """Minimal FEN reader for the reference encoder: returns
    (placement[64] as piece chars, side, castling str, ep file or None,
    halfmove clock)."""
    placement, side, castling, ep, halfmove, _fullmove = fen.split()
    board: List[Optional[str]] = [None] * 64
    sq = 56  # FEN starts at a8
    for ch in placement:
        if ch == "/":
            sq -= 16
        elif ch.isdigit():
            sq += int(ch)
        else:
            board[sq] = ch
            sq += 1
    ep_file = None if ep == "-" else "abcdefgh".index(ep[0])
    return board, (0 if side == "w" else 1), castling, ep_file, int(halfmove)


def reference_tokens(fen_history: Sequence[str]) -> np.ndarray:
    """Encode from a list of FENs — the Python reference implementation.

    `fen_history` = positions of the game, OLDEST FIRST, current position
    LAST, up to any length (only the last 8 are used). Using FENs (rather
    than live board objects) forces this implementation to work from
    serialized observable state, the same way an independent implementation
    would — see the module docstring on why that matters.

    Repetition flags are NOT recoverable from FENs alone (they need the
    game's position history), so the caller passes them alongside:
    see `reference_flags()`; tests pass both.
    """
    window = list(fen_history[-NN_HISTORY:])   # oldest … current
    t = np.zeros((NN_TOKENS, NN_TOKEN_DIM), dtype=np.float32)

    placement, side, castling, ep_file, halfmove = _parse_fen(window[-1])
    mover = side

    for h, fen in enumerate(reversed(window)):          # h = 0 current …
        plc = placement if h == 0 else _parse_fen(fen)[0]
        for sq, piece in enumerate(plc):
            if piece is None:
                continue
            token = rel_square(sq, mover)
            own = (piece.isupper() and mover == 0) or (piece.islower() and mover == 1)
            ptype = "pnbrqk".index(piece.lower())
            t[token, DIM_PIECES + h * 12 + (ptype if own else 6 + ptype)] = 1.0

    if ep_file is not None:
        t[:, DIM_EP + ep_file] = 1.0

    rights = {"K": 0, "Q": 1, "k": 2, "q": 3}
    have = set(castling) - {"-"}
    order = [0, 1, 2, 3] if mover == 0 else [2, 3, 0, 1]   # mover's first
    for slot, right in enumerate(order):
        t[:, DIM_CASTLE + slot] = 1.0 if "KQkq"[right] in have else 0.0

    t[:, DIM_HALFMOVE] = halfmove / 100.0
    return t


def reference_flags(start_fen: str, moves: Sequence[str]) -> List[bool]:
    """Per-ply repetition flags for the position after `moves`, h = 0 … 7.

    Uses only Board.is_repetition(2) ("this position occurred before"),
    the public engine predicate — a genuinely different computation from
    the C encoder's raw Zobrist-key scan. Computed by replaying the game
    prefix in scratch Boards; O(ply²), test-only.

    Returns flags for the position reached after ALL of `moves`; call with
    moves[:t] to get the flags "as of ply t".
    """
    flags = []
    for h in range(NN_HISTORY):
        target_ply = len(moves) - h
        if target_ply < 0:
            flags.append(False)
            continue
        scratch = Board(start_fen)
        for m in moves[:target_ply]:
            scratch.push(m)
        flags.append(scratch.is_repetition(2))
    return flags


def apply_reference_flags(tokens: np.ndarray, flags: Sequence[bool]) -> np.ndarray:
    """Write `flags` into the repetition dims of a reference tensor."""
    for h, f in enumerate(flags):
        tokens[:, DIM_REPFLAG + h] = 1.0 if f else 0.0
    return tokens
