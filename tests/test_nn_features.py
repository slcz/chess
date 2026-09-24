"""
tests/test_nn_features.py — cross-checks of the Phase 2 feature layer.

Three layers of evidence, each independent of the others:

1. ENCODER PARITY — the C encoder (engine/src/nn_encode.c, reached through
   the FFI) vs the pure-Python reference (chess_ai/net/features.py) over
   positions of random games. The reference is written from the spec using
   only FENs and the public Board API, so agreement is real evidence, not
   a tautology. Covers everything the C unit tests cover plus the full
   random-walk sweep: history frames, flips, repetition flags, ep,
   castling, halfmove clock, short-game zero padding.

2. MOVE-SPACE ROUND-TRIP — every legal move of every sampled position maps
   to a distinct, in-range policy slot that decodes back to the same UCI
   string (both colors; promotions incl. underpromotions, castling, ep).

3. Small hand cases for the promotion edge of the 4240-slot space.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from chess_ai.engine import Board, Termination
from chess_ai.net.features import (
    FT_SIZE,
    POLICY_SIZE,
    apply_reference_flags,
    index_to_uci,
    legal_move_indices,
    reference_flags,
    reference_tokens,
    uci_to_index,
    wdl_onehot,
)
from chess_ai.train.data import START_FEN, generate_random_games

START = START_FEN


def random_walk_positions(n_games: int, seed: int, max_plies: int = 60):
    """Yield (board, moves_so_far, fen_history) at sampled plies of random
    games. fen_history is OLDEST-first per the reference encoder's API."""
    rng = random.Random(seed)
    for _ in range(n_games):
        board = Board(START)
        fens = [board.fen]
        moves = []
        while board.termination == Termination.PLAYING and board.ply < max_plies:
            move = rng.choice(board.legal_moves)
            board.push(move)
            moves.append(move)
            fens.append(board.fen)
        # sample ~8 plies per game, always including a few early ones
        picks = sorted(set([0, 1, 2, 3] + rng.sample(range(len(moves)), min(4, len(moves)))))
        for p in picks:
            replay = Board(START)
            for m in moves[:p]:
                replay.push(m)
            yield replay, moves[:p], fens[: p + 1]


# ---------------------------------------------------------------------------
# 1. encoder parity
# ---------------------------------------------------------------------------

def test_encoder_parity_vs_reference():
    checked = 0
    for board, moves, fens in random_walk_positions(n_games=4, seed=42):
        want = apply_reference_flags(
            reference_tokens(fens), reference_flags(START, moves)
        )
        got = board.encode()
        assert got.shape == want.shape
        assert np.allclose(got, want, atol=1e-6), (
            f"parity break at ply {board.ply} after {' '.join(moves[-3:])}"
        )
        checked += 1
    assert checked >= 20


def test_encoder_parity_deep_history():
    """Past ply 8 the window slides — check positions deep in a shuffle-y
    game where repetition flags are also live."""
    rng = random.Random(1234)
    board = Board(START)
    fens = [board.fen]
    moves = []
    for _ in range(50):
        if board.termination != Termination.PLAYING:
            break
        move = rng.choice(board.legal_moves)
        board.push(move)
        moves.append(move)
        fens.append(board.fen)
    for p in range(8, len(moves), 7):
        replay = Board(START)
        for m in moves[:p]:
            replay.push(m)
        want = apply_reference_flags(
            reference_tokens(fens[: p + 1]), reference_flags(START, moves[:p])
        )
        assert np.allclose(replay.encode(), want, atol=1e-6), f"ply {p}"


def test_encoder_fen_start_has_no_history():
    """A mid-game FEN load has no past: history frames must be zero."""
    board = Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    t = board.encode()
    assert t[:, :96].sum() == float((t[:, :12].sum()))      # only frame 0 set
    assert t[:, 96:104].sum() == 0.0                        # no repetition info


# ---------------------------------------------------------------------------
# 2. move-space round-trip
# ---------------------------------------------------------------------------

def test_move_space_round_trip():
    """Honest round-trip: slot→UCI recovers the move up to the documented
    from-to ambiguity — a from-to slot for a rel-rank-7→8 pair decodes with
    an appended "=q", which is correct for pawn moves and merely cosmetic
    for non-pawn moves on those squares (they are distinct slots from any
    pawn's move in the same position, which the injectivity check covers).
    Underpromotion slots are fully unambiguous."""
    for board, moves, _fens in random_walk_positions(n_games=6, seed=7, max_plies=120):
        side = board.side_to_move
        uci_moves = board.legal_moves
        idxs = [uci_to_index(u, side) for u in uci_moves]
        assert len(set(idxs)) == len(idxs), "slot collision among legal moves"
        assert all(0 <= i < POLICY_SIZE for i in idxs)
        for u, i in zip(uci_moves, idxs):
            decoded = index_to_uci(i, side)
            assert decoded in (u, u + "q"), (u, i, decoded)
            if len(u) == 5:
                # promotions are rare in uniform random play; when one
                # appears, it must decode exactly (hand-case tests pin the
                # full promo mapping for both colors).
                assert decoded == u
        # the same UCI maps differently by side (perspective flip)
        if uci_moves:
            assert uci_to_index(uci_moves[0], 0) != uci_to_index(uci_moves[0], 1)


def test_promotion_slots_hand_case():
    board = Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
    uci = board.legal_moves
    assert {"a7a8q", "a7a8r", "a7a8b", "a7a8n"} <= set(uci)   # kings move too
    iq = uci_to_index("a7a8q", 0)
    assert iq < FT_SIZE                      # queen promo in the from-to region
    for promo in "nbr":
        i = uci_to_index(f"a7a8{promo}", 0)
        assert FT_SIZE <= i < POLICY_SIZE    # underpromos in the extra region
        assert index_to_uci(i, 0) == f"a7a8{promo}"
    assert index_to_uci(iq, 0) == "a7a8q"

    # Black's promotion: pawn on a2 → a1, mover-relative rank 7 → 8.
    board = Board("8/7k/8/8/8/8/p5K1/8 b - - 0 1")
    assert {"a2a1q", "a2a1r", "a2a1b", "a2a1n"} <= set(board.legal_moves)
    iq = uci_to_index("a2a1q", 1)
    assert index_to_uci(iq, 1) == "a2a1q"
    assert index_to_uci(uci_to_index("a2a1n", 1), 1) == "a2a1n"


def test_castling_and_ep_in_from_to_region():
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    for u in ("e1g1", "e1c1"):
        assert uci_to_index(u, 0) < FT_SIZE
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    for u in ("e8g8", "e8c8"):
        assert uci_to_index(u, 1) < FT_SIZE
    board = Board("8/8/8/3pP3/8/8/8/k6K w - d6 0 1")
    assert uci_to_index("e5d6", 0) < FT_SIZE


def test_enemy_promotion_in_our_frame():
    """The aux-reply target encodes the ENEMY's promotion in OUR frame —
    downward geometry, mirrored slot region, distinct from our own region
    and from the from-to pair (which would collide across promo pieces)."""
    enemy = ["h2h1q", "h2h1r", "h2h1b", "h2h1n"]
    idxs = [uci_to_index(u, 0) for u in enemy]      # white's frame
    # Queen promo keeps the "=Q implied" from-to convention even for the
    # enemy; only its underpromotions need the mirrored region.
    assert idxs[0] < FT_SIZE
    assert all(FT_SIZE + 72 <= i < POLICY_SIZE for i in idxs[1:])
    assert len(set(idxs)) == 4                      # no promo-piece collision
    for u, i in zip(enemy, idxs):
        assert index_to_uci(i, 0) == u
    # own promos and enemy promos never share a slot
    own = [uci_to_index(f"a7a8{p}", 0) for p in "qrbn"]
    assert not set(idxs) & set(own)
    # symmetric from black's frame: white's a7a8n seen by black is downward
    i = uci_to_index("a7a8n", 1)
    assert FT_SIZE + 72 <= i < POLICY_SIZE
    assert index_to_uci(i, 1) == "a7a8n"


def test_wdl_onehot_perspective():
    assert list(wdl_onehot(+1.0, 0)) == [1.0, 0.0, 0.0]
    assert list(wdl_onehot(+1.0, 1)) == [0.0, 0.0, 1.0]     # black lost
    assert list(wdl_onehot(-1.0, 1)) == [1.0, 0.0, 0.0]     # black won
    assert list(wdl_onehot(0.0, 0)) == [0.0, 1.0, 0.0]


def test_legal_move_indices_matches_engine():
    for board, _moves, _fens in random_walk_positions(n_games=3, seed=9):
        assert len(legal_move_indices(board)) == len(board.legal_moves)
