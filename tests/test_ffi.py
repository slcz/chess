"""
tests/test_ffi.py — pytest gate for the Python ↔ C boundary (Phase 0.3/1).

Layers:
  1. binding sanity      — library loads, basic Board ops round-trip
  2. perft via FFI       — exact node counts on a few canonical positions
                           (the Phase 1 gate explicitly includes "via FFI")
  3. termination parity  — canonical game-over FENs and played-out
                           sequences, each cross-checked against
                           python-chess's outcome(claim_draw=True):
                           same reason, same winner

The random-walk fuzz (legal moves + FEN + termination, everything vs the
python-chess oracle) lives in tests/test_engine.py::test_oracle_fuzz.
"""
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import pytest

from chess_ai.engine import Board, Termination
from chess_ai.engine import CHESS_PY_NAMES
from chess_ai.san import uci_to_san

REPO = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO / "engine"

chess = pytest.importorskip("chess")  # oracle for section 3


@pytest.fixture(scope="session", autouse=True)
def build_engine():
    subprocess.run(["make", "all"], cwd=ENGINE_DIR, check=True,
                   capture_output=True, text=True)
    yield


# ------------------------------------------------------------------ 1

def test_binding_basics():
    b = Board()
    assert b.fen == chess.STARTING_FEN
    assert len(b.legal_moves) == 20
    b.push("e2e4")
    assert b.fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    assert b.side_to_move == 1
    assert not b.is_check
    b.push("e7e5")
    b.pop()
    assert b.fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    b.pop()
    assert b.fen == chess.STARTING_FEN
    with pytest.raises(IndexError):
        b.pop()


def test_bad_fen_and_illegal_move():
    with pytest.raises(ValueError):
        Board("not a fen")
    b = Board()
    with pytest.raises(Exception):
        b.push("e2e5")  # pawns don't move three squares


# ------------------------------------------------------------------ 2

PERFT_CASES = [
    (chess.STARTING_FEN, 4, 197_281),
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4_085_603),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674_624),
]


@pytest.mark.parametrize("fen,depth,nodes", PERFT_CASES)
def test_perft_via_ffi(fen, depth, nodes):
    assert Board(fen).perft(depth) == nodes


# ------------------------------------------------------------------ 3

def outcome_of(board):
    """(reason-name, winner) per python-chess, our granularity.

    Mind the color encodings: python-chess uses WHITE=True / BLACK=False,
    our engine uses white=0 / black=1 — translate with `not`.
    """
    out = board.outcome(claim_draw=True)
    if out is None:
        return None, None
    return out.termination.name, (None if out.winner is None else int(not out.winner))


TERMINATION_FENS = [
    # (fen, expected Termination, expected winner) — every reason code
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", Termination.PLAYING, None),
    ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", Termination.CHECKMATE, 1),
    ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", Termination.STALEMATE, None),
    ("k7/8/8/8/8/8/8/B6K w - - 0 1", Termination.INSUFFICIENT_MATERIAL, None),
    ("k7/8/2b5/8/8/8/8/5B1K w - - 0 1", Termination.INSUFFICIENT_MATERIAL, None),
    ("8/8/8/8/8/5k2/8/5K1R w - - 100 60", Termination.FIFTY_MOVES, None),
    ("8/8/8/8/8/5k2/8/5K1R w - - 150 100", Termination.SEVENTYFIVE_MOVES, None),
]


@pytest.mark.parametrize("fen,term,winner", TERMINATION_FENS)
def test_termination_fens(fen, term, winner):
    b = Board(fen)
    assert b.termination == term
    assert b.winner == winner

    oracle = chess.Board(fen)
    name, owinner = outcome_of(oracle)
    assert CHESS_PY_NAMES.get(term) == name
    assert winner == (None if owinner is None else int(owinner))


def test_termination_played_sequences():
    """Terminations the FEN field can't express alone: build them by play."""
    # Threefold by knight shuffle: startpos occurs 3x after two rounds.
    b = Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        b.push(uci)
    assert b.termination == Termination.THREEFOLD_REPETITION

    # Fivefold: two more rounds -> 5x; still under 100 halfmoves, so the
    # automatic fivefold outranks everything else here.
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8") * 2:
        b.push(uci)
    assert b.termination == Termination.FIVEFOLD_REPETITION

    # And the oracle agrees on both, including the winner (draw = None).
    oracle = chess.Board()
    oracle_moves = ("g1f3", "g8f6", "f3g1", "f6g8") * 4
    for uci in oracle_moves:
        oracle.push(chess.Move.from_uci(uci))
    name, winner = outcome_of(oracle)
    assert name == CHESS_PY_NAMES[Termination.FIVEFOLD_REPETITION]
    assert winner is None


def test_ep_pinned_repetition_parity():
    """The subtle one: an ep square that is set but pinned does not make a
    position distinct — the identical placement without ep is a repeat.
    Naive raw-ep hashing gets this wrong; the oracle pins the behavior.
    The d2d4 push makes occurrence 1 (phantom ep); each 4-ply king shuffle
    back to the same placement adds one more occurrence."""
    shuffle = ["a4a5", "d1d2", "a5a4", "d2d1"]
    moves = ["d2d4"] + shuffle + shuffle  # 3 occurrences of the placement

    b = Board("8/8/8/8/k3p2Q/8/3P4/3K4 w - - 0 1")
    for uci in moves:
        b.push(uci)
    assert b.termination == Termination.THREEFOLD_REPETITION

    # Same shape with the pin removed: the ep capture is real, so the
    # post-push placement and the no-ep placements are NOT the same
    # position — only 2 occurrences of the no-ep key. (The overall game
    # may still be claimable via a *different* repeated position the king
    # shuffle creates — that's the claim lookahead doing its job — so pin
    # the ep semantics with the granular predicate, plus full outcome
    # parity with the oracle.)
    b = Board("8/8/8/8/k3p3/7Q/3P4/3K4 w - - 0 1")
    for uci in moves:
        b.push(uci)
    assert not b.is_repetition(3)

    for fen in ("8/8/8/8/k3p2Q/8/3P4/3K4 w - - 0 1",
                "8/8/8/8/k3p3/7Q/3P4/3K4 w - - 0 1"):
        mine = Board(fen)
        oracle = chess.Board(fen)
        for uci in moves:
            mine.push(uci)
            oracle.push(chess.Move.from_uci(uci))
        assert mine.is_repetition(3) == oracle.is_repetition(3)
        name, winner = outcome_of(oracle)
        assert (CHESS_PY_NAMES.get(mine.termination), mine.winner) == (name, winner)


# ------------------------------------------------------------------ 4

SAN_STARTS = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/P1k5/K7/8/8/8/8/8 w - - 0 1",          # promotion-heavy endgame
    "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",    # ep available
]


def test_san_parity():
    """Our hand-written UCI→SAN must equal python-chess's Board.san() on
    random walks — castling, captures, promotions, disambiguation (Rae1,
    Nbd2, Qh4e1…), and check/mate suffixes all get exercised."""
    rng = random.Random(20260922)
    checked = 0
    for g in range(25):
        oracle = chess.Board(rng.choice(SAN_STARTS))
        mine = Board(oracle.fen(en_passant="fen"))
        for _ in range(200):
            legal = list(oracle.legal_moves)
            if not legal or oracle.is_game_over(claim_draw=True):
                break
            mv = rng.choice(legal)
            expected = oracle.san(mv)
            got = uci_to_san(mine, mv.uci())
            assert got == expected, (
                f"SAN mismatch at {oracle.fen()}: move {mv.uci()}: "
                f"ours={got!r} oracle={expected!r}")
            oracle.push(mv)
            mine.push(mv.uci())
            checked += 1
    assert checked > 500, checked  # the walks must actually exercise the code
    print(f"\nSAN parity: {checked} moves agree with python-chess")
