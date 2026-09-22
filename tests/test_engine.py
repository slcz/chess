"""
tests/test_engine.py — pytest entry point for the C rules engine (Phase 1).

Layout of the verification stack (PLAN §10):

  1. test_c_unit_gate      — runs engine's own C unit-test binary
                             (208k checks: rule edge cases, make/unmake
                             integrity, FEN round-trips, termination,
                             random games)
  2. test_perft_cli_gate   — runs the perft CLI over every line of
                             engine/tests/perft_suite.txt (canonical positions
                             at depth 5-6 + targeted edge cases)
  3. test_oracle_fuzz (tests/test_ffi.py: perft + termination parity FENs)
                           — random game walks cross-checked against
                             python-chess THROUGH THE FFI BINDING: at every
                             position, the engine's legal-move set, FEN
                             serialization, AND termination call (reason +
                             winner, python-chess outcome(claim_draw=True)
                             granularity) must match the oracle exactly.

python-chess is a *test-only* oracle (PLAN §10.3): it never touches training
data, so the zero-knowledge constraint is intact.

Usage:
    .venv/bin/python -m pytest tests/ -v          # everything (~1 min)
    FUZZ_GAMES=400 .venv/bin/python -m pytest tests/ -v -k fuzz   # long fuzz

Environment knobs: FUZZ_GAMES (default 40), FUZZ_MAX_PLIES (default 300).
"""
import os
import random
import subprocess
from pathlib import Path

import pytest

from chess_ai.engine import Board, Termination, CHESS_PY_NAMES

REPO = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO / "engine"
PERFT_BIN = ENGINE_DIR / "build" / "perft"
TEST_UNITS_BIN = ENGINE_DIR / "build" / "test_units"

# Distinct start positions for the fuzz walks (canonical suite + spice).
FUZZ_STARTS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "8/P1k5/K7/8/8/8/8/8 w - - 0 1",
    "8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1",   # ep-available start
]


@pytest.fixture(scope="session", autouse=True)
def build_engine():
    """Build the C engine once per pytest session (no-op when up to date)."""
    subprocess.run(["make", "all"], cwd=ENGINE_DIR, check=True,
                   capture_output=True, text=True)
    yield


def test_c_unit_gate():
    """The C unit-test binary must exit 0 (it prints failures on stdout)."""
    r = subprocess.run([TEST_UNITS_BIN], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert " 0 failed" in r.stdout


def test_perft_cli_gate():
    """The perft CLI must verify every line of the suite file exactly."""
    r = subprocess.run([PERFT_BIN, "suite"], capture_output=True, text=True,
                       timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL OK" in r.stdout


def _oracle_outcome(board):
    """(termination-name, winner) from python-chess, in our color encoding
    (white=0/black=1; python-chess uses WHITE=True/BLACK=False — flip)."""
    out = board.outcome(claim_draw=True)
    if out is None:
        return None, None
    return (out.termination.name,
            None if out.winner is None else int(not out.winner))


def test_oracle_fuzz():
    """Random game walks through the FFI: legal-move sets, FEN serialization
    and full termination calls (reason + winner) must agree with python-chess
    at every single position (PLAN 1.5)."""
    chess = pytest.importorskip("chess")  # skip cleanly if not installed

    games = int(os.environ.get("FUZZ_GAMES", "40"))
    max_plies = int(os.environ.get("FUZZ_MAX_PLIES", "300"))
    rng = random.Random(20260922)  # fixed seed: failures reproduce exactly

    positions = 0
    term_seen = {}  # Termination -> count (printed for visibility)
    for g in range(games):
        board = chess.Board(rng.choice(FUZZ_STARTS))
        mine = Board()  # startpos handle; retargeted per game below
        mine.set_fen(board.fen(en_passant="fen"))

        for _ in range(max_plies):
            # en_passant="fen": report the ep square whenever it is set,
            # which is exactly our engine's convention (verified: our FEN
            # output must be byte-identical to the oracle's).
            fen = board.fen(en_passant="fen")

            assert mine.fen == fen, (
                f"FEN round-trip mismatch (game {g}):\n"
                f"  in:      {fen}\n  engine:  {mine.fen}")
            my_moves = sorted(mine.legal_moves)
            oracle_moves = sorted(m.uci() for m in board.legal_moves)
            assert my_moves == oracle_moves, (
                f"legal-move mismatch (game {g}) at {fen}:\n"
                f"  engine-only: {sorted(set(my_moves) - set(oracle_moves))}\n"
                f"  oracle-only: {sorted(set(oracle_moves) - set(my_moves))}")

            term = mine.termination
            name, winner = _oracle_outcome(board)
            assert (CHESS_PY_NAMES.get(term), mine.winner) == (name, winner), (
                f"termination mismatch (game {g}) at {fen}:\n"
                f"  engine:  {term.name}, winner={mine.winner}\n"
                f"  oracle:  {name}, winner={winner}")
            positions += 1
            if term != Termination.PLAYING:
                term_seen[term.name] = term_seen.get(term.name, 0) + 1
                break

            board.push(rng.choice(list(board.legal_moves)))
            mine.push(board.peek().uci())

    coverage = ", ".join(f"{k}×{v}" for k, v in sorted(term_seen.items()))
    print(f"\noracle fuzz: {games} games, {positions} positions agree with "
          f"python-chess (terminations hit: {coverage or 'none'})")
    # Random walks must actually exercise the draw machinery; if they ever
    # stop doing so (RNG change, early break bug), demand attention rather
    # than silently passing.
    assert any(t in term_seen for t in (
        "THREEFOLD_REPETITION", "FIFTY_MOVES", "STALEMATE",
        "INSUFFICIENT_MATERIAL", "CHECKMATE")), term_seen
