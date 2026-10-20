"""
tests/test_uci_dialog.py — incremental position dialogs (tree reuse).

The GUI's net controller talks to engine/build/uci_puct with "root + full
move list" lines instead of bare FENs, so the engine plays only the new
suffix into its tree and each think inherits the previous think's subtree
(the retention the mux drivers always had). Two gates:

  1. test_move_log        — the Board-side source of the dialog: root_fen
                            + move_stack_uci track construction/set_fen
                            origin, pushes and pops (no engine needed).
  2. test_dialog_protocol — the engine side: an extending line keeps the
                            tree ("tree kept"), a rewind or a new start
                            rebuilds it ("tree rebuilt"), and every
                            bestmove is legal in our own Board. Skipped
                            when the binary or a small checkpoint is not
                            built/present (runs/ artifacts are local).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from chess_ai.engine import Board  # noqa: E402

DRIVER = REPO / "engine" / "build" / "uci_puct"


def _small_ckpt() -> Path | None:
    """The smallest local checkpoint (loading it on cpu is quick)."""
    pick, size = None, 1 << 60
    for p in (REPO / "runs").glob("*.pt"):
        if p.stat().st_size < size:
            pick, size = p, p.stat().st_size
    return pick


def test_move_log():
    b = Board()
    assert b.root_fen is None and b.move_stack_uci == []
    for mv in ("e2e4", "e7e5", "g1f3"):
        b.push(mv)
    assert b.move_stack_uci == ["e2e4", "e7e5", "g1f3"]
    b.pop()
    assert b.move_stack_uci == ["e2e4", "e7e5"]

    # a FEN start lands mid-count (ply > 0): the log origin moves with it
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    f = Board(fen)
    assert f.root_fen == fen and f.move_stack_uci == []
    f.push("b1c3")
    assert f.move_stack_uci == ["b1c3"]

    # set_fen re-roots the log
    f.set_fen(fen)
    assert f.move_stack_uci == []


def test_dialog_protocol():
    ckpt = _small_ckpt()
    if not DRIVER.exists() or ckpt is None:
        import pytest
        pytest.skip("uci_puct binary or a checkpoint is not available")

    env = dict(os.environ, PUCT_CKPT=str(ckpt), PUCT_DEVICE="cpu",
               PUCT_SIMS="16", PUCT_SAMPLE_PLYS="0")
    p = subprocess.Popen([str(DRIVER)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, env=env, cwd=str(REPO))

    def send(line):
        p.stdin.write(line + "\n")
        p.stdin.flush()

    def bestmove():
        send("go nodes 16")
        while True:
            parts = p.stdout.readline().split()
            if parts[:1] == ["bestmove"]:
                return parts[1]

    board = Board()
    send("uci")
    send("isready")
    try:
        send("position startpos moves e2e4")
        board.push("e2e4")
        board.push(bestmove())                        # legality: push raises

        send(f"position startpos moves {' '.join(board.move_stack_uci)} g1f3")
        board.push("g1f3")
        board.push(bestmove())

        board.pop()                                   # rewind to the
        board.pop()                                   # earlier decision point
        send(f"position startpos moves {' '.join(board.move_stack_uci)}")
        board.push(bestmove())

        send("quit")
        p.wait(timeout=10)
    finally:
        if p.poll() is None:
            p.kill()
    log = p.stderr.read()

    # board legality was enforced incrementally above; now the telemetry
    lines = [l for l in log.splitlines() if l.startswith("puct:")]
    kinds = [l.split()[2] for l in lines]
    assert kinds == ["kept", "kept", "rebuilt"], lines
    assert "(+1 moves" in lines[0] and "(+1 moves" in lines[1]
    assert "(+2 moves" in lines[2]          # rewind replays e2e4 + reply1
