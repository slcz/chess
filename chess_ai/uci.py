"""
chess_ai/uci.py — a minimal hand-rolled UCI client.

Why hand-rolled: python-chess is contractually a *test-only oracle* in this
project (PLAN §10.3 — zero-knowledge), so the play path must not depend on
it, not even for engine plumbing. The UCI subset needed to play a game is
tiny: handshake, `position fen ...`, `go movetime N`, read `bestmove`.

UCI engines are also the calibration opponents later (node-limited
Stockfish, PLAN §1), so this client is the seed of the eval harness, not
just a demo toy. It speaks the protocol as a *stateless per-move* dialog:
we always send the full FEN (no `moves` list accumulation), which is
slightly more traffic but immune to drift between our board and the
engine's.
"""
from __future__ import annotations

import subprocess


class UCIEngine:
    """One spawned UCI engine process. Not thread-safe (one pipe pair)."""

    def __init__(self, path: str, movetime_ms: int = 100):
        self.movetime_ms = movetime_ms
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Handshake: identify, then confirm the engine is responsive. Some
        # engines only finish table setup on the first isready.
        self._send("uci")
        if not self._wait_for("uciok", timeout=10.0):
            raise RuntimeError(f"{path}: no uciok — not a UCI engine?")
        self._send("isready")
        if not self._wait_for("readyok", timeout=10.0):
            raise RuntimeError(f"{path}: no readyok")

    # --- protocol plumbing ------------------------------------------------

    def _send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, token: str, timeout: float) -> bool:
        # Line-by-line read with a timeout via a poll on the raw fd would be
        # the fully general answer; engines answer the handshake instantly,
        # so a simple blocking readline suffices here.
        while True:
            line = self.proc.stdout.readline()
            if not line:  # EOF: engine died
                return False
            if line.split()[:1] == [token]:
                return True

    # --- playing -----------------------------------------------------------

    def bestmove(self, fen: str) -> str:
        """Ask for the best move in `fen` (full position each call)."""
        self._send(f"position fen {fen}")
        self._send(f"go movetime {self.movetime_ms}")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine terminated during search")
            parts = line.split()
            if parts[:1] == ["bestmove"]:
                if len(parts) < 2 or parts[1] in ("(none)", "0000"):
                    raise RuntimeError(f"engine has no move in {fen!r}")
                return parts[1]
            # "info depth ..." lines are ignored.

    def quit(self) -> None:
        try:
            self._send("quit")
            self.proc.wait(timeout=5.0)
        except Exception:
            self.proc.kill()

    def __enter__(self) -> "UCIEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.quit()


class UCIStrategy:
    """Adapter: use an external UCI engine as a Strategy in chess_ai.match.

    movetime is the per-move budget; keep it tiny when playing the dummies,
    or the engine eats them before anything interesting happens.
    """

    def __init__(self, path: str, movetime_ms: int = 100):
        self.engine = UCIEngine(path, movetime_ms)

    def choose_move(self, board) -> str:
        move = self.engine.bestmove(board.fen)
        if move not in board.legal_moves:
            raise RuntimeError(f"engine returned illegal move {move!r}")
        return move

    def close(self) -> None:
        self.engine.quit()
