"""
chess_ai/uci.py — a minimal hand-rolled UCI client.

Why hand-rolled: python-chess is contractually a *test-only oracle* in this
project (PLAN §10.3 — zero-knowledge), so the play path must not depend on
it, not even for engine plumbing. The UCI subset needed to play a game is
tiny: handshake, `position fen ...`, `go movetime N`, read `bestmove`.

UCI engines are also the calibration opponents later (node-limited
Stockfish, PLAN §1), so this client is the seed of the eval harness, not
just a demo toy. Two dialog styles: `bestmove(fen)` is the stateless
full-FEN form (external engines — immune to drift between our board and
the engine's); `ask(line)` sends an arbitrary position line, which
PuctNetStrategy uses for incremental "root + full move list" dialogs —
uci_puct detects the extension, plays only the suffix, and keeps its
search tree across moves. A stale line (e.g. after an undo) is not a
prefix and the engine rebuilds, so drift still can't persist.
"""
from __future__ import annotations

import os
import subprocess


class UCIEngine:
    """One spawned UCI engine process. Not thread-safe (one pipe pair).

    go_nodes: when set, `bestmove` asks `go nodes N` (a fixed simulation
    budget) instead of `go movetime`. env: extra environment for the
    child (e.g. PUCT_CKPT for uci_puct). handshake_timeout: engines that
    load a network before answering `uci` need more than 10 s."""

    def __init__(self, path: str, movetime_ms: int = 100, env=None,
                 go_nodes: int | None = None, handshake_timeout: float = 10.0):
        self.movetime_ms = movetime_ms
        self.go_nodes = go_nodes
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=env,
        )
        # Handshake: identify, then confirm the engine is responsive. Some
        # engines only finish table setup on the first isready — and
        # uci_puct loads its network before it answers at all.
        self._send("uci")
        if not self._wait_for("uciok", timeout=handshake_timeout):
            raise RuntimeError(f"{path}: no uciok — not a UCI engine?")
        self._send("isready")
        if not self._wait_for("readyok", timeout=handshake_timeout):
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

    def ask(self, position_line: str) -> str:
        """Best move after an arbitrary position line — used for the
        incremental "position startpos/fen ... moves ..." dialogs that let
        a tree-keeping engine (uci_puct) inherit the previous search."""
        self._send(position_line)
        if self.go_nodes is not None:
            self._send(f"go nodes {self.go_nodes}")
        else:
            self._send(f"go movetime {self.movetime_ms}")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine terminated during search")
            parts = line.split()
            if parts[:1] == ["bestmove"]:
                if len(parts) < 2 or parts[1] in ("(none)", "0000"):
                    raise RuntimeError(
                        f"engine has no move after {position_line!r}")
                return parts[1]
            # "info depth ..." lines are ignored.

    def bestmove(self, fen: str) -> str:
        """Ask for the best move in `fen` (full position each call)."""
        return self.ask(f"position fen {fen}")

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


class PuctNetStrategy:
    """The PUCT search engine (engine/build/uci_puct) as a GUI/strategy
    opponent: one process per side, each with its OWN checkpoint
    (PUCT_CKPT env) and its own simulation budget (`go nodes N`) — so
    white and black can be two different nets at different strengths.

    Dialogs are incremental: every ask sends the game root
    (startpos/book FEN) plus the full move list, so the engine plays only
    the new suffix into its tree and each think inherits the previous
    think's subtree — the same retention the persistent-tree drivers
    (uci_mux/gate_match) get. Undos/rewinds arrive as a non-prefix line
    and the engine falls back to a full rebuild, so nothing can go stale.
    """

    def __init__(self, ckpt: str, nodes: int, driver: str | None = None,
                 device: str = "mps"):
        if driver is None:
            from pathlib import Path
            driver = str(Path(__file__).resolve().parents[1]
                         / "engine/build/uci_puct")
        env = dict(os.environ, PUCT_CKPT=ckpt, PUCT_DEVICE=device)
        self.ckpt = ckpt
        self.engine = UCIEngine(driver, env=env, go_nodes=nodes,
                                handshake_timeout=120.0)

    def choose_move(self, board) -> str:
        root = board.root_fen
        head = f"position fen {root}" if root else "position startpos"
        moves = board.move_stack_uci
        line = head + (f" moves {' '.join(moves)}" if moves else "")
        move = self.engine.ask(line)
        if move not in board.legal_moves:
            raise RuntimeError(f"engine returned illegal move {move!r}")
        return move

    def close(self) -> None:
        self.engine.quit()
