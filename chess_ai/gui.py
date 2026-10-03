"""
chess_ai/gui.py — a tiny local web server that puts a board in your browser.

Modes (pick per side in the page, or via CLI flags):
    human   — click pieces; legal destinations light up; promotion picker
    greedy  — GreedyMaterialStrategy (the dummy)
    random  — RandomStrategy
    engine  — any UCI engine binary (ENGINE_PATH or `stockfish` on PATH);
              test/calibration only, never touches training data
    value   — the supervised net's VALUE head: 1-ply greedy over legal
              moves by the checkpoint's WDL value (VALUE_CKPT env or
              play_gui.py --value-ckpt)
    policy  — the supervised net's POLICY head: imitation, argmax over
              legal moves (same VALUE_CKPT checkpoint)
    net     — the FULL self-play player: Gumbel MCTS (the Python
              reference, chess_ai/selfplay/gumbel.py) over the
              checkpoint's policy+value heads, n sims/move (NET_SIMS,
              default 100), first NET_SAMPLE_PLIES plies sampled ∝ π′
              (the ladder eval protocol), resignation off. Checkpoint
              per side: NET_CKPT_WHITE / NET_CKPT_BLACK (else NET_CKPT),
              or play_gui.py --white-ckpt/--black-ckpt. This is the
              playback mode: baseline seed vs the latest self-play net
              exactly as the ladder matches play them, one game at a
              time.

The browser is a dumb view/controller: it renders FENs, tracks clicks, and
POSTs moves. ALL chess truth stays server-side behind the FFI Board — the
page never decides legality, termination, or SAN on its own (it *does*
render historical positions client-side, but only from FENs the server
produced).

Zero new dependencies: http.server (stdlib) + one self-contained HTML
file (chess_ai/static/board.html — CSS board, Unicode pieces, no assets,
no network beyond localhost).

Usage:
    .venv/bin/python scripts/play_gui.py                 # human vs greedy
    .venv/bin/python scripts/play_gui.py --black random
    .venv/bin/python scripts/play_gui.py --white random --black greedy   # watch
    ENGINE_PATH=/path/to/engine .venv/bin/python scripts/play_gui.py --black engine

API (JSON):
    GET  /state                     full state incl. history + legal moves
    POST /move   {uci}               human move; machine replies inline
    POST /step   {}                  make one machine move (watch mode)
    POST /undo   {}                  rewind to the human's previous turn
    POST /reset  {white, black}      new game with given controllers
"""
from __future__ import annotations

import json
import os
import random
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chess_ai.engine import Board, Termination
from chess_ai.san import uci_to_san
from chess_ai.selfplay.gumbel import SelfPlayConfig, SelfPlayGame  # no torch here
from chess_ai.strategies import GreedyMaterialStrategy, RandomStrategy

_PAGE = Path(__file__).resolve().parent / "static" / "board.html"
CONTROLLERS = ("human", "greedy", "random", "engine", "value", "policy", "net")


class NetSearchStrategy:
    """The full self-play player as a Strategy: Gumbel MCTS at n sims/move,
    the chess_ai/selfplay/gumbel.py reference (verified against the C engine
    by the L2 suite — playback uses the simple sequential one; the C engine
    stays a bulk-matches tool).

    The net loads through selfplay.ladder.NetEvaluator — the SAME loader
    the ladder/policy_match evals use (checkpoint's saved net_config drives
    the architecture, legacy-shaw pin), so anything the ladder can play,
    the GUI can play.

    Protocol per move = the ladder eval protocol (R5): the first
    `sample_plies` plies of the game sample the played move ∝ π′ (varied
    openings), after that the deterministic sequential-halving survivor.
    Resignation is off (GUI games end by mate or a claimable draw only).

    The search runs on the GUI's LIVE Board (not a fresh FEN-only copy) so
    the 8-ply history ring the encoder reports is the real game history.
    Stateless per move: the tree is rebuilt from scratch, so undo/step-back
    in the GUI is always consistent. n=100 ≈ ~120 size-1 evals/move — about
    a second on the Mac's MPS."""

    def __init__(self, ckpt_path: str | None = None, n: int = 100,
                 sample_plies: int = 10, seed: int = 0,
                 device: str | None = None, evaluator=None):
        from chess_ai.selfplay.ladder import NetEvaluator
        from chess_ai.train.trainer import pick_device
        # evaluator= is the test seam (fixture evaluators, no torch);
        # production passes just the checkpoint path.
        self.evaluator = evaluator if evaluator is not None \
            else NetEvaluator(ckpt_path,
                              device=str(pick_device(device or "auto")))
        self.cfg = SelfPlayConfig(n=n, sample_plies=sample_plies,
                                  resign_value=-1.01)  # below −1: never fires
        self.seed = seed
        self.label = f"net({Path(ckpt_path).stem}@n{n})" if ckpt_path else "net"

    def choose_move(self, board: Board) -> str:
        start_ply = board.ply
        # game_index = ply → distinct but reproducible Gumbel breaks per move.
        game = _LiveBoardGame(self.cfg, self.seed, start_ply, board)
        try:
            decision, root = game._search_move(self.evaluator)
            if board.ply < self.cfg.sample_plies:
                slot = game._sample_pi(decision, root)
            else:
                slot = decision.survivor_slot
            return root.slot_to_uci[slot]
        finally:
            while board.ply > start_ply:   # the search rewinds itself; this
                board.pop()                # guards an exception mid-descent


class _LiveBoardGame(SelfPlayGame):
    """SelfPlayGame that searches the GUI's live Board instead of building
    its own from the FEN — board.encode() is the real 8-ply history ring."""

    def __init__(self, cfg, seed, game_index, board: Board):
        super().__init__(cfg, seed, game_index, start_fen=board.fen)
        self.board = board


def _make_controller(name: str, seed: int | None, movetime_ms: int,
                     side: str = "w"):
    if name == "human":
        return "human"
    if name == "greedy":
        return GreedyMaterialStrategy(seed=seed)
    if name == "random":
        return RandomStrategy(seed=seed)
    if name == "value":
        # The supervised net's VALUE head: 1-ply greedy over legal moves by
        # the checkpoint's WDL value (VALUE_CKPT picks the file — same
        # convention as ENGINE_PATH for the UCI option).
        path = os.environ.get("VALUE_CKPT")
        if not path:
            raise RuntimeError("value controller requested but VALUE_CKPT "
                               "is not set to a checkpoint path")
        from chess_ai.supervised.play import ValueStrategy
        return ValueStrategy(path)
    if name == "policy":
        # The supervised net's POLICY head: imitation play — argmax of the
        # trained policy head over legal moves (same VALUE_CKPT checkpoint).
        path = os.environ.get("VALUE_CKPT")
        if not path:
            raise RuntimeError("policy controller requested but VALUE_CKPT "
                               "is not set to a checkpoint path")
        from chess_ai.supervised.play import PolicyStrategy
        return PolicyStrategy(path)
    if name == "net":
        # The searched self-play player; checkpoint per side so a GUI game
        # can pit the baseline seed against the latest self-play net.
        env_name = {"w": "NET_CKPT_WHITE", "b": "NET_CKPT_BLACK"}[side]
        path = os.environ.get(env_name) or os.environ.get("NET_CKPT")
        if not path:
            raise RuntimeError("net controller requested but no checkpoint: "
                               f"set {env_name} (or NET_CKPT) — "
                               "play_gui.py --white-ckpt/--black-ckpt")
        sims = int(os.environ.get("NET_SIMS", "100"))
        sample_plies = int(os.environ.get("NET_SAMPLE_PLIES", "10"))
        return NetSearchStrategy(path, n=sims, sample_plies=sample_plies,
                                 seed=seed)
    if name == "engine":
        path = os.environ.get("ENGINE_PATH") or shutil.which("stockfish")
        if not path:
            raise RuntimeError("engine requested but none found: set ENGINE_PATH "
                               "or install stockfish")
        from chess_ai.uci import UCIStrategy
        return UCIStrategy(path, movetime_ms=movetime_ms)
    raise RuntimeError(f"unknown controller {name!r}")


class Game:
    """One game + its controllers. Mutated only under `lock`."""

    def __init__(self, white: str = "human", black: str = "greedy",
                 seed: int | None = None, movetime_ms: int = 200):
        self.lock = threading.Lock()
        self.movetime_ms = movetime_ms
        self._reset(white, black, seed)

    def _reset(self, white: str, black: str, seed: int | None) -> None:
        for c in (getattr(self, "white", None), getattr(self, "black", None)):
            if c is not None and c != "human" and hasattr(c, "close"):
                c.close()  # shut down any spawned UCI engine
        self.board = Board()
        rng_seed = seed if seed is not None else random.randrange(1 << 30)
        self.white = _make_controller(white, rng_seed, self.movetime_ms,
                                      side="w")
        self.black = _make_controller(black, rng_seed + 1, self.movetime_ms,
                                      side="b")
        # Display name: "net" shows WHICH checkpoint + sims it's playing
        # (e.g. net(policy_value_net_rope_25k@n100)); other kinds use the
        # plain controller name. /reset still validates on the kind.
        self.white_name = getattr(self.white, "label", white)
        self.black_name = getattr(self.black, "label", black)
        self.history = [{"ply": 0, "fen": self.board.fen, "uci": None, "san": None}]

    def controller(self, side: str):
        return self.white if side == "w" else self.black

    def controller_name(self, side: str) -> str:
        return self.white_name if side == "w" else self.black_name

    def _apply(self, uci: str) -> None:
        san = uci_to_san(self.board, uci)  # needs the pre-move position
        self.board.push(uci)
        self.history.append({"ply": self.board.ply, "fen": self.board.fen,
                             "uci": uci, "san": san})

    def machine_move_if_due(self) -> None:
        """If the side to move is a machine and the game is live, play it."""
        side = "w" if self.board.side_to_move == 0 else "b"
        c = self.controller(side)
        if c != "human" and self.board.termination == Termination.PLAYING:
            self._apply(c.choose_move(self.board))

    def state(self) -> dict:
        b = self.board
        side = "w" if b.side_to_move == 0 else "b"
        term = b.termination
        winner = None
        if term == Termination.CHECKMATE:
            winner = "b" if b.side_to_move == 0 else "w"
        return {
            "fen": b.fen,
            "side_to_move": side,
            "white": self.white_name,
            "black": self.black_name,
            "human_turn": self.controller_name(side) == "human"
                          and term == Termination.PLAYING,
            "legal": b.legal_moves,
            "last_move": self.history[-1]["uci"] if b.ply else None,
            "check": b.is_check,
            "termination": term.name,
            "winner": winner,
            "history": self.history,
            # Human games: undo returns to your previous decision point;
            # watch mode: undo rewinds exactly one ply.
            "can_undo": b.ply > 0,
        }


class Handler(BaseHTTPRequestHandler):
    game: Game = None  # set by serve()

    def log_message(self, *a):  # quiet; it's a local dev tool
        pass

    # --- plumbing -------------------------------------------------------

    def _json(self, obj, code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # --- routes ----------------------------------------------------------

    def do_GET(self):
        if self.path == "/":
            body = _PAGE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            with self.game.lock:
                self._json(self.game.state())
        elif self.path == "/favicon.ico":
            self.send_response(204)  # no content: nothing for the tab to show
            self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        g = self.game
        try:
            if self.path == "/move":
                uci = self._body().get("uci", "")
                with g.lock:
                    if g.controller_name("w" if g.board.side_to_move == 0 else "b") == "human":
                        if uci not in g.board.legal_moves:
                            return self._json({"error": f"illegal move {uci!r}"}, 400)
                        g._apply(uci)
                        g.machine_move_if_due()
                    else:
                        return self._json({"error": "not the human's turn"}, 400)
                    return self._json(g.state())

            if self.path == "/step":
                with g.lock:
                    g.machine_move_if_due()
                    return self._json(g.state())

            if self.path == "/undo":
                with g.lock:
                    if not g.history or g.board.ply == 0:
                        return self._json({"error": "nothing to undo"}, 400)
                    if g.white_name == "human" or g.black_name == "human":
                        # Rewind to the human's previous decision point: pop
                        # at least once, keep popping while it's a machine's
                        # move (i.e. skip the machine's reply too).
                        while True:
                            g.board.pop()
                            g.history.pop()
                            side = "w" if g.board.side_to_move == 0 else "b"
                            if g.controller_name(side) == "human" or g.board.ply == 0:
                                break
                    else:
                        # Watch mode (no humans): rewind exactly one ply.
                        g.board.pop()
                        g.history.pop()
                    return self._json(g.state())

            if self.path == "/reset":
                body = self._body()
                white, black = body.get("white", "human"), body.get("black", "greedy")
                if white not in CONTROLLERS or black not in CONTROLLERS:
                    return self._json({"error": "bad controller"}, 400)
                with g.lock:
                    g._reset(white, black, seed=None)
                    g.machine_move_if_due()  # engine may open as white
                    return self._json(g.state())

            return self._json({"error": "not found"}, 404)
        except Exception as e:  # surface engine/strategy failures as JSON
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def serve(white: str = "human", black: str = "greedy", port: int = 8466,
          seed: int | None = None, movetime_ms: int = 200,
          open_browser: bool = True) -> None:
    Handler.game = Game(white, black, seed=seed, movetime_ms=movetime_ms)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"chess GUI on {url}  (white={white}, black={black}) — Ctrl-C to stop")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
