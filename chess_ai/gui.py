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
import random
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chess_ai.engine import Board, Termination
from chess_ai.san import uci_to_san
from chess_ai.strategies import GreedyMaterialStrategy, RandomStrategy

_PAGE = Path(__file__).resolve().parent / "static" / "board.html"
CONTROLLERS = ("human", "greedy", "random", "engine", "value", "policy",
               "net")


def _make_controller(name: str, seed: int | None, movetime_ms: int,
                     side: str | None = None, net_cfg: dict | None = None):
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
        import os
        path = os.environ.get("VALUE_CKPT")
        if not path:
            raise RuntimeError("value controller requested but VALUE_CKPT "
                               "is not set to a checkpoint path")
        from chess_ai.supervised.play import ValueStrategy
        return ValueStrategy(path)
    if name == "policy":
        # The supervised net's POLICY head: imitation play — argmax of the
        # trained policy head over legal moves (same VALUE_CKPT checkpoint).
        import os
        path = os.environ.get("VALUE_CKPT")
        if not path:
            raise RuntimeError("policy controller requested but VALUE_CKPT "
                               "is not set to a checkpoint path")
        from chess_ai.supervised.play import PolicyStrategy
        return PolicyStrategy(path)
    if name == "engine":
        import os
        path = os.environ.get("ENGINE_PATH") or shutil.which("stockfish")
        if not path:
            raise RuntimeError("engine requested but none found: set ENGINE_PATH "
                               "or install stockfish")
        from chess_ai.uci import UCIStrategy
        return UCIStrategy(path, movetime_ms=movetime_ms)
    if name == "net":
        # The PUCT search engine (engine/build/uci_puct), one process per
        # side with its own checkpoint + sims budget — two sides can be
        # two different nets. Config is per side and bound at server
        # start (play_gui.py --{white,black}-net-ckpt / -net-nodes).
        from chess_ai.uci import PuctNetStrategy
        cfg = net_cfg or {}
        ckpt = cfg.get(f"ckpt_{side}")
        if not ckpt:
            flag = "white" if side == "w" else "black"
            raise RuntimeError(f"net controller on {side!r} needs the "
                               f"{flag}-side checkpoint "
                               f"(--{flag}-net-ckpt)")
        return PuctNetStrategy(ckpt, int(cfg.get(f"nodes_{side}") or 160),
                               cfg.get("driver"), cfg.get("device", "mps"))
    raise RuntimeError(f"unknown controller {name!r}")


class Game:
    """One game + its controllers. Mutated only under `lock`."""

    def __init__(self, white: str = "human", black: str = "greedy",
                 seed: int | None = None, movetime_ms: int = 200,
                 net_cfg: dict | None = None, book_fens: list | None = None):
        self.lock = threading.Lock()
        self.movetime_ms = movetime_ms
        self.net_cfg = net_cfg
        self.book_fens = book_fens or []
        self._reset(white, black, seed)

    def _reset(self, white: str, black: str, seed: int | None) -> None:
        for c in (getattr(self, "white", None), getattr(self, "black", None)):
            if c is not None and c != "human" and hasattr(c, "close"):
                c.close()  # shut down any spawned UCI engine
        # With a book, each new game starts from a random book line (FEN
        # per line, e.g. the noob_3moves eval book) — the net controllers
        # are full-FEN dialogs, so a mid-game start needs nothing extra.
        self.board = (Board(random.choice(self.book_fens))
                      if self.book_fens else Board())
        rng_seed = seed if seed is not None else random.randrange(1 << 30)
        self.white = _make_controller(white, rng_seed, self.movetime_ms,
                                      side="w", net_cfg=self.net_cfg)
        self.black = _make_controller(black, rng_seed + 1, self.movetime_ms,
                                      side="b", net_cfg=self.net_cfg)
        self.white_name, self.black_name = white, black
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
            # watch mode: undo rewinds exactly one ply. Based on the GUI's
            # own history length (a book-start board's internal ply counter
            # begins above zero — history is the truth for what's undoable).
            "can_undo": len(self.history) > 1,
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
          open_browser: bool = True, net_cfg: dict | None = None,
          book: str | None = None) -> None:
    book_fens = None
    if book:
        p = Path(book if book.startswith("/") else
                 Path(__file__).resolve().parents[1] / book)
        book_fens = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        print(f"book: {len(book_fens)} lines from {p}")
    Handler.game = Game(white, black, seed=seed, movetime_ms=movetime_ms,
                        net_cfg=net_cfg, book_fens=book_fens)
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
