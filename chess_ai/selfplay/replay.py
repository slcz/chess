"""
chess_ai/selfplay/replay.py — browse a self-play unit in the browser.

Server side of scripts/sp_replay.py: load one work unit (shard + DONE)
once, and serve two JSON endpoints plus the page. All chess truth is
server-side behind the FFI Board (same rule as gui.py): each game is
replayed once on load, producing per-ply FENs and SAN; the page only
renders what the server produced.

    GET  /games              the dataset index: index, plies, result,
                             ending, opening move per game
    GET  /game?i=<index>     one game's full replay payload: per-ply FEN,
                             SAN, played move, π (decoded to UCI, sorted),
                             mover-view root value, result/ending

Zero new dependencies: http.server + the existing FFI/san/reader.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from chess_ai.engine import Board
from chess_ai.net.features import index_to_uci
from chess_ai.san import uci_to_san
from chess_ai.selfplay.shard import SpGame, read_unit

_PAGE = Path(__file__).resolve().parent.parent / "static" / "replay.html"
PI_TOP = 12          # per-ply π entries shipped to the page (bars show 8)


def z_str(z: float) -> str:
    return "1-0" if z == 1.0 else "0-1" if z == 0.0 else "1/2-1/2"


def game_summary(g: SpGame) -> dict:
    """One row of the dataset index."""
    return {
        "idx": g.game_index,
        "plies": g.n_plies,
        "result": z_str(g.z_white),
        "end": "natural" if g.end_kind == 1 else "cap",
        "open": g.plies[0].played_uci if g.plies else "-",
    }


def game_payload(g: SpGame) -> dict:
    """Replay one game through the Board: per-ply FEN/SAN plus the shard's
    π (slots decoded to UCI in the mover's frame, sorted by prob) and the
    mover-view root value."""
    board = Board(g.start_fen)
    plies = []
    for p in g.plies:
        fen = board.fen
        san = uci_to_san(board, p.played_uci)
        pi = sorted(p.pi, key=lambda kv: -kv[1])[:PI_TOP]
        pi_ui = [[index_to_uci(s, p.mover), round(prob, 4)] for s, prob in pi]
        plies.append({
            "fen": fen,
            "san": san,
            "uci": p.played_uci,
            "mover": p.mover,
            "outlook": round(p.outlook, 4),
            "pi": pi_ui,
        })
        board.push(p.played_uci)
    return {
        "idx": g.game_index,
        "start_fen": g.start_fen,
        "n_sims": g.n_sims,
        "result": z_str(g.z_white),
        "end": "natural" if g.end_kind == 1 else "ply-cap",
        "plies": plies,
    }


def make_handler(unit_dir: str):
    games = read_unit(unit_dir)
    by_idx = {g.game_index: g for g in games}
    index = [game_summary(g) for g in sorted(games, key=lambda g: g.game_index)]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):      # quiet local dev tool
            pass

        def _json(self, obj):
            body = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/":
                page = _PAGE.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
            elif url.path == "/meta":
                self._json({"unit_dir": unit_dir, "games": len(games),
                            "positions": sum(g.n_plies for g in games)})
            elif url.path == "/games":
                self._json(index)
            elif url.path == "/game":
                i = int(parse_qs(url.query).get("i", ["0"])[0])
                if i not in by_idx:
                    self._json({"error": f"no game {i}"}, )
                    return
                self._json(game_payload(by_idx[i]))
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def serve(unit_dir: str, port: int = 8467) -> None:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(unit_dir))
    print(f"sp_replay: {unit_dir} at http://127.0.0.1:{port}")
    httpd.serve_forever()
