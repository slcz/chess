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

Optional live-net layer (--ckpt): the checkpoint's policy/value heads
evaluate the position the browser is looking at —

    GET  /eval?i=<index>&p=<ply>   per legal move: net prior, net value
                             (child WDL flipped to the mover's view,
                             terminal children exact), merged with the
                             recorded π (visits) when this position is a
                             recorded search root

Zero new dependencies: http.server + the existing FFI/san/reader; torch
loads lazily, only when --ckpt is given.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from chess_ai.engine import Board, Termination
from chess_ai.net.features import index_to_uci, uci_to_index
from chess_ai.san import uci_to_san
from chess_ai.selfplay.shard import SpGame, read_unit

_PAGE = Path(__file__).resolve().parent.parent / "static" / "replay.html"
PI_TOP = 12          # per-ply π entries shipped to the page (bars show 8)


class Evaluator:
    """The checkpoint's policy + value heads over one position's legal
    moves — the supervised/play.py ValueStrategy machinery, exposed
    per-position for the replay overlay instead of distilled into a move
    choice. Mover's view everywhere; terminal children exact."""

    def __init__(self, ckpt_path: str, device: str | None = None):
        from chess_ai.train.trainer import Trainer, pick_device
        trainer = Trainer.load(ckpt_path)
        self.device = pick_device(device or "auto")
        self.model = trainer.model.to(self.device).eval()

    def eval_position(self, board: Board) -> list[dict]:
        import numpy as np
        import torch

        legal = board.legal_moves
        side = board.side_to_move
        slots = [uci_to_index(u, side) for u in legal]
        mask = np.zeros((1, 4240), dtype=bool)
        priors: dict[str, float] = {}
        if legal:
            for s in slots:
                mask[0, s] = True
            t = torch.from_numpy(board.encode()[None]).to(self.device)
            m = torch.from_numpy(mask).to(self.device)
            with torch.no_grad():
                out = self.model(t, m, m)
            logp = out["policy_logp"][0]
            # softmax over each legal move's OWN slot — aligned to the
            # legal-move order (the mask-boolean gather is ascending-slot
            # order, which scrambled priors onto the wrong moves)
            slot_t = torch.tensor(slots, dtype=torch.long,
                                  device=self.device)
            pz = torch.softmax(logp[slot_t], -1).cpu().numpy()
            priors = {u: float(p) for u, p in zip(legal, pz)}

        # children: net WDL at each resulting position (opponent's frame
        # — flipped to ours below), terminal children scored exactly
        tokens, exact, cand = [], [], []
        for uci in legal:
            board.push(uci)
            try:
                term = board.termination
                if term == Termination.CHECKMATE:
                    exact.append((1.0, 0.0, 0.0))       # we mated them
                elif term != Termination.PLAYING:
                    exact.append((0.0, 1.0, 0.0))       # stalemate / draw
                else:
                    exact.append(None)
                    tokens.append(board.encode())       # opponent to move
            finally:
                board.pop()
            cand.append(uci)

        wdls = iter(self._wdl(tokens)) if tokens else iter(())
        items = []
        for uci, e in zip(cand, exact):
            if e is not None:
                wdl = e
            else:
                wdl = tuple(reversed([float(x) for x in next(wdls)]))
            items.append({"uci": uci, "prior": priors.get(uci, 0.0),
                          "wdl": [round(x, 4) for x in wdl]})
        return items

    def _wdl(self, tokens) -> list[tuple[float, float, float]]:
        import numpy as np
        import torch
        out = []
        t = torch.from_numpy(np.stack(tokens)).to(self.device)
        with torch.no_grad():
            p = torch.softmax(self.model.value(self.model.trunk(t)), dim=-1)
        return [tuple(map(float, row)) for row in p.cpu().numpy()]


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
    term = board.termination.name.lower().replace("_", " ")   # e.g.
    # "THREEFOLD_REPETITION" -> "threefold repetition" — the WHY of a draw
    return {
        "idx": g.game_index,
        "start_fen": g.start_fen,
        "end_fen": board.fen,        # the terminal position — the board
                                     # after the last move (each plies[i]
                                     # .fen is the position BEFORE move i)
        "n_sims": g.n_sims,
        "result": z_str(g.z_white),
        "end": "natural" if g.end_kind == 1 else "ply-cap",
        "end_reason": term,
        "plies": plies,
    }


def make_handler(unit_dir: str, ckpt_path: str | None = None,
                 device: str | None = None):
    games = read_unit(unit_dir)
    by_idx = {g.game_index: g for g in games}
    index = [game_summary(g) for g in sorted(games, key=lambda g: g.game_index)]
    eval_lock = threading.Lock()
    evaluator = None       # lazy: torch loads on the first /eval request

    def get_evaluator():
        nonlocal evaluator
        with eval_lock:
            if evaluator is None:
                evaluator = Evaluator(ckpt_path, device)
            return evaluator

    def eval_payload(g: SpGame, p: int) -> dict:
        """Net priors + child values at the position before ply p, with
        the recorded π merged by policy slot (visits ≈ prob · n_sims)."""
        board = Board(g.start_fen)
        for i in range(p):
            board.push(g.plies[i].played_uci)
        side = board.side_to_move
        rec = {slot: prob for slot, prob in g.plies[p].pi} \
            if p < g.n_plies else {}
        items = get_evaluator().eval_position(board)
        for it in items:
            it["san"] = uci_to_san(board, it["uci"])
            slot = uci_to_index(it["uci"], side)
            if slot in rec:
                it["pi_prob"] = round(rec[slot], 4)
                it["pi_visits"] = round(rec[slot] * g.n_sims)
        return {"i": g.game_index, "p": p, "side": side, "sims": g.n_sims,
                "items": items}

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
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
            elif url.path == "/meta":
                self._json({"unit_dir": unit_dir, "games": len(games),
                            "positions": sum(g.n_plies for g in games),
                            "has_net": ckpt_path is not None,
                            "ckpt": ckpt_path or ""})
            elif url.path == "/games":
                self._json(index)
            elif url.path == "/game":
                i = int(parse_qs(url.query).get("i", ["0"])[0])
                if i not in by_idx:
                    self._json({"error": f"no game {i}"}, )
                    return
                self._json(game_payload(by_idx[i]))
            elif url.path == "/eval":
                if ckpt_path is None:
                    self._json({"error": "no --ckpt given"})
                    return
                q = parse_qs(url.query)
                i, p = int(q.get("i", ["0"])[0]), int(q.get("p", ["0"])[0])
                if i not in by_idx:
                    self._json({"error": f"no game {i}"})
                    return
                self._json(eval_payload(by_idx[i], max(0, p)))
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def serve(unit_dir: str, port: int = 8467,
          ckpt_path: str | None = None, device: str | None = None) -> None:
    httpd = ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(unit_dir, ckpt_path, device))
    net = f" + net ({ckpt_path})" if ckpt_path else ""
    print(f"sp_replay: {unit_dir} at http://127.0.0.1:{port}{net}")
    httpd.serve_forever()
