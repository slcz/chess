"""
chess_ai/selfplay/azsearch.py — Python binding over the C AZ-PUCT search
(engine/src/az.c via chess_ai.engine's ABI cffi), the driver half of the
az_eval port (the reference is scripts/az_eval.py, kept untouched as the
clean-room second opinion).

Same pull-model shape as csearch.CSpGame over the Gumbel search: the C
game parks each NN evaluation request, Python reads it over the boundary,
calls any Evaluator (gumbel.py's interface), and delivers the answer
aligned to the request's slot order:

    game = CAzGame()
    game.think(evaluator, sims=128, c_puct=1.25)
    children = game.root_children()      # (uci, visits, prior, q) rows
    uci = choose_move_visits(children, rng, sample=ply < sample_plies)
    game.play_uci(uci)

Move selection stays HERE on purpose: the sampled opening phase draws from
Python's random.Random (az_eval.py's per-game stream), so the R5-twin
protocol is bit-compatible with the Python engine. The C side has no RNG
at all.

The evaluator interface is the shared one, so the ladder's
BatchedInference service plugs in directly: run one CAzGame per game
thread, fulfil through the shared service, and every thread's requests
pool into GPU batches (one forward per net per batch). cffi releases the
GIL around the C calls, so the per-game searches — single-threaded in C,
as designed — genuinely run in parallel between parks.

Q SIGN (see chess_az.h's header for the full story): az_eval.py scores
children by their own-mover q UNNEGATED (the fleet-measured formula);
the AlphaGo paper's Q(s,a) is the parent's view. think()'s
q_parent_view=False is the verbatim port (default, matches the Python
engine exactly); q_parent_view=True flips to the paper's sign.
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from chess_ai.engine import ffi, lib
from chess_ai.net.features import NN_TOKEN_DIM, NN_TOKENS

_NN_TOTAL = NN_TOKENS * NN_TOKEN_DIM          # 7488 floats per request
_FEN_MAX = 96                                 # chess_ffi.h CHESS_FEN_MAX
_UCI_MAX = 8                                  # chess_ffi.h CHESS_UCI_MAX

_EVENT_NEED_EVAL = lib.AZ_EVENT_NEED_EVAL
_EVENT_DONE = lib.AZ_EVENT_DONE

TERM_PLAYING = 0                              # chess.h TERM_* (engine.Termination mirrors)


class CAzGame:
    """One game driven by the C AZ-PUCT search — az_eval.py's play_game
    loop with the walk in C. One handle per game; one thread per handle
    (handles share only the read-only chess_init tables)."""

    def __init__(self, fen: Optional[str] = None):
        fen_c = ffi.NULL if not fen else fen.encode("ascii")
        self.g = lib.az_game_new(fen_c)
        if not self.g:
            raise ValueError(f"az_game_new failed (bad FEN? {fen!r})")

    # -- the driver loop --------------------------------------------------

    def think(self, evaluator, sims: int, c_puct: float,
              q_parent_view: bool = False) -> None:
        """Run `sims` PUCT simulations on the current position through
        `evaluator` (gumbel.py's Evaluator interface), leaving the root
        statistics for root_children()/root_value()."""
        lib.az_think(self.g, sims, c_puct, 1 if q_parent_view else 0)
        while lib.az_step(self.g) == _EVENT_NEED_EVAL:
            self.fulfil(evaluator)

    def fulfil(self, evaluator) -> None:
        """One evaluation request: read it over the boundary, call the
        evaluator, deliver the answer aligned to the request's slot order
        (movegen enumeration order — what root_children() indexes by)."""
        g = self.g
        n = lib.az_request_count(g)
        planes = np.frombuffer(
            ffi.buffer(ffi.cast("const float *", lib.az_request_planes(g)),
                       _NN_TOTAL * 4),
            dtype=np.float32)
        slots_ptr = lib.az_request_slots(g)
        slots = [slots_ptr[i] for i in range(n)]
        mover = lib.az_request_mover(g)
        logp, wdl = evaluator.evaluate(planes, slots, mover)
        logp_c = ffi.new("double[]", [float(logp[s]) for s in slots])
        wdl_c = ffi.new("double[]", [float(w) for w in wdl])
        lib.az_deliver(g, logp_c, wdl_c)

    # -- root statistics ----------------------------------------------------

    def root_children(self) -> list[tuple[str, int, float, float]]:
        """The root's legal moves as (uci, visits, prior, q) rows in
        movegen enumeration order. Unvisited children report visits 0 /
        q 0.0 (Python Node.q's convention); q is the child's OWN mover's
        view — the number az_eval.py's analyse mode prints."""
        g = self.g
        out = []
        uci_c = ffi.new(f"char[{_UCI_MAX}]")
        for i in range(lib.az_root_count(g)):
            lib.az_root_uci(g, i, uci_c)
            out.append((ffi.string(uci_c).decode("ascii"),
                        lib.az_root_visits(g, i),
                        lib.az_root_prior(g, i),
                        lib.az_root_q(g, i)))
        return out

    def root_value(self) -> float:
        """The root's overall value estimate — az_eval.py's effective
        search() second return (its root_v is dead code, so the reference
        surfaces root.q: the visits-weighted backup value, the root
        mover's view)."""
        return lib.az_root_value(self.g)

    # -- the game ------------------------------------------------------------

    def play_uci(self, uci: str) -> None:
        if lib.az_play_uci(self.g, uci.encode("ascii")) != 0:
            raise ValueError(f"illegal move {uci!r} in {self.fen}")

    @property
    def termination(self) -> int:
        """TERM_* code (chess.h); 0 = PLAYING. Use chess_ai.engine's
        Termination enum for named comparisons."""
        return lib.az_termination(self.g)

    @property
    def winner(self) -> int:
        """-1 none (draw/ongoing), else 0 white / 1 black — Board.winner's
        convention."""
        return lib.az_winner(self.g)

    @property
    def side(self) -> int:
        return lib.az_side(self.g)

    @property
    def ply(self) -> int:
        return lib.az_ply(self.g)

    @property
    def fen(self) -> str:
        buf = ffi.new(f"char[{_FEN_MAX}]")
        lib.az_fen(self.g, buf)
        return ffi.string(buf).decode("ascii")

    def close(self) -> None:
        if getattr(self, "g", None):
            lib.az_game_free(self.g)
            self.g = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def choose_move_visits(children, rng: random.Random, sample: bool) -> str:
    """az_eval.py's choose_move over root_children() rows: argmax visits
    (ties by movegen order via the stable sort), or inverse-CDF ∝ visits
    during the sampled opening phase. Same rng consumption as the Python
    engine — one random() draw per sampled ply, none otherwise."""
    items = sorted(children, key=lambda c: -c[1])
    if sample:
        total = sum(c[1] for c in items)
        r, acc = rng.random() * total, 0.0
        for c in items:
            acc += c[1]
            if r < acc:
                return c[0]
    return items[0][0]


def play_az_game(white, black, *, sims: int, c_puct: float = 1.25,
                 q_parent_view: bool = False, rng: random.Random,
                 sample_plies: int = 10, max_plies: Optional[int] = None,
                 fen: Optional[str] = None,
                 log=None) -> dict:
    """One C-engine game — az_eval.py's play_game, walk in C: each side
    searches through its own evaluator, picks argmax-visits (∝ visits
    during the first `sample_plies` plies, per-game rng), plays it here.
    Returns {"winner": None/0/1, "plies", "kind": natural|adjudicated,
    "moves": [uci...]} — the caller scores it against its own seat map.

    `white`/`black` are evaluators (gumbel.py's Evaluator interface);
    pass the same BatchedInference service side to both seats of a match.
    """
    game = CAzGame(fen)
    try:
        moves = []
        while game.termination == TERM_PLAYING and (
                max_plies is None or game.ply < max_plies):
            ev = white if game.side == 0 else black
            game.think(ev, sims, c_puct, q_parent_view)
            uci = choose_move_visits(game.root_children(), rng,
                                     game.ply < sample_plies)
            game.play_uci(uci)
            moves.append(uci)
            if log:
                log(game.ply, uci)
        w = game.winner
        kind = ("natural" if game.termination != TERM_PLAYING
                else "adjudicated")
        return {"winner": None if w < 0 else w, "plies": game.ply,
                "kind": kind, "moves": moves}
    finally:
        game.close()
