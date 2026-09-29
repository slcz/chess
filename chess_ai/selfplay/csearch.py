"""
chess_ai/selfplay/csearch.py — Python binding over the C Gumbel search
(Phase 3.1: engine/src/search.c via chess_ai.engine's ABI cffi).

This is the *driver half* of the L2 comparison contract
(PARALLEL_DESIGN R6): the C search runs as the per-game state machine
`sp_game_step` / `sp_deliver` (the same entry points the 3.3b parallel
worker loop will use), and this module is the sequential driver that
fulfils each evaluation request in Python. cffi ABI mode has no callbacks,
so the request flow is pull-based — which is exactly the worker/evaluator
split of the parallel design in miniature:

    while True:
        ev = lib.sp_game_step(game)
        if ev == SP_EVENT_NEED_EVAL:
            planes, slots, mover = read request          # sp_request_*
            logp, wdl = evaluator.evaluate(...)          # any Python evaluator
            lib.sp_deliver(game, logp_aligned_to_slots, wdl)
        elif ev == SP_EVENT_DONE:
            break

CSpGame.play() returns the same `GameRecord` dataclass the Python reference
(chess_ai/selfplay/gumbel.py) produces, so tests compare records field for
field (tests/test_search_l2.py — the L2 exact-diff gate: same seeds ⇒
identical games, π′ within 1e-6).

The evaluation request is the parallel design's batch element, read-only
over the boundary: planes (the same nn_encode floats chess_encode
produces), the legal policy slots in movegen enumeration order (the order
the logp array must be aligned to — it is the order the RNG and every
tie-break are specified over), the mover, and an is-root flag (handoff
telemetry). sp_deliver consumes doubles: policy_logp widened to double
(the net's f32 tensor, widened exactly, both sides) and the WDL triple.
"""
from __future__ import annotations

import ctypes
import mmap
from typing import Optional

import numpy as np

from chess_ai.engine import ffi, lib
from chess_ai.net.features import NN_TOKEN_DIM, NN_TOKENS
from chess_ai.selfplay.gumbel import GameRecord, MoveRecord, SelfPlayConfig

_NN_TOTAL = NN_TOKENS * NN_TOKEN_DIM          # 7488 floats per request
_FEN_MAX = 96                                 # chess_ffi.h CHESS_FEN_MAX

_EVENT_DONE = lib.SP_EVENT_DONE
_EVENT_NEED_EVAL = lib.SP_EVENT_NEED_EVAL

_END_KINDS = {
    lib.SP_END_NATURAL: "natural",
    lib.SP_END_RESIGN: "resign",
    lib.SP_END_ADJUDICATED: "adjudicated",
}


def game_record_from_handle(ffi, lib, g, *, seed=0, game_index=0):
    """Export one SpGame handle's records into gumbel.py's GameRecord
    shape. Sequential-driver only — the parallel engine's records come
    back through its shard files instead (spengine.read_records)."""
    moves = []
    fen_c = ffi.new(f"char[{_FEN_MAX}]")
    uci_c = ffi.new("char[8]")
    for i in range(lib.sp_record_count(g)):
        n_pi = lib.sp_rec_n_pi(g, i)
        lib.sp_rec_fen_before(g, i, fen_c)
        lib.sp_rec_played_uci(g, i, uci_c)
        moves.append(MoveRecord(
            ply=lib.sp_rec_ply(g, i),
            mover=lib.sp_rec_mover(g, i),
            fen_before=ffi.string(fen_c).decode("ascii"),
            played_uci=ffi.string(uci_c).decode("ascii"),
            played_slot=lib.sp_rec_played_slot(g, i),
            pi_slots=[lib.sp_rec_pi_slot(g, i, j) for j in range(n_pi)],
            pi_probs=[lib.sp_rec_pi_prob(g, i, j) for j in range(n_pi)],
            outlook=lib.sp_rec_outlook(g, i)))
    kind = _END_KINDS[lib.sp_result_kind(g)]
    resigner = lib.sp_result_resigner(g)
    lib.sp_result_start_fen(g, fen_c)
    return GameRecord(
        start_fen=ffi.string(fen_c).decode("ascii"),
        moves=moves,
        z_white=lib.sp_result_z_white(g),
        outcome_kind=kind,
        resigner=None if resigner < 0 else resigner,
        seed=seed,
        game_index=game_index,
        n_sims=lib.sp_result_n_sims(g))


class CSpGame:
    """One game inside the C search engine — the mirror of gumbel.py's
    SelfPlayGame with the same constructor knobs and the same play() loop.

    `cfg` is the reference's SelfPlayConfig (field-for-field); `seed` is
    the work-unit seed and `game_index` the slot — the per-game RNG is
    derived inside C exactly as rng.py does, so identical (seed,
    game_index) pairs replay identical games in both engines."""

    def __init__(self, cfg: SelfPlayConfig, seed: int, game_index: int,
                 start_fen: Optional[str] = None):
        self.cfg = cfg
        self.seed = seed
        self.game_index = game_index
        # the game lives in caller-provided static storage (deviation 9):
        # one anonymous mmap slot, closed on close(). The ctypes view is
        # a buffer export — released in close() before the mmap closes.
        self._mm = mmap.mmap(-1, lib.sp_game_slot_size())
        self._view = ctypes.c_char.from_buffer(self._mm)
        addr = ctypes.addressof(self._view)
        self._cfg_c = ffi.new("SpConfig *")
        self._cfg_c.n = cfg.n
        self._cfg_c.m_cap = cfg.m_cap
        self._cfg_c.c_visit = cfg.c_visit
        self._cfg_c.c_scale = cfg.c_scale
        self._cfg_c.c_fpu = cfg.c_fpu
        self._cfg_c.sample_plies = (-1 if cfg.sample_plies is None
                                    else cfg.sample_plies)
        self._cfg_c.resign_value = cfg.resign_value
        fen_c = ffi.NULL if not start_fen else start_fen.encode("ascii")
        self.g = lib.sp_game_init(ffi.cast("void *", addr), self._cfg_c,
                                  fen_c, seed & 0xFFFFFFFFFFFFFFFF,
                                  game_index)
        if not self.g:
            self._mm.close()
            raise ValueError(f"sp_game_init failed (bad FEN? {start_fen!r})")

    # -- the driver loop --------------------------------------------------

    def play(self, evaluator, max_plies: Optional[int] = None) -> GameRecord:
        """Play to natural end / resignation / adjudication, fulfilling
        every evaluation synchronously through `evaluator` (gumbel.py's
        Evaluator interface). Returns the GameRecord."""
        if max_plies is not None:
            lib.sp_set_max_plies(self.g, max_plies)
        while True:
            ev = lib.sp_game_step(self.g)
            if ev == _EVENT_DONE:
                break
            if ev == _EVENT_NEED_EVAL:
                self.fulfil(evaluator)
        return self.game_record()

    def fulfil(self, evaluator) -> None:
        """One evaluation request: read it over the boundary, call the
        evaluator, deliver the answer aligned to the request's slot order."""
        g = self.g
        n = lib.sp_request_count(g)
        planes = np.frombuffer(
            ffi.buffer(ffi.cast("const float *", lib.sp_request_planes(g)),
                       _NN_TOTAL * 4),
            dtype=np.float32)
        slots_ptr = lib.sp_request_slots(g)
        slots = [slots_ptr[i] for i in range(n)]
        mover = lib.sp_request_mover(g)
        logp, wdl = evaluator.evaluate(planes, slots, mover)
        logp_c = ffi.new("double[]", [float(logp[s]) for s in slots])
        wdl_c = ffi.new("double[]", [float(w) for w in wdl])
        lib.sp_deliver(g, logp_c, wdl_c)

    # -- telemetry ----------------------------------------------------------

    def request_is_root(self) -> int:
        """1 when the pending request is a root evaluation (handoff
        telemetry — every root after move 1 should be free)."""
        return lib.sp_request_is_root(self.g)

    # -- records ------------------------------------------------------------

    def game_record(self) -> GameRecord:
        """Export the game's records into gumbel.py's GameRecord shape."""
        return game_record_from_handle(ffi, lib, self.g, seed=self.seed,
                                       game_index=self.game_index)

    # -- lifecycle ------------------------------------------------------------

    def close(self) -> None:
        if getattr(self, "g", None):
            self.g = None
        if getattr(self, "_view", None):
            del self._view          # release the buffer export first
        if getattr(self, "_mm", None):
            self._mm.close()
            self._mm = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
