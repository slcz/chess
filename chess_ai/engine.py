"""
chess_ai/engine.py — Python binding over the C rules engine (cffi, ABI mode).

This is the Phase 0.3 boundary from PLAN §3: everything on the play path
(rules, later search) stays in C behind `engine/include/chess_ffi.h`; Python
drives it through this module. ABI-mode cffi (ffi.dlopen of the shared
library, no compile step) keeps `import chess_ai.engine` instant: rebuild
the C library with `make -C engine` and re-import.

Usage:
    from chess_ai.engine import Board, Termination

    b = Board()                    # standard start position
    b = Board(fen)                 # or any FEN
    b.legal_moves                   # ["a2a3", "a2a4", ...]
    b.push("e2e4"); b.pop()
    b.termination                   # Termination.PLAYING / CHECKMATE / ...
    b.perft(5)                      # 4,865,609 — the Phase 1 FFI gate

Design notes:
    - One Board = one C handle = one game (State + move log). Handles are
      not thread-safe (legal-move probing make/unmakes internally); use one
      Board per thread. Any number of Boards may coexist.
    - The library is loaded once per process; CHESS_LIB env var overrides
      the default path (engine/build/libchess.{dylib,so}) — the ASan build
      can be swapped in for a memory-checked run.
    - cffi releases the GIL during calls, so Boards in worker threads
      parallelize (the Phase 3 batched self-play design).
"""
from __future__ import annotations

import enum
import os
from pathlib import Path

from cffi import FFI

_REPO = Path(__file__).resolve().parents[1]

_CDEF = """
typedef struct ChessGame ChessGame;
void chess_init(void);
ChessGame *chess_game_new(const char *fen);
void chess_game_free(ChessGame *g);
int chess_set_fen(ChessGame *g, const char *fen);
void chess_fen(const ChessGame *g, char *out);
int chess_legal_moves(ChessGame *g, char *out, int max);
int chess_push_uci(ChessGame *g, const char *uci);
int chess_pop(ChessGame *g);
int chess_termination(ChessGame *g);
int chess_is_checkmate(ChessGame *g);
int chess_is_stalemate(ChessGame *g);
int chess_insufficient_material(ChessGame *g);
int chess_is_fifty_moves(ChessGame *g);
int chess_is_repetition(ChessGame *g, int count);
int chess_in_check(ChessGame *g);
int chess_side(ChessGame *g);
int chess_halfmove_clock(ChessGame *g);
int chess_ply(ChessGame *g);
int chess_piece_at(ChessGame *g, int sq);
uint64_t chess_perft(ChessGame *g, int depth);
int chess_encode(ChessGame *g, float *out);

/* --- Phase 3.1: the Gumbel search core (engine/include/chess_search.h) --- */

typedef struct {
    int n;
    int m_cap;
    double c_visit;
    double c_scale;
    double c_fpu;
    int sample_plies;      /* < 0 = None (sample every ply) */
    double resign_value;
} SpConfig;

enum {
    SP_EVENT_NEED_EVAL = 0,
    SP_EVENT_MOVED     = 1,
    SP_EVENT_DONE      = 2,
};

enum {
    SP_END_PLAYING = 0,
    SP_END_NATURAL,
    SP_END_RESIGN,
    SP_END_ADJUDICATED,
};

typedef struct SpGame SpGame;
void sp_config_init(SpConfig *cfg);

/* games live in CALLER-provided static storage (engine/include/chess_search.h,
 * deviation 9): one slot = sp_game_slot_size() bytes, 64-aligned */
SpGame *sp_game_init(void *storage, const SpConfig *cfg, const char *fen,
                     uint64_t seed, uint32_t game_index);
size_t sp_game_slot_size(void);
int  sp_game_step(SpGame *g);
const float    *sp_request_planes(const SpGame *g);
const uint16_t *sp_request_slots(const SpGame *g);
int  sp_request_count(const SpGame *g);
int  sp_request_mover(const SpGame *g);
int  sp_request_is_root(const SpGame *g);
void sp_deliver(SpGame *g, const double *logp, const double *wdl);
void sp_set_max_plies(SpGame *g, int max_plies);
int  sp_record_count(const SpGame *g);
int  sp_rec_ply(const SpGame *g, int i);
int  sp_rec_mover(const SpGame *g, int i);
void sp_rec_fen_before(const SpGame *g, int i, char *out);
int  sp_rec_played_slot(const SpGame *g, int i);
void sp_rec_played_uci(const SpGame *g, int i, char out[8]);
int  sp_rec_n_pi(const SpGame *g, int i);
int  sp_rec_pi_slot(const SpGame *g, int i, int j);
double sp_rec_pi_prob(const SpGame *g, int i, int j);
double sp_rec_outlook(const SpGame *g, int i);
int    sp_result_kind(const SpGame *g);
int    sp_result_resigner(const SpGame *g);
double sp_result_z_white(const SpGame *g);
void   sp_result_start_fen(const SpGame *g, char *out);
int    sp_result_game_index(const SpGame *g);
int    sp_result_n_sims(const SpGame *g);

/* --- the search RNG (engine/include/chess_rng.h; rng.py mirror) ---------- */

typedef struct {
    uint64_t s[4];
} Rng256;
uint64_t splitmix64_next(uint64_t *state);
uint64_t rng_game_seed(uint64_t unit_seed, uint64_t game_index);
void     rng_init(Rng256 *r, uint64_t seed);
uint64_t rng_next_u64(Rng256 *r);
double   rng_uniform(Rng256 *r);
double   rng_gumbel(Rng256 *r);
"""

_FEN_MAX = 96
_UCI_MAX = 8
_MAX_MOVES = 256
_NN_TOKENS = 64
_NN_TOKEN_DIM = 117          # layout spec: engine/include/chess_nn.h
_NN_TOTAL = _NN_TOKENS * _NN_TOKEN_DIM


def _load_library():
    """Locate and dlopen libchess; returns (ffi, lib)."""
    candidates = []
    env = os.environ.get("CHESS_LIB")
    if env:
        candidates.append(Path(env))
    else:
        candidates.append(_REPO / "engine" / "build" / "libchess.dylib")
        candidates.append(_REPO / "engine" / "build" / "libchess.so")

    ffi = FFI()
    ffi.cdef(_CDEF)
    for path in candidates:
        if path.exists():
            lib = ffi.dlopen(str(path))
            lib.chess_init()
            return ffi, lib
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"libchess not found (tried: {tried}). Build it first: make -C engine"
    )


ffi, _lib = _load_library()
lib = _lib  # public alias: the search binding (chess_ai.selfplay.csearch) shares it


class Termination(enum.IntEnum):
    """Mirror of the TERM_* codes in engine/include/chess.h.

    Same granularity and precedence as python-chess's
    Board.outcome(claim_draw=True) — that exactness is what lets the oracle
    fuzz (tests/test_engine.py) compare our answer to the oracle's one
    reason code at a time.
    """

    PLAYING = 0
    CHECKMATE = enum.auto()              # winner: the side NOT to move
    INSUFFICIENT_MATERIAL = enum.auto()  # dead material (KvK, KN/KBvK, ...)
    STALEMATE = enum.auto()
    SEVENTYFIVE_MOVES = enum.auto()      # automatic draw (clock >= 150)
    FIVEFOLD_REPETITION = enum.auto()    # automatic draw
    FIFTY_MOVES = enum.auto()            # claimable draw (clock >= 100)
    THREEFOLD_REPETITION = enum.auto()   # claimable draw


# Names in python-chess's Termination enum, for the fuzz parity map.
CHESS_PY_NAMES = {
    Termination.CHECKMATE: "CHECKMATE",
    Termination.INSUFFICIENT_MATERIAL: "INSUFFICIENT_MATERIAL",
    Termination.STALEMATE: "STALEMATE",
    Termination.SEVENTYFIVE_MOVES: "SEVENTYFIVE_MOVES",
    Termination.FIVEFOLD_REPETITION: "FIVEFOLD_REPETITION",
    Termination.FIFTY_MOVES: "FIFTY_MOVES",
    Termination.THREEFOLD_REPETITION: "THREEFOLD_REPETITION",
}


class IllegalMove(ValueError):
    """push() of a move that is not legal in the current position."""


class Board:
    """One chess game, backed by a C handle. See module docstring."""

    def __init__(self, fen: str | None = None):
        fen_bytes = fen.encode() if fen is not None else ffi.NULL
        self._g = _lib.chess_game_new(fen_bytes)
        if self._g == ffi.NULL:
            raise ValueError(f"malformed FEN: {fen!r}")
        self._buf = ffi.new("char[]", _MAX_MOVES * _UCI_MAX)

    def __del__(self):
        g = getattr(self, "_g", None)
        if g is not None and g != ffi.NULL:
            _lib.chess_game_free(g)
            self._g = ffi.NULL

    # --- position ------------------------------------------------------

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        return cls(fen)

    def set_fen(self, fen: str) -> None:
        if _lib.chess_set_fen(self._g, fen.encode()) != 0:
            raise ValueError(f"malformed FEN: {fen!r}")

    @property
    def fen(self) -> str:
        out = ffi.new("char[]", _FEN_MAX)
        _lib.chess_fen(self._g, out)
        return ffi.string(out).decode()

    # --- playing ---------------------------------------------------------

    @property
    def legal_moves(self) -> list[str]:
        """Current legal moves as UCI strings, in generation order."""
        n = _lib.chess_legal_moves(self._g, self._buf, _MAX_MOVES)
        raw = ffi.buffer(self._buf, n * _UCI_MAX)
        return [bytes(raw[i * _UCI_MAX:(i + 1) * _UCI_MAX]).split(b"\0", 1)[0].decode()
                for i in range(n)]

    def push(self, uci: str) -> None:
        if _lib.chess_push_uci(self._g, uci.encode()) != 0:
            raise IllegalMove(f"{uci!r} is not a legal move in {self.fen}")

    def pop(self) -> None:
        if _lib.chess_pop(self._g) != 0:
            raise IndexError("pop from an empty game")

    # --- observation -----------------------------------------------------

    @property
    def termination(self) -> Termination:
        return Termination(_lib.chess_termination(self._g))

    @property
    def is_check(self) -> bool:
        return bool(_lib.chess_in_check(self._g))

    @property
    def side_to_move(self) -> int:
        """0 = white, 1 = black."""
        return _lib.chess_side(self._g)

    @property
    def halfmove_clock(self) -> int:
        return _lib.chess_halfmove_clock(self._g)

    @property
    def ply(self) -> int:
        return _lib.chess_ply(self._g)

    def piece_at(self, square: int) -> int:
        """Piece code on `square` (a1=0 … h8=63): color*6+type, 12 = EMPTY."""
        return _lib.chess_piece_at(self._g, square)

    @property
    def is_game_over(self) -> bool:
        return self.termination != Termination.PLAYING

    def is_repetition(self, count: int = 3) -> bool:
        """Has the current position occurred `count` times (incl. now)?
        Granular predicate behind the THREEFOLD/FIVEFOLD termination codes,
        exposed for tests and eval tooling."""
        return bool(_lib.chess_is_repetition(self._g, count))

    @property
    def winner(self) -> int | None:
        """0 = white won, 1 = black won, None = draw/ongoing."""
        if self.termination == Termination.CHECKMATE:
            return 1 - self.side_to_move
        return None

    # --- perft (Phase 1 gate: identical counts through the FFI) ----------

    def perft(self, depth: int) -> int:
        return _lib.chess_perft(self._g, depth)

    # --- NN input encoding (Phase 2.1) ------------------------------------

    def encode(self) -> "np.ndarray":
        """Feature tensor of the current position, float32 [64, 117].

        The handle maintains the 8-ply placement-history ring behind the
        scenes (recorded on push, dropped on pop), so this is always in
        step with the game. Layout spec: engine/include/chess_nn.h, with a
        byte-exact Python reference in chess_ai/net/features.py (the tests
        assert the two agree tensor-for-tensor).
        """
        import numpy as np  # local import: keep module import light

        buf = ffi.new(f"float[{_NN_TOTAL}]")
        rc = _lib.chess_encode(self._g, buf)
        if rc != 0:
            raise RuntimeError("chess_encode failed")
        return np.frombuffer(ffi.buffer(buf, _NN_TOTAL * 4), dtype=np.float32) \
                 .reshape(_NN_TOKENS, _NN_TOKEN_DIM).copy()

    # --- niceties ---------------------------------------------------------

    def __repr__(self) -> str:
        return f"Board({self.fen!r})"
