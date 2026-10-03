"""
chess_ai/selfplay/spengine.py — Python binding over the parallel self-play
engine (Phase 3.3b: engine/src/sp_engine.c in engine/build/libchess_sp,
which also carries the rules + search + the embedded-CPython evaluator).

This is the orchestration seam of D7: Python launches a whole self-play
run (cfg.n_workers C worker threads + this thread playing the evaluator
between epochs, running the net) with one blocking call — cffi releases
the GIL while it runs, so the evaluator role can take the GIL for its
batched forward passes (F8). Finished games are DUMPED by the workers to
per-thread shard files in record_dir (native format, chess_sp_dump.h) at
collect time — no callback crosses C→Python on the record path. play()
reads the shards back and returns gumbel.py GameRecords; the trainer-side
conversion (shards → tensors) builds on the same reader later (3.4).

    from chess_ai.selfplay.spengine import SpRun
    run = SpRun(cfg, unit_seed=..., game_budget=...,
                record_dir="runs/unit0")
    records = run.play()          # [GameRecord], one per budgeted game

The evaluator's Python implementation lives in sp_evaluator.py ("fixture"
for batch-invariant tests, "torch" for the real net).

Build note: libchess_sp is built by `make -C engine build/libchess_sp.dylib`
(the pure rules library libchess stays Python-free; only this library
embeds the interpreter-bound evaluator).
"""
from __future__ import annotations

import ctypes
import mmap
import os
import struct
from pathlib import Path

from cffi import FFI

from chess_ai.engine import Board
from chess_ai.selfplay.gumbel import GameRecord, MoveRecord, SelfPlayConfig

_REPO = Path(__file__).resolve().parents[2]

_CDEF = """
void chess_init(void);

typedef struct {
    int n;
    int m_cap;
    double c_visit;
    double c_scale;
    double c_fpu;
    int sample_plies;
    double resign_value;
} SpConfig;

typedef struct SpEvalBatch SpEvalBatch;
typedef struct SpEngine SpEngine;

size_t sp_engine_static_size(void);

typedef struct {
    void         *ctx;
    int  (*init)(void *ctx);
    int  (*batch)(void *ctx, SpEvalBatch *b);
    void (*free_fn)(void *ctx);
} SpEvaluator;

typedef struct {
    SpConfig game_cfg;
    uint64_t unit_seed;
    int32_t  game_budget;
    int32_t  n_workers;
    int32_t  buffer_requests;
    int32_t  max_plies;
    int32_t  interleave;
    char     record_dir[256];   /* shard dump dir; empty = no dump */
    const char *const *start_fens;   /* diverse game init pool; NULL = off */
    int32_t  n_start_fens;
} SpEngineConfig;

void sp_engine_config_init(SpEngineConfig *cfg);
SpEngine *sp_engine_init(void *storage, size_t storage_size,
                         const SpEngineConfig *cfg, const SpEvaluator *ev);
void      sp_engine_destroy(SpEngine *e);
int       sp_engine_start(SpEngine *e);
void      sp_engine_wait(SpEngine *e);
int       sp_engine_error(const SpEngine *e);
int    sp_engine_games_started(const SpEngine *e);
int    sp_engine_games_done(const SpEngine *e);
long   sp_engine_evals(const SpEngine *e);
long   sp_engine_epochs(const SpEngine *e);
int    sp_engine_max_batch(const SpEngine *e);
int    sp_py_evaluator_make(const char *module, const char *spec,
                            SpEvaluator *ev);
"""

_FEN_MAX = 96

# ---------------------------------------------------------------------------
# The shard reader — the Python twin of engine/src/sp_dump.c (the format is
# documented in engine/include/chess_sp_dump.h; little-endian, packed).
# Per-ply FENs are not stored: the reader replays played_uci from
# start_fen — the same rule as the sequential exporter, so both consume
# identical information. The resign ply (played_slot = -1) pushes nothing.
# ---------------------------------------------------------------------------

SHARD_MAGIC = b"SPSHARD1"
_SHARD_HDR = struct.Struct("<8sIIQ")     # magic, version, worker, unit_seed
_GAME_HDR = struct.Struct("<IIBbId")     # gi, n_plies, kind, resigner(i8),
                                         # n_sims, z
_PLY_HDR = struct.Struct("<Bh8sHd")      # mover, played_slot, uci, n_pi, outlook
_PI = struct.Struct("<Hf")               # slot, prob
_END_KINDS = ("playing", "natural", "resign", "adjudicated")


def shard_path(record_dir: str | Path, worker: int) -> Path:
    """The one shard filename convention (sp_dump_shard_path's twin)."""
    return Path(record_dir) / f"shard-w{worker}.spbin"


def _read_shard(path: Path, seed: int) -> list[GameRecord]:
    data = path.read_bytes()
    magic, version, _worker, _unit_seed = _SHARD_HDR.unpack_from(data, 0)
    if magic != SHARD_MAGIC or version != 1:
        raise ValueError(f"{path}: not a v1 self-play shard")
    records = []
    off = _SHARD_HDR.size
    while off < len(data):
        gi, n_plies, kind, resigner, n_sims, z = _GAME_HDR.unpack_from(data, off)
        off += _GAME_HDR.size
        start_fen = data[off:off + _FEN_MAX].split(b"\0", 1)[0].decode("ascii")
        off += _FEN_MAX
        board = Board(start_fen)
        moves = []
        for i in range(n_plies):
            mover, played_slot, uci_b, n_pi, outlook = _PLY_HDR.unpack_from(data, off)
            off += _PLY_HDR.size
            slots, probs = [], []
            for _ in range(n_pi):
                slot, prob = _PI.unpack_from(data, off)
                off += _PI.size
                slots.append(slot)
                probs.append(prob)
            played_uci = uci_b.split(b"\0", 1)[0].decode("ascii")
            moves.append(MoveRecord(
                ply=i, mover=mover, fen_before=board.fen,
                played_uci=played_uci, played_slot=played_slot,
                pi_slots=slots, pi_probs=probs, outlook=outlook))
            if played_uci:
                board.push(played_uci)
        records.append(GameRecord(
            start_fen=start_fen, moves=moves, z_white=z,
            outcome_kind=_END_KINDS[kind],
            resigner=None if resigner < 0 else resigner,
            seed=seed, game_index=gi, n_sims=n_sims))
    return records


def read_records(record_dir: str | Path, n_workers: int,
                 seed: int) -> list[GameRecord]:
    """All games from a run's shards, sorted by game_index."""
    records: list[GameRecord] = []
    for w in range(n_workers):
        records.extend(_read_shard(shard_path(record_dir, w), seed))
    records.sort(key=lambda r: r.game_index)
    return records


def _load_library():
    """Locate and dlopen libchess_sp (engine/build next to libchess)."""
    candidates = []
    env = os.environ.get("CHESS_SP_LIB")
    if env:
        candidates.append(Path(env))
    else:
        for ext in ("dylib", "so"):
            candidates.append(_REPO / "engine" / "build" / f"libchess_sp.{ext}")
    ffi = FFI()
    ffi.cdef(_CDEF)
    for path in candidates:
        if path.exists():
            lib = ffi.dlopen(str(path))
            lib.chess_init()
            return ffi, lib
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"libchess_sp not found (tried: {tried}). "
        "Build it: make -C engine build/libchess_sp.dylib")


ffi, lib = _load_library()


class SpRun:
    """One parallel self-play run. play() blocks until game_budget games
    have completed, then reads the record shards back (the GIL is released
    while the engine runs — cffi releases it around C calls — so the
    evaluator role can use torch)."""

    def __init__(self, cfg: SelfPlayConfig, unit_seed: int,
                 game_budget: int, n_workers: int = 8,
                 buffer_requests: int = 256, max_plies: int | None = None,
                 interleave: bool = False, eval_spec: str = "fixture",
                 eval_module: str = "chess_ai.selfplay.sp_evaluator",
                 record_dir: str | Path | None = None,
                 start_fens: list[str] | None = None):
        self.cfg = cfg
        self.unit_seed = unit_seed
        self.game_budget = game_budget
        self.n_workers = n_workers
        self.eval_spec = eval_spec
        if record_dir is None:
            raise ValueError("SpRun needs record_dir — finished games are "
                             "dumped to per-worker shard files there")
        self.record_dir = str(record_dir)

        self._cfg_c = ffi.new("SpEngineConfig *")
        lib.sp_engine_config_init(self._cfg_c)
        self._cfg_c.game_cfg.n = cfg.n
        self._cfg_c.game_cfg.m_cap = cfg.m_cap
        self._cfg_c.game_cfg.c_visit = cfg.c_visit
        self._cfg_c.game_cfg.c_scale = cfg.c_scale
        self._cfg_c.game_cfg.c_fpu = cfg.c_fpu
        self._cfg_c.game_cfg.sample_plies = (-1 if cfg.sample_plies is None
                                             else cfg.sample_plies)
        self._cfg_c.game_cfg.resign_value = cfg.resign_value
        self._cfg_c.unit_seed = unit_seed & 0xFFFFFFFFFFFFFFFF
        self._cfg_c.game_budget = game_budget
        self._cfg_c.n_workers = n_workers
        self._cfg_c.buffer_requests = buffer_requests
        self._cfg_c.max_plies = -1 if max_plies is None else max_plies
        self._cfg_c.interleave = 1 if interleave else 0
        self._cfg_c.record_dir = str(record_dir).encode()

        # Diverse game init (3.5 finding): game i starts from
        # start_fens[i % len]. The engine copies only the config struct, so
        # the pointed-to char buffers must outlive the run — cffi's array
        # does not keep its pointees alive, so the buffers AND the array
        # both live on self.
        self._fen_bufs: list = []
        self._fen_arr = None
        if start_fens:
            if not all(0 < len(f.encode()) < 96 for f in start_fens):
                raise ValueError("start_fens entries must be non-empty FENs "
                                 "under 96 bytes")
            self._fen_bufs = [ffi.new("char[]", f.encode())
                              for f in start_fens]
            self._fen_arr = ffi.new("char *[]", self._fen_bufs)
            self._cfg_c.start_fens = self._fen_arr
            self._cfg_c.n_start_fens = len(start_fens)

        # The evaluator's C context copies the strings (strdup), but keep
        # the cffi buffers alive anyway for the whole run.
        self._module_c = ffi.new("char[]", eval_module.encode())
        self._spec_c = ffi.new("char[]", eval_spec.encode())
        self._ev = ffi.new("SpEvaluator *")
        if lib.sp_py_evaluator_make(self._module_c, self._spec_c, self._ev) != 0:
            raise RuntimeError("sp_py_evaluator_make failed")

        # The engine's static storage: one anonymous mmap (lazily
        # committed — the ~2.3 GB bound costs only what the run touches).
        # The ctypes view is a buffer EXPORT: it must be released before
        # the mmap can close, so keep it on self and drop it in close().
        self._mm = mmap.mmap(-1, lib.sp_engine_static_size())
        self._view = ctypes.c_char.from_buffer(self._mm)
        addr = ctypes.addressof(self._view)
        self._e = lib.sp_engine_init(ffi.cast("void *", addr),
                                     lib.sp_engine_static_size(),
                                     self._cfg_c, self._ev)
        if not self._e:
            raise RuntimeError("sp_engine_init failed")

    def play(self) -> list[GameRecord]:
        """Run to completion; return the records read back from the shard
        files, sorted by game_index. The engine stays alive afterwards for
        telemetry() — close() frees it."""
        if lib.sp_engine_start(self._e) != 0:
            raise RuntimeError("sp_engine_start failed")
        lib.sp_engine_wait(self._e)
        if lib.sp_engine_error(self._e):
            raise RuntimeError("engine run failed (evaluator error?)")
        records = read_records(self.record_dir, self.n_workers,
                               self.unit_seed)
        assert len(records) == self.game_budget, (
            f"shards hold {len(records)} of {self.game_budget}")
        return records

    def telemetry(self) -> dict:
        """Run counters (valid after play, before close)."""
        return {
            "games_started": lib.sp_engine_games_started(self._e),
            "games_done": lib.sp_engine_games_done(self._e),
            "evals": lib.sp_engine_evals(self._e),
            "epochs": lib.sp_engine_epochs(self._e),
            "max_batch": lib.sp_engine_max_batch(self._e),
        }

    def close(self) -> None:
        if getattr(self, "_e", None):
            lib.sp_engine_destroy(self._e)
            self._e = None
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
