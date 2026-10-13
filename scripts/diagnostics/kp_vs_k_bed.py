#!/usr/bin/env python3
"""
probe_labels/kp_vs_k_bed.py — TEMPORARY feasibility check (not pipeline code).

Validates the "simplest ground-truth task" idea end to end:
  1. build exact K+P vs K positions with the ENGINE's own State + the real
     FEN parser (no hand-written FEN generator to get wrong),
  2. play full games from those positions with one net on both sides through
     engine/build/uci_mux (the driver that HAS a per-game FEN hook; plain
     `selfplay` hardcodes startpos),
  3. report the pawn side's conversion rate at two node budgets.

Why this is the measurement that matters for the "do we need a simpler
task to prove the loop is sane" question: K+P vs K has a known answer (the
pawn promotes), a tiny tree, and cheap games, so it has a WIDE dynamic
range — if a fresh net already converts everything at n=64 the task cannot
show learning, and if it converts nothing at n=600 the task is out of reach.
The reported numbers tell you which regime the current net is in.

Usage: .venv/bin/python probe_labels/kp_vs_k_bed.py [n_pos] [sims_low] [sims_high]
"""
from __future__ import annotations

import ctypes as C
import os
import random
import subprocess
import sys
from pathlib import Path

import chess  # noqa: E402  (python-chess is the test-only referee)

ROOT = Path(__file__).resolve().parent.parent
LIB = C.CDLL(str(ROOT / "engine/build/libchess.dylib"))
LIB.chess_init()

MAX_PLY = 1024
EMPTY = 12
FILES = "abcdefgh"


class Undo(C.Structure):
    _fields_ = [("captured", C.c_uint8), ("castling", C.c_uint8),
                ("ep", C.c_uint8), ("halfmove", C.c_uint16),
                ("key", C.c_uint64)]


class State(C.Structure):
    _fields_ = [("board", C.c_uint8 * 64),
                ("bb", (C.c_uint64 * 6) * 2),
                ("occ", C.c_uint64 * 2),
                ("side", C.c_uint8), ("castling", C.c_uint8),
                ("ep", C.c_uint8), ("halfmove", C.c_uint16),
                ("fullmove", C.c_uint32), ("ply", C.c_uint16),
                ("hist", Undo * MAX_PLY),
                ("keys", C.c_uint64 * (MAX_PLY + 1))]

LIB.fen_serialize.argtypes = [C.POINTER(State), C.c_char_p, C.c_size_t]
LIB.fen_parse.argtypes = [C.POINTER(State), C.c_char_p]
LIB.fen_parse.restype = C.c_int


def sq_of(file: int, rank: int) -> int:
    return rank * 8 + file


def make_fen(wk: int, bk: int, wp: int, side: int) -> str:
    """Build a K+P vs K position directly in the engine's State."""
    st = State()
    for i in range(64):
        st.board[i] = EMPTY
    st.board[wk] = 5                  # white king  (0*6+5)
    st.board[bk] = 11                 # black king  (1*6+5)
    st.board[wp] = 0                  # white pawn  (0*6+0)
    st.bb[0][5] = 1 << wk
    st.bb[1][5] = 1 << bk
    st.bb[0][0] = 1 << wp
    st.occ[0] = (1 << wk) | (1 << wp)
    st.occ[1] = 1 << bk
    st.side = side
    st.ep = 64                        # NO_SQ
    st.fullmove = 1
    out = C.create_string_buffer(96)
    LIB.fen_serialize(C.byref(st), out, 96)
    fen = out.value.decode()
    chk = State()
    assert LIB.fen_parse(C.byref(chk), fen.encode()) == 0, f"unparseable {fen}"
    # the parser must agree with what we wrote (guards the byte layout)
    assert chk.board[wk] == 5 and chk.board[bk] == 11 and chk.board[wp] == 0, \
        f"state layout mismatch for {fen}"
    assert chk.side == side, f"side mismatch for {fen}"
    return fen


def build_positions(n: int, seed: int = 20261011) -> list[str]:
    rng = random.Random(seed)
    fens, seen = [], set()
    guard = 0
    while len(fens) < n and guard < 20000:
        guard += 1
        wk = sq_of(rng.randrange(8), rng.randrange(8))
        bk = sq_of(rng.randrange(8), rng.randrange(8))
        wp = sq_of(rng.randrange(8), rng.randrange(1, 5))     # ranks 2..5
        if len({wk, bk, wp}) != 3:
            continue
        if max(abs(wk % 8 - bk % 8), abs(wk // 8 - bk // 8)) <= 1:
            continue
        side = rng.randrange(2)
        if side == 1 and max(abs(bk % 8 - wp % 8), abs(bk // 8 - wp // 8)) == 1:
            continue                                          # pawn hangs
        fen = make_fen(wk, bk, wp, side)
        if fen in seen:
            continue
        seen.add(fen)
        fens.append(fen)
    return fens


class Mux:
    """One uci_mux process hosting every game of a wave (PUCT_CKPT picks
    the net). Protocol: 'game <id> fen <fen>', 'game <id> go nodes <n>',
    'run', then one 'game <id> bestmove <uci>' per queued game."""

    def __init__(self, ckpt: Path, games: int, rows: int, device: str):
        env = dict(os.environ, PUCT_CKPT=str(ckpt))
        self.p = subprocess.Popen(
            [str(ROOT / "engine/build/uci_mux"), str(games), str(rows), device],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        if self.p.stdout.readline().strip() != "ready":
            raise SystemExit("uci_mux failed to attach")

    def send(self, line: str) -> None:
        self.p.stdin.write(line + "\n")
        self.p.stdin.flush()

    def read(self) -> str:
        return self.p.stdout.readline().strip()

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait(timeout=30)


def play_wave(ckpt: Path, fens: list[str], sims: int, device: str,
              cap: int = 120) -> dict:
    boards = [chess.Board(f) for f in fens]
    mux = Mux(ckpt, len(fens), 8 * len(fens), device)
    for gid, fen in enumerate(fens):
        mux.send(f"game {gid} fen {fen}")
    alive = list(range(len(fens)))
    plies = 0
    while alive and plies < cap:
        pend = [g for g in alive if not boards[g].is_game_over(claim_draw=True)]
        for gid in pend:
            mux.send(f"game {gid} go nodes {sims}")
        if not pend:
            break
        mux.send("run")
        for gid in pend:
            parts = mux.read().split()
            if len(parts) < 4 or parts[0] != "game" or parts[2] != "bestmove":
                raise SystemExit(f"protocol error: {parts}")
            boards[gid].push_uci(parts[3])
        plies += 1
        alive = [g for g in alive if not boards[g].is_game_over(claim_draw=True)]
    res = {"pawn_wins": 0, "draws": 0, "pawn_losses": 0, "unfinished": 0,
           "mean_plies": 0}
    tot = 0
    for gid, b in enumerate(boards):
        tot += b.ply()
        out = b.outcome(claim_draw=True) if b.is_game_over(claim_draw=True) else None
        if out is None:
            res["unfinished"] += 1
        elif out.winner is None:
            res["draws"] += 1
        elif out.winner == chess.WHITE:
            res["pawn_wins"] += 1
        else:
            res["pawn_losses"] += 1
    res["mean_plies"] = tot / len(boards)
    mux.close()
    return res


def main() -> int:
    n_pos = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    budgets = [int(x) for x in sys.argv[2:]] or [64, 600]
    fens = build_positions(n_pos)
    print(f"built {len(fens)} K+P vs K positions from the engine's own State")
    for f in fens[:3]:
        print("   ", f)
    ckpt = ROOT / "runs/policy_value_net_42k.pt"
    print(f"\nnet: {ckpt.name} (the supervised base), both sides, "
          f"deterministic bestmove")
    print(f"  {'nodes':>6}  {'pawn wins':>9} {'draws':>6} {'losses':>7} "
          f"{'unfin':>6} {'mean plies':>10}   conversion")
    for sims in budgets:
        r = play_wave(ckpt, fens, sims, "mps")
        n = len(fens)
        conv = r["pawn_wins"] / n
        print(f"  {sims:>6}  {r['pawn_wins']:>9} {r['draws']:>6} "
              f"{r['pawn_losses']:>7} {r['unfinished']:>6} "
              f"{r['mean_plies']:>10.1f}   {100 * conv:5.1f}%")
    print("\nReading of the result:")
    print("  conversion ~100% at the low budget -> task too easy, cannot show learning")
    print("  conversion low at BOTH budgets    -> task out of reach for this net")
    print("  low at 64 and high at 600         -> useful dynamic range: the")
    print("      loop (self-play -> train -> search) has something to move")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
