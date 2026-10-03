"""
chess_ai/selfplay/mixed.py — head-to-head match where the SAME weights play
with DIFFERENT search families per seat: one seat runs the C AZ-PUCT walk,
the other the Python reference Gumbel search (gumbel.SelfPlayGame). The
user's question (2026-10-03): PUCT games are draw-heavy (84-90%) while
Gumbel games are decisive — which search is actually stronger on one net?

Composition: the game lives in ONE gumbel.SelfPlayGame (board, rng,
records). The Gumbel seat moves through `_play_one_move`, preserving the
R5 seed-phase sampling rule (first `sample_plies` plies sampled ∝π′) and
the F4 handoff chain. The PUCT seat's moves are injected as FORCED moves —
a per-move CAzGame(fen) probe running az_eval.py's move rule (argmax
visits; inverse-CDF ∝ visits during the seed phase), pushed straight onto
the shared board. That is exactly gumbel.py's "after a forced move" case:
the played child was never created in the Gumbel tree, so handoff=None
ends the reuse chain and the next Gumbel search starts from a fresh root.

Seat config mirrors the two instruments this project measured all along:
Gumbel at the gate settings (c_visit 50, c_scale 1.0 — the σ-commitment
config), PUCT at the fleet settings (c_puct 1.25, child q-view). `n` is
shared so both seats get the same sim budget. Resignation off, ply cap
adjudicates draws — the gate protocol. Each seat's RNG is deterministic
in (match_seed, seed); no cross-harness stream parity is claimed (the
seats differ, so the ladder's exact-0.5 symmetry identity can't apply).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from chess_ai.engine import Termination
from chess_ai.selfplay.azsearch import CAzGame, choose_move_visits
from chess_ai.selfplay.gumbel import SelfPlayConfig, SelfPlayGame

_MIXED_RNG_TAG = 0x5EEDC0DE   # de-correlates the az seat's stream from
                              # other harnesses keyed on (match_seed, seed)


@dataclass
class MixedConfig:
    """Shared match protocol + each search family's knobs. `n` is the sim
    budget BOTH seats get per move."""
    seeds: int = 50                 # per color pairing → 2 × seeds games
    match_seed: int = 0
    workers: int = 32               # games in flight (thread pool size)
    batch_size: int = 64
    window_ms: float = 5.0
    sample_plies: int = 10          # seed phase, both seats
    max_plies: Optional[int] = 128  # adjudicated draws (gate protocol)
    # gumbel seat (SelfPlayConfig passthrough — the gate instrument)
    n: int = 128
    m_cap: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    c_fpu: float = 0.0
    resign_value: float = -2.0      # disabled: games play out (R5)
    # az seat (the fleet-measured instrument)
    c_puct: float = 1.25
    q_view: str = "child"


def _play_one_mixed(slot: int, cfg: MixedConfig,
                    evaluators: dict, a_search: str, b_search: str) -> dict:
    """One mixed game. `evaluators` maps search name → service-side
    evaluator ("az" / "gumbel"). Seat A is the PUCT seat when
    a_search="az"; slots < seeds have A as white, the rest are the
    color-reversed partners on the same seed."""
    a_white = slot < cfg.seeds
    seed = slot % cfg.seeds
    white_search, black_search = ((a_search, b_search) if a_white
                                  else (b_search, a_search))

    sp_cfg = SelfPlayConfig(n=cfg.n, m_cap=cfg.m_cap, c_visit=cfg.c_visit,
                            c_scale=cfg.c_scale, c_fpu=cfg.c_fpu,
                            sample_plies=cfg.sample_plies,
                            resign_value=cfg.resign_value)
    sg = SelfPlayGame(sp_cfg, seed=cfg.match_seed, game_index=seed)
    az_rng = random.Random((((cfg.match_seed & 0xFFFFFFFF) << 32)
                            | (seed & 0xFFFFFFFF)) ^ _MIXED_RNG_TAG)
    board = sg.board
    handoff = None
    while board.termination == Termination.PLAYING and (
            cfg.max_plies is None or board.ply < cfg.max_plies):
        mover = board.side_to_move
        search = white_search if mover == 0 else black_search
        if search == "az":
            sub = CAzGame(board.fen)
            try:
                sub.think(evaluators["az"], cfg.n, cfg.c_puct,
                          q_parent_view=(cfg.q_view == "parent"))
                uci = choose_move_visits(sub.root_children(), az_rng,
                                         sample=(board.ply
                                                 < cfg.sample_plies))
            finally:
                sub.close()
            board.push(uci)
            handoff = None   # forced move — see module docstring
        else:
            handoff = sg._play_one_move(evaluators["gumbel"], handoff)

    if board.termination == Termination.CHECKMATE:
        winner, kind = 1 - board.side_to_move, "natural"
    elif board.termination != Termination.PLAYING:
        winner, kind = -1, "natural"
    else:
        winner, kind = -1, "adjudicated"
    z_white = 0 if winner < 0 else (1 if winner == 0 else -1)
    return {
        "slot": slot, "seed": seed, "a_white": a_white,
        "z_white": z_white, "outcome_kind": kind, "plies": board.ply,
        "a_score": 0.5 if z_white == 0
        else (1.0 if (z_white > 0) == a_white else 0.0),
    }


def play_mixed_match(evaluators: dict, cfg: MixedConfig,
                     a_name: str, b_name: str,
                     a_search: str = "az", b_search: str = "gumbel",
                     log=print) -> dict:
    """2 × seeds games, `workers` in flight. Returns the ladder-shaped
    summary (score_a = seat A's score) plus a draws-by-search split —
    the user's draw-rate observation, quantified per seat."""
    t0 = time.perf_counter()
    games = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = [pool.submit(_play_one_mixed, slot, cfg, evaluators,
                               a_search, b_search)
                   for slot in range(2 * cfg.seeds)]
        done = 0
        for fut in as_completed(futures):
            g = fut.result()
            games.append(g)
            done += 1
            running = sum(x["a_score"] for x in games) / max(1, done)
            res = ("1/2" if g["z_white"] == 0
                   else ("1-0" if g["z_white"] > 0 else "0-1"))
            log(f"  game {done:3d}/{2 * cfg.seeds}  seed {g['seed']:2d}  "
                f"A={'white' if g['a_white'] else 'black'}  "
                f"{res} ({g['outcome_kind']}, {g['plies']} plies)  "
                f"A-score {g['a_score']:.1f}  running {running:.3f}  "
                f"({time.perf_counter() - t0:.0f} s)")
    games.sort(key=lambda g: g["slot"])
    n = len(games)
    score_a = sum(g["a_score"] for g in games) / max(1, n)
    s = min(max(score_a, 0.5 / max(1, n)), 1.0 - 0.5 / max(1, n))
    draws = [g for g in games if g["z_white"] == 0]
    draws_by_search = {}
    for g in games:
        mover_search = (a_search if g["a_white"] else b_search,
                        b_search if g["a_white"] else a_search)
        key = f"{mover_search[0]}-vs-{mover_search[1]}"
        draws_by_search.setdefault(key, 0)
        draws_by_search[key] += g["z_white"] == 0
    half = cfg.seeds // 2
    mean = lambda gs: (sum(g["a_score"] for g in gs) / len(gs)
                       if gs else float("nan"))
    return {
        "a": a_name, "b": b_name,
        "search_a": a_search, "search_b": b_search,
        "n": cfg.n, "seeds": cfg.seeds, "games": n,
        "score_a": round(score_a, 4),
        "elo_a_minus_b": round(400.0 * math.log10(s / (1.0 - s)), 1),
        "draws": len(draws),
        "draws_by_search": draws_by_search,
        "split_halves": [round(mean([g for g in games
                                     if g["seed"] < half]), 4),
                         round(mean([g for g in games
                                     if g["seed"] >= half]), 4)],
        "match_seed": cfg.match_seed,
        "max_plies": cfg.max_plies,
        "c_puct": cfg.c_puct, "q_view": cfg.q_view,
        "c_visit": cfg.c_visit, "c_scale": cfg.c_scale,
        "wall_s": round(time.perf_counter() - t0),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m chess_ai.selfplay.mixed",
        description="Head-to-head: one checkpoint, two search families "
                    "(PUCT seat vs Gumbel seat).")
    p.add_argument("--a", required=True, help="seat A checkpoint ('fixture' ok)")
    p.add_argument("--b", required=True, help="seat B checkpoint ('fixture' ok)")
    p.add_argument("--a-search", choices=["az", "gumbel"], default="az")
    p.add_argument("--b-search", choices=["az", "gumbel"], default="gumbel")
    p.add_argument("--seeds", type=int, default=MixedConfig.seeds)
    p.add_argument("--n", type=int, default=MixedConfig.n)
    p.add_argument("--max-plies", type=int, default=MixedConfig.max_plies)
    p.add_argument("--sample-plies", type=int,
                   default=MixedConfig.sample_plies)
    p.add_argument("--c-puct", type=float, default=MixedConfig.c_puct)
    p.add_argument("--q-view", choices=["child", "parent"],
                   default=MixedConfig.q_view)
    p.add_argument("--c-visit", type=float, default=MixedConfig.c_visit)
    p.add_argument("--c-scale", type=float, default=MixedConfig.c_scale)
    p.add_argument("--match-seed", type=int, default=MixedConfig.match_seed)
    p.add_argument("--workers", type=int, default=MixedConfig.workers)
    p.add_argument("--device", default="cpu")
    p.add_argument("--jsonl", default=None)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    from chess_ai.selfplay.ladder import (BatchedInference, NetEvaluator,
                                          _ServiceSide, _load_side)

    cfg = MixedConfig(seeds=a.seeds, match_seed=a.match_seed,
                      workers=a.workers, sample_plies=a.sample_plies,
                      max_plies=a.max_plies, n=a.n, c_puct=a.c_puct,
                      q_view=a.q_view, c_visit=a.c_visit, c_scale=a.c_scale)
    same = a.a == a.b
    net_a = _load_side(a.a, a.device)
    net_b = net_a if same else _load_side(a.b, a.device)
    nets = [net_a] if same else [net_a, net_b]
    id_a = 0
    id_b = 0 if same else 1
    service = BatchedInference(nets, batch_size=cfg.batch_size,
                               window_ms=cfg.window_ms)
    if a.a_search == a.b_search:
        raise SystemExit("--a-search and --b-search must differ (this "
                         "harness exists to compare search families)")
    evaluators = {a.a_search: _ServiceSide(service, id_a),
                  a.b_search: _ServiceSide(service, id_b)}
    log = (lambda *_: None) if a.quiet else (lambda m: print(m, flush=True))
    log(f"mixed[{a.a_search} vs {a.b_search}]: {a.a} vs {a.b}  "
        f"(n={cfg.n}, {2 * cfg.seeds} games, seed phase "
        f"{cfg.sample_plies} plies, resignation off, ply cap "
        f"{cfg.max_plies})")
    summary = play_mixed_match(evaluators, cfg,
                               a_name=Path(a.a).stem + f"({a.a_search})",
                               b_name=Path(a.b).stem + f"({a.b_search})",
                               a_search=a.a_search, b_search=a.b_search,
                               log=log)
    log(json.dumps(summary))
    if a.jsonl:
        out = Path(a.jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc)
                                .isoformat(timespec="seconds"),
                                **summary}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
