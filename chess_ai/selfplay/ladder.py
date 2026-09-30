"""
chess_ai/selfplay/ladder.py — the checkpoint ladder runner (PARALLEL_DESIGN
R5): net@A vs net@B at the agreed eval-match protocol, the pipeline-health
signal for Phases 3→6 ("monotone checkpoint ladder").

Protocol (PLAN §3, R5):

- The same Full Gumbel search as self-play, at eval strength (n = 100–200
  sims), resignation DISABLED — a ladder game must play out to a natural
  result, or a miscalibrated value head would record its own bias as the
  score (the Step-1 finding).
- Opening diversity WITHOUT human book or random prefixes: a sampled seed
  phase (first `sample_plies` plies sampled ∝ π′, per-game RNG), then the
  deterministic measurement phase (the sequential-halving survivor —
  `sample_plies ≥ 0` in SpConfig).
- Every seed is played TWICE with colors reversed: game s has net A as
  white, game s + seeds has net B as white, both from the same (match_seed,
  seed) RNG pair — so each seed contributes a paired result that cancels
  first-move advantage, and a seat-symmetric pair of identical nets scores
  exactly 0.5.
- `seeds` × 2 games per ladder point, with per-seed outcome matrices and
  split-half variance monitoring; Elo is reported with the usual caveat
  (net-vs-net at fixed settings, small N — a strength DIFFERENCE signal,
  not an absolute rating).

Mechanics: games run through the sequential C driver (csearch.CSpGame —
one mmap slot per game, so `workers` games play concurrently in Python
threads; the two torch models are shared and inference-only). The two
nets never see each other's weights — the SeatEvaluator routes each
evaluation request to the side-to-move's net.

CLI (run on the node, alongside the pilot):

    python -m chess_ai.selfplay.ladder \
        --a runs/chess_pilot/net-iter-0020.pt \
        --b runs/policy_value_net_rope_25k.pt \
        --seeds 32 --device cuda --jsonl runs/eval/ladder.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from chess_ai.selfplay.csearch import CSpGame
from chess_ai.selfplay.gumbel import Evaluator, SelfPlayConfig


@dataclass
class LadderConfig:
    """Search knobs = the eval-strength settings (R5: 100–200 sims, same
    Gumbel selector everywhere); match knobs = the R5 protocol defaults."""

    # search (the shared selector; see PLAN §3 search design)
    n: int = 160                    # sims per move (eval 100–200)
    m_cap: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    c_fpu: float = 0.0
    sample_plies: int = 10          # seed phase ∝π′, then the survivor
    resign_value: float = -2.0      # DISABLED: ladder games play out (R5)
    max_plies: Optional[int] = None # natural termination; knob for tests
    # match protocol
    seeds: int = 64                 # per color pairing → 2 × seeds games
    match_seed: int = 0             # every game's RNG derives from this
    workers: int = 6                # games in flight (thread pool size)


class NetEvaluator(Evaluator):
    """One checkpoint as a sequential-driver evaluator. Loads the net from
    the checkpoint's saved net_config (same arch-safety rules and legacy
    "shaw" pin as Trainer.load / the batch evaluator)."""

    def __init__(self, ckpt: str, device: str = "cpu"):
        import torch

        from chess_ai.net.model import ChessTransformer, NetConfig

        self._torch = torch
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        raw = dict(state.get("net_config") or {})
        if raw:
            raw.setdefault("pos_scheme", "shaw")
            model = ChessTransformer(NetConfig(**raw))
        else:
            model = ChessTransformer()
        model.load_state_dict(state.get("model", state))
        self.model = model.to(device=device, dtype=torch.float32).eval()
        self._device = next(self.model.parameters()).device
        self.ckpt = str(ckpt)

    def evaluate(self, planes, legal_slots, mover):
        torch = self._torch
        x = torch.from_numpy(np.asarray(planes, dtype=np.float32)
                             .reshape(1, 64, 117)).to(self._device)
        mask = torch.zeros(1, 4240, dtype=torch.bool)
        mask[0, list(legal_slots)] = True
        mask = mask.to(self._device)
        with torch.no_grad():
            out = self.model(x, mask)
        logp = out["policy_logp"][0].cpu()
        lp = {s: float(logp[s]) for s in legal_slots}
        w = torch.softmax(out["wdl"][0], dim=-1).cpu()
        wdl = (float(w[0]), float(w[1]), float(w[2]))
        return lp, wdl


class SeatEvaluator(Evaluator):
    """Routes each evaluation to the side-to-move's net — how two distinct
    checkpoints meet inside one C game."""

    def __init__(self, white: Evaluator, black: Evaluator):
        self.white = white
        self.black = black

    def evaluate(self, planes, legal_slots, mover):
        return (self.white if mover == 0 else self.black).evaluate(
            planes, legal_slots, mover)


@dataclass
class MatchResult:
    a_name: str
    b_name: str
    cfg: LadderConfig
    games: list = field(default_factory=list)   # one dict per game

    @property
    def n_games(self) -> int:
        return len(self.games)

    @property
    def score_a(self) -> float:
        return sum(g["a_score"] for g in self.games) / max(1, self.n_games)

    @property
    def draws(self) -> int:
        return sum(1 for g in self.games if g["z_white"] == 0)

    @property
    def elo_a_minus_b(self) -> float:
        """−400·log10(1/s − 1) for net A, clamped so 0/1 scores stay finite.
        Net-vs-net at fixed settings with small N — a difference signal."""
        s = min(max(self.score_a, 0.5 / max(1, self.n_games)),
                1.0 - 0.5 / max(1, self.n_games))
        return 400.0 * math.log10(s / (1.0 - s))

    def split_halves(self) -> tuple[float, float]:
        """Mean A-score over the first vs second half of the SEEDS (both
        color pairings included) — R5's split-half variance monitor."""
        half = self.cfg.seeds // 2
        first = [g for g in self.games if g["seed"] < half]
        second = [g for g in self.games if g["seed"] >= half]
        mean = lambda gs: (sum(g["a_score"] for g in gs) / len(gs)
                           if gs else float("nan"))
        return mean(first), mean(second)

    def seed_pairs(self) -> dict[int, tuple[float, float]]:
        """seed → (A's score as white, A's score as black): the per-seed
        outcome matrix; pair sums of 2/1/0 = won both / split / lost both."""
        pairs: dict[int, list[float]] = {}
        for g in self.games:
            entry = pairs.setdefault(g["seed"], [None, None])  # type: ignore
            entry[0 if g["a_white"] else 1] = g["a_score"]
        return {s: (v[0], v[1]) for s, v in pairs.items()}     # type: ignore

    def summarize(self) -> dict:
        first, second = self.split_halves()
        pairs = self.seed_pairs()
        sums = [round(a + b, 9) for a, b in pairs.values()]
        return {
            "a": self.a_name, "b": self.b_name,
            "n": self.cfg.n, "seeds": self.cfg.seeds,
            "games": self.n_games,
            "score_a": round(self.score_a, 4),
            "elo_a_minus_b": round(self.elo_a_minus_b, 1),
            "draws": self.draws,
            "split_halves": [round(first, 4), round(second, 4)],
            "won_both": sums.count(2.0), "split": sums.count(1.0),
            "lost_both": sums.count(0.0),
            "match_seed": self.cfg.match_seed,
        }


def _play_one(slot: int, cfg: LadderConfig, a: Evaluator, b: Evaluator) -> dict:
    """One ladder game: seed = slot % seeds; the first `seeds` slots have A
    as white, the rest are the color-reversed partners with the SAME
    (match_seed, seed) RNG pair."""
    a_white = slot < cfg.seeds
    seed = slot % cfg.seeds
    white, black = (a, b) if a_white else (b, a)
    sp_cfg = SelfPlayConfig(n=cfg.n, m_cap=cfg.m_cap, c_visit=cfg.c_visit,
                            c_scale=cfg.c_scale, c_fpu=cfg.c_fpu,
                            sample_plies=cfg.sample_plies,
                            resign_value=cfg.resign_value)
    game = CSpGame(sp_cfg, seed=cfg.match_seed, game_index=seed)
    try:
        rec = game.play(SeatEvaluator(white, black), max_plies=cfg.max_plies)
    finally:
        game.close()
    z = rec.z_white
    return {
        "slot": slot, "seed": seed, "a_white": a_white,
        "z_white": z, "outcome_kind": rec.outcome_kind,
        "plies": len(rec.moves),
        "a_score": 0.5 if z == 0 else (1.0 if (z > 0) == a_white else 0.0),
    }


def play_match(a: Evaluator, b: Evaluator, cfg: LadderConfig,
               a_name: str = "A", b_name: str = "B",
               log=print) -> MatchResult:
    """2 × seeds games, `workers` in flight. Per-game results are
    deterministic in (match_seed, seed, evaluator outputs); thread
    scheduling cannot reorder them into the records."""
    result = MatchResult(a_name=a_name, b_name=b_name, cfg=cfg)
    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = [pool.submit(_play_one, slot, cfg, a, b)
                   for slot in range(2 * cfg.seeds)]
        for fut in as_completed(futures):
            result.games.append(fut.result())
            done += 1
            if done % max(1, cfg.seeds // 4) == 0 or done == 2 * cfg.seeds:
                partial = MatchResult(a_name=a_name, b_name=b_name,
                                      cfg=cfg, games=list(result.games))
                log(f"  {done:3d}/{2 * cfg.seeds} games  "
                    f"score {partial.score_a:.3f}  "
                    f"({time.perf_counter() - t0:.0f} s)")
    result.games.sort(key=lambda g: g["slot"])
    return result


def _load_side(spec: str, device: str) -> Evaluator:
    """--a/--b take a checkpoint path, or 'fixture' (uniform priors + WDL)
    for smoke tests. Note uniform-prior fixtures make both seats play
    identically — they exercise the machinery, never strength."""
    if spec == "fixture":
        from chess_ai.selfplay.gumbel import TableEvaluator
        return TableEvaluator()
    return NetEvaluator(spec, device=device)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m chess_ai.selfplay.ladder",
        description="Checkpoint ladder point: net A vs net B at the R5 "
                    "eval-match protocol.")
    p.add_argument("--a", required=True, help="checkpoint A (or 'fixture')")
    p.add_argument("--b", required=True, help="checkpoint B (or 'fixture')")
    p.add_argument("--seeds", type=int, default=LadderConfig.seeds,
                   help="seeds per color pairing (games = 2 × seeds)")
    p.add_argument("--n", type=int, default=LadderConfig.n)
    p.add_argument("--match-seed", type=int, default=LadderConfig.match_seed)
    p.add_argument("--workers", type=int, default=LadderConfig.workers)
    p.add_argument("--device", default="cuda",
                   help="torch device for both nets (cuda on the node)")
    p.add_argument("--max-plies", type=int, default=None)
    p.add_argument("--jsonl", default=None,
                   help="append the summary line here (runs/eval/…jsonl)")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    cfg = LadderConfig(seeds=a.seeds, n=a.n, match_seed=a.match_seed,
                       workers=a.workers, max_plies=a.max_plies)
    net_a = _load_side(a.a, a.device)
    net_b = _load_side(a.b, a.device)
    log = (lambda *_: None) if a.quiet else print
    log(f"ladder: {a.a} vs {a.b}  (n={cfg.n}, {2 * cfg.seeds} games, "
        f"seed phase {cfg.sample_plies} plies, resignation off)")
    result = play_match(net_a, net_b, cfg,
                        a_name=Path(a.a).stem, b_name=Path(a.b).stem, log=log)
    summary = result.summarize()
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
