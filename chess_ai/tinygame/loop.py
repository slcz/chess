"""chess_ai/tinygame/loop.py — the learn loop + the perfect-play gate
(Phase 3.5's deliverable).

One iteration = the Phase 3 cycle at tiny scale:

    1. SELF-PLAY   N games with the CURRENT net (NumpyEvaluator synced
                   first — the C engine's "evaluator holds the weights"
                   seam), sampled plies, Gumbel n = 8
    2. TRAIN       train_steps minibatches sampled uniformly-with-
                   replacement from the rolling window
    3. EVAL        the deployed player vs the exhaustive oracle

THE TRAINER is trainer.py's `train_step` with the chess trunk swapped for
TinyNet — the loss MATH is copied line for line from the dense-π′ branch
(target-weighted policy CE over the π′ support; WDL cross-entropy vs z,
mover's view; aux opponent-reply CE with weight 0.3 and aux_w-gated
denominator; Adam 3e-4, grad clip 1.0 — TrainConfig defaults).

THE GATE (what "perfect play" means here, exactly): over ALL reachable
positions the deployed player — net + deterministic Gumbel search
(sample_plies=0: the sequential-halving survivor; m = k since eval_n ≥ 9
so candidate subsetting is off, the plan's "Gumbels-off eval variant";
per-position seeded RNG so the measurement is reproducible) — must choose
a VALUE-PRESERVING action in 100% of them:

    plays move a  →  value(child, opponent view) == −value(position)
    resigns       →  value(position) == −1 (resigning a lost position
                     preserves the outcome; resigning anything else is a
                     blunder and is counted)

Diagnostics reported alongside: policy-only perfect % (masked-logit
argmax, no search — how much the NET itself knows) and value-head exact
accuracy (argmax WDL == oracle value class over all reachable positions).

CLI: python -m chess_ai.tinygame.loop [--iters ...] [--out DIR]
"""
from __future__ import annotations

import json
import random
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from chess_ai.tinygame import game as ttt
from chess_ai.tinygame.data import TinyReplayBuffer
from chess_ai.tinygame.net import NumpyEvaluator, TinyNet
from chess_ai.tinygame.search import SelfPlayConfig, SelfPlayGame

_TORCH_DTYPES = {
    "tokens": torch.float32, "mask": torch.bool, "pol_idx": torch.int64,
    "wdl": torch.float32, "aux_idx": torch.int64, "aux_mask": torch.bool,
    "aux_w": torch.float32, "pi_slots": torch.int64, "pi_probs": torch.float32,
}


@dataclass
class SmokeConfig:
    # self-play (SelfPlayConfig defaults, restated for one place)
    n: int = 16                         # Gumbel sims/move while training
    resign_value: float = -0.90
    # σ's value-dominance constant DURING SELF-PLAY (the gate/eval search
    # keeps the paper's 50 — see eval_c_visit). The 3.5 bring-up finding
    # that motivated exposing it: the outcome label at a drawn position is
    # one-sided (the attacker wins or draws, never loses), so the value
    # head learns outcome-under-current-play and stays pessimistic about
    # the defender exactly where the policy is right; with c_visit=50 the
    # σ-term overrides the (correct) policy prior at those traps and the
    # poisoned q's reproduce themselves. Prior-led self-play (small
    # c_visit) lets the defender hold draws, cleaning the labels — the
    # loop then unwinds its own bias. Chess run-1 A/B candidate.
    c_visit: float = 5.0
    eval_c_visit: float = 5.0           # the gate player uses the measured
                                        # game value too (see the c_visit note);
                                        # at 50 the same nets score ~99.7-99.9
                                        # — σ's amplification of the residual
                                        # value noise — the finding that makes
                                        # c_visit a per-game knob
    # None = sample the played move ∝ π′ at every ply (chess training
    # default, 2026-09-27 amendment); 0 = the deterministic survivor every
    # ply — the openingSampleK knob's extreme setting, which the plan's
    # KataGo table reserves for THIS smoke test to measure. Bring-up found
    # π′-sampled defenders blunder away drawn positions, poisoning their z
    # labels (the value head learns "lost" where perfect play draws); the
    # survivor defends at the net's best belief, and the diverse-init
    # starts keep the game variety sampling would have provided.
    sample_plies: Optional[int] = None
    sample_mode: str = "visits"         # the bring-up headline: ∝π′ sampling
                                        # self-consistently poisoned the last
                                        # trap positions (σ amplifies the value
                                        # head's errors into the targets and
                                        # the played moves); ∝N sampling is
                                        # flat in the value signal — the
                                        # correct defense keeps ~1/k play
                                        # probability, clean outcomes unwind
                                        # the poison, gate met. See search.py.
    games_per_iter: int = 384
    # training (TrainConfig defaults where they exist)
    train_steps: int = 120
    batch_size: int = 256
    lr: float = 3e-4
    lr_decay: float = 0.99             # per-iteration anneal (1.0 = off) —
                                       # sharpens the endgame of training;
                                       # brought in for the 100% tail, where
                                       # near-tie q's need a crisp value head
    aux_reply_weight: float = 0.3
    value_weight: float = 3.0          # TrainConfig's knob; raised from 1.0
                                       # after the bring-up grids: the gate's
                                       # residual misses were deep-leaf value
                                       # errors that σ amplifies into flips
    grad_clip: float = 1.0
    # loop
    max_iters: int = 120
    buffer_games: int = 45000           # keep the whole run's games: label
                                        # quality only improves as play
                                        # strengthens, and the early data stays
                                        # useful coverage
    eval_every: int = 2                 # light eval cadence
    eval_sample: int = 256              # light eval: fixed critical subset
    eval_n: int = 128                   # light-eval search sims (plan's 100–200)
    full_eval_n: int = 512              # gate eval sims — more root visits
                                        # per candidate sharpens the near-tie
                                        # q's that σ amplifies (brought in when
                                        # the last ~0.3% were measured to be
                                        # value-noise flips, not policy gaps)
    # diverse game initialization (KataGo's side-position init, adapted):
    # each self-play game starts after a uniformly-random legal prefix of
    # 0..random_start_plies plies. Found during 3.5 bring-up: startpos-only
    # self-play self-confirms on the last ~2% of positions — the policy
    # avoids its blind spots, so their z labels never appear, so the deep
    # values stay wrong and keep π′ wrong there. Random prefixes give every
    # reachable position root-level π′ + outcome supervision. 0 = off
    # (chess run 1's default: startpos-only; adopting it there is a
    # separate user decision).
    random_start_plies: int = 8
    seed: int = 0
    hidden: int = 256
    out_dir: Optional[str] = None


class TinyTrainer:
    """model + optimizer + step count; train_step mirrors
    Trainer.train_step's dense-π′ branch (device fixed to CPU: an 18-dim
    MLP is dispatch-bound — MPS/CUDA would only add launch latency and
    CI flakiness to a minutes-scale smoke)."""

    def __init__(self, model: TinyNet, cfg: SmokeConfig):
        self.cfg = cfg
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.steps = 0
        torch.manual_seed(cfg.seed)

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        b = {k: torch.from_numpy(np.ascontiguousarray(v)).to(_TORCH_DTYPES[k])
             for k, v in batch.items()}
        out = self.model(b["tokens"], b["mask"], b["aux_mask"])

        # Dense π′ target: target-weighted CE over the support (padding
        # repeats the first real slot with prob 0 — never a blind slot 0,
        # which could be illegal where policy_logp is −inf: −inf·0 = NaN).
        gathered = out["policy_logp"].gather(1, b["pi_slots"])
        policy_loss = -(gathered * b["pi_probs"]).sum(-1).mean()

        wdl_logp = F.log_softmax(out["wdl"], dim=-1)
        value_loss = -(wdl_logp * b["wdl"]).sum(-1).mean()

        aux_nll = -out["aux_logp"].gather(1, b["aux_idx"][:, None]).squeeze(1)
        w = b["aux_w"]
        aux_loss = (aux_nll * w).sum() / w.sum().clamp(min=1.0)

        loss = (policy_loss + self.cfg.value_weight * value_loss
                + self.cfg.aux_reply_weight * aux_loss)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.steps += 1

        with torch.no_grad():
            scalars = {
                "loss": loss.detach(), "policy": policy_loss.detach(),
                "value": value_loss.detach(), "aux": aux_loss.detach(),
                "policy_acc": (out["policy_logp"].argmax(-1) == b["pol_idx"]).float().mean(),
                "value_acc": (out["wdl"].argmax(-1) == b["wdl"].argmax(-1)).float().mean(),
            }
            values = torch.stack(list(scalars.values())).tolist()
        return dict(zip(scalars.keys(), values))


# ---------------------------------------------------------------------------
# Perfect-play evaluation vs the oracle
# ---------------------------------------------------------------------------

_ALL_POSITIONS: Optional[List[Tuple[int, ...]]] = None
_CRITICAL: Optional[List[Tuple[int, ...]]] = None


def all_positions() -> List[Tuple[int, ...]]:
    """Every DECISION position (non-terminal, reachable): the gate's domain.
    reachable_positions() is the canonical 5478 including terminal boards —
    no decision exists in those, so they are not evaluated."""
    global _ALL_POSITIONS
    if _ALL_POSITIONS is None:
        _ALL_POSITIONS = [c for c in ttt.reachable_positions()
                          if ttt.TicTacToe(c).termination == ttt.PLAYING]
    return _ALL_POSITIONS


def critical_positions() -> List[Tuple[int, ...]]:
    global _CRITICAL
    if _CRITICAL is None:
        _CRITICAL = [c for c in all_positions() if ttt.is_critical(c)]
    return _CRITICAL

def _pos_seed(cells: Tuple[int, ...]) -> int:
    """Stable per-position seed (zlib.crc32 — hash() is process-salted and
    would make the gate number irreproducible)."""
    return zlib.crc32(repr(cells).encode())


def _move_value(cells: Tuple[int, ...], move: int) -> int:
    """Value of playing `move` from the CURRENT mover's view (terminal
    children resolved directly; negamax otherwise)."""
    board = ttt.TicTacToe(cells)
    board.push(move)
    if board.termination == ttt.WIN:
        return 1
    if board.termination == ttt.DRAW:
        return 0
    return -ttt.value(board.cells)


def search_action(cells: Tuple[int, ...], evaluator: NumpyEvaluator,
                  eval_n: int, resign_value: float,
                  c_visit: float = 50.0) -> Tuple[str, int]:
    """The deployed player's action at `cells`: ("move", cell) or
    ("resign", −1). Deterministic Gumbel eval — the plan's Gumbels-off
    variant (m = k, z = logp: the survivor is argmax of π′'s score, no
    Gumbel randomness in strength measurement; per-position seeding keeps
    the whole evaluation reproducible)."""
    cfg = SelfPlayConfig(n=eval_n, m_cap=16, sample_plies=0,
                         resign_value=resign_value, gumbels_off=True,
                         c_visit=c_visit)
    game = SelfPlayGame(cfg, seed=_pos_seed(cells), game_index=0,
                        start_cells=cells)
    decision, _root = game._search_move(evaluator)
    if decision.outlook <= resign_value:
        return "resign", -1
    return "move", decision.survivor_slot


def eval_positions(evaluator: NumpyEvaluator, positions: Iterable[Tuple[int, ...]],
                   eval_n: int, resign_value: float,
                   c_visit: float = 50.0) -> Dict[str, float]:
    """Perfect-play stats over `positions` (see module docstring for the
    gate definition)."""
    n = crit = perfect = crit_perfect = resign_blunders = 0
    policy_perfect = 0
    value_acc = 0
    for cells in positions:
        val = ttt.value(cells)
        n += 1
        is_crit = ttt.is_critical(cells)
        crit += is_crit

        # deployed player: search + net
        kind, act = search_action(cells, evaluator, eval_n, resign_value,
                                  c_visit)
        if kind == "resign":
            ok = val == -1                    # resigning only a lost position
            resign_blunders += val != -1
        else:
            ok = _move_value(cells, act) == val
        perfect += ok
        crit_perfect += is_crit and ok

        # diagnostics: raw policy argmax over legal slots, no search
        # (max over the ascending legal tuple tie-breaks to the first max)
        pol, wdl, _ = evaluator._forward(ttt.TicTacToe(cells).encode())
        legal = ttt.TicTacToe(cells).legal_moves
        best = max(legal, key=lambda s: pol[s])
        policy_perfect += _move_value(cells, best) == val
        value_acc += int(np.argmax(wdl)) == (0 if val > 0 else (1 if val == 0 else 2))

    return {
        "positions": float(n),
        "perfect_pct": 100.0 * perfect / n,
        "critical_positions": float(crit),
        "critical_perfect_pct": 100.0 * crit_perfect / max(crit, 1),
        "resign_blunders": float(resign_blunders),
        "policy_perfect_pct": 100.0 * policy_perfect / n,
        "value_acc_pct": 100.0 * value_acc / n,
    }


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class SmokeResult:
    gate: bool
    iterations: int
    games: int
    positions: int
    final: Dict[str, float]
    wall_seconds: float
    gen_seconds: float = 0.0
    train_seconds: float = 0.0
    eval_seconds: float = 0.0
    history: List[Dict] = field(default_factory=list)


def random_start(max_prefix: int, rng: random.Random) -> Tuple[int, ...]:
    """A position reached by a uniformly-random legal prefix of length
    0..max_prefix — the diverse-init source (see SmokeConfig). Always a
    decision position: the walk stops before any terminal (a random prefix
    can complete a line; that push is popped rather than kept). Data-
    collection RNG only; the search's stream is untouched by this."""
    board = ttt.TicTacToe()
    for _ in range(rng.randint(0, max_prefix)):
        moves = board.legal_moves
        if not moves:
            break
        board.push(rng.choice(moves))
        if board.termination != ttt.PLAYING:
            board.pop()
            break
    return board.cells


def run_smoke(cfg: SmokeConfig, log=print) -> SmokeResult:
    """Generate → train → eval, until the full-oracle gate passes or
    max_iters. Early-stop protocol: a clean LIGHT eval (fixed critical
    sample) triggers the FULL 5478-position eval; only a clean full eval
    stops the loop (light is a cheap proxy, full is the gate)."""
    t0 = time.perf_counter()
    net = TinyNet(cfg.hidden)
    trainer = TinyTrainer(net, cfg)
    evaluator = NumpyEvaluator(net)
    buffer = TinyReplayBuffer(cfg.buffer_games)
    sp_cfg = SelfPlayConfig(n=cfg.n, resign_value=cfg.resign_value,
                            sample_plies=cfg.sample_plies,
                            sample_mode=cfg.sample_mode,
                            c_visit=cfg.c_visit)
    rng = random.Random(cfg.seed)
    light_set = sorted(rng.sample(critical_positions(),
                                  min(cfg.eval_sample, len(critical_positions()))))

    gen_s = train_s = eval_s = 0.0
    games = 0
    history: List[Dict] = []
    final: Dict[str, float] = {}
    gate = False

    for it in range(1, cfg.max_iters + 1):
        # anneal the LR each iteration (no-op at lr_decay = 1.0)
        lr = cfg.lr * (cfg.lr_decay ** (it - 1))
        for group in trainer.opt.param_groups:
            group["lr"] = lr

        # 1. self-play on the current weights (diverse-init starts; the
        # per-game start RNG is derived from the game's identity)
        tg = time.perf_counter()
        evaluator.sync_from(net)
        records = []
        for g in range(cfg.games_per_iter):
            start = (random_start(cfg.random_start_plies, rng)
                     if cfg.random_start_plies else None)
            records.append(SelfPlayGame(
                sp_cfg, seed=cfg.seed * 100_000 + it, game_index=g,
                start_cells=start).play(evaluator))
        buffer.add_all(records)
        games += len(records)
        gen_s += time.perf_counter() - tg

        # 2. train on the window
        tt = time.perf_counter()
        agg: Dict[str, float] = {}
        for _ in range(cfg.train_steps):
            batch = buffer.sample_tensors(rng, cfg.batch_size)
            for k, v in trainer.train_step(batch).items():
                agg[k] = agg.get(k, 0.0) + v
        train_s += time.perf_counter() - tt
        metrics = {k: v / cfg.train_steps for k, v in agg.items()}
        metrics.update(iteration=it, games=games,
                       buffer_positions=buffer.total_positions)

        # 3. eval (light on cadence; full when light is clean / at the end)
        if it % cfg.eval_every == 0 or it == cfg.max_iters:
            te = time.perf_counter()
            light = eval_positions(evaluator, light_set, cfg.eval_n,
                                   cfg.resign_value, cfg.eval_c_visit)
            eval_s += time.perf_counter() - te
            metrics.update({f"light_{k}": v for k, v in light.items()})
            log(f"it {it:3d}  games {games:5d}  policy {metrics['policy']:.4f} "
                f"value {metrics['value']:.4f}  light-perfect "
                f"{light['critical_perfect_pct']:.1f}%  policy-only "
                f"{light['policy_perfect_pct']:.1f}%  value-acc "
                f"{light['value_acc_pct']:.1f}%")
            if light["critical_perfect_pct"] == 100.0 and light["resign_blunders"] == 0:
                te = time.perf_counter()
                final = eval_positions(evaluator, all_positions(), cfg.full_eval_n,
                                       cfg.resign_value, cfg.eval_c_visit)
                eval_s += time.perf_counter() - te
                gate = (final["perfect_pct"] == 100.0
                        and final["resign_blunders"] == 0)
                metrics.update({f"full_{k}": v for k, v in final.items()})
                log(f"     FULL eval: perfect {final['perfect_pct']:.2f}% "
                    f"(critical {final['critical_perfect_pct']:.2f}% of "
                    f"{int(final['critical_positions'])})  "
                    f"resign-blunders {int(final['resign_blunders'])}  "
                    f"value-acc {final['value_acc_pct']:.2f}%  "
                    f"-> {'GATE MET' if gate else 'gate not met'}")
                if gate:
                    history.append(metrics)
                    break
        history.append(metrics)

    if not final:                                  # never triggered light-clean
        te = time.perf_counter()
        final = eval_positions(evaluator, all_positions(), cfg.full_eval_n,
                               cfg.resign_value, cfg.eval_c_visit)
        eval_s += time.perf_counter() - te
        gate = (final["perfect_pct"] == 100.0 and final["resign_blunders"] == 0)
        log(f"     FULL eval (max_iters): perfect {final['perfect_pct']:.2f}%  "
            f"-> {'GATE MET' if gate else 'GATE NOT MET'}")

    wall = time.perf_counter() - t0
    result = SmokeResult(gate=gate, iterations=len(history), games=games,
                         positions=buffer.total_positions, final=final,
                         wall_seconds=wall, gen_seconds=gen_s,
                         train_seconds=train_s, eval_seconds=eval_s,
                         history=history)

    if cfg.out_dir:
        out = Path(cfg.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": net.state_dict(),
                    "config": asdict(cfg)}, out / "net.pt")
        (out / "metrics.json").write_text(json.dumps({
            "gate": result.gate, "final": result.final,
            "wall_seconds": result.wall_seconds,
            "gen_seconds": result.gen_seconds,
            "train_seconds": result.train_seconds,
            "eval_seconds": result.eval_seconds,
            "games": result.games, "iterations": result.iterations,
        }, indent=2))
        log(f"artifacts -> {out}")
    return result


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=SmokeConfig.max_iters)
    ap.add_argument("--games-per-iter", type=int, default=SmokeConfig.games_per_iter)
    ap.add_argument("--steps", type=int, default=SmokeConfig.train_steps)
    ap.add_argument("--n", type=int, default=SmokeConfig.n,
                    help="self-play Gumbel sims per move")
    ap.add_argument("--seed", type=int, default=SmokeConfig.seed)
    ap.add_argument("--out", default="runs/tinygame/smoke")
    args = ap.parse_args()
    cfg = SmokeConfig(max_iters=args.iters, games_per_iter=args.games_per_iter,
                      train_steps=args.steps, n=args.n, seed=args.seed,
                      out_dir=args.out)
    result = run_smoke(cfg)
    raise SystemExit(0 if result.gate else 1)


if __name__ == "__main__":
    main()
