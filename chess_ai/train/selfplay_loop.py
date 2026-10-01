"""
chess_ai/train/selfplay_loop.py — the chess self-play training loop
(Phase 3→4: the tinygame learn loop, pointed at the real game).

One iteration = one round, the shape proven by tinygame/loop.py:

    1. SELF-PLAY  the C parallel engine (SpRun) plays games_per_iter games
                  with the CURRENT net (passed to the evaluator as a
                  checkpoint file — the same handoff the cloud worker will
                  use), from a start-FEN pool of random legal prefixes
                  (3.5 finding 1: startpos-only self-play self-confirms).
    2. INGEST     records into the rolling SelfPlayBuffer.
    3. TRAIN      train_steps × train_step on batches sampled (with
                  replacement) from the buffer's whole window.
    4. CHECKPOINT atomic Trainer.save(out/latest.pt) every round.
    5. LOG        one log.jsonl line per round: train metrics, self-play
                  telemetry, wall-clock split.

RESUME (the §5 cloud rules, one round at a time): each round's games land
in out/units/iter-XXXX/shard-w*.spbin plus a DONE.json manifest the driver
writes in the chess_selfplay binary's schema — read_unit/ingest_units
consume both interchangeably. On restart, completed units are re-ingested
(dedup by (seed, game_index)) and training resumes from latest.pt; a unit
dir WITHOUT DONE.json is re-played (same unit seed ⇒ deterministic replay
into the O_TRUNC'd shards). A crash between a round's DONE.json and its
save re-trains that round from the pre-round checkpoint — the data gets a
double pass, which is harmless; the DONE manifest is what bounds replay at
"≤ one round lost".

Eval matches (the checkpoint ladder) are deliberately NOT here — that is
the Step-3 machinery (PARALLEL_DESIGN R5); this loop's health signals are
the train metrics plus self-play draw/resign rates and game lengths.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from chess_ai.engine import Board
from chess_ai.selfplay.gumbel import SelfPlayConfig
from chess_ai.selfplay.spengine import SpRun
from chess_ai.train.selfplay_data import SelfPlayBuffer, ingest_units, playable_plies
from chess_ai.train.trainer import Trainer, tensors_to_torch


@dataclass
class LoopConfig:
    """Defaults = the Mac pilot (Step 1): n=16, ~500 games, MPS evaluator.
    Cloud runs change workers/buffer/device, not the loop's shape."""

    # --- self-play -------------------------------------------------------
    n: int = 16                     # sims per move (self-play range 2–32)
    m_cap: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    c_fpu: float = 0.0
    resign_value: float = -0.90
    sample_plies: int = -1          # -1 = sample ∝ π′ every ply (training)
    max_plies: int = -1             # -1 = unbounded (termination rules end games)
    workers: int = 6                # ≤ SP_ENGINE_MAX_WORKERS (8)
    buffer_requests: int = 64       # epoch batch requests/worker (≤ 256)
    games_per_iter: int = 64
    start_plies: int = 8            # diverse init: prefix depth 1..start_plies
    start_pool_size: int = 32       # FENs per round; 0/0-depth = startpos only

    # --- training --------------------------------------------------------
    train_steps: int = 48
    batch_size: int = 256
    lr: float = 3e-4
    policy_weight: float = 1.0
    value_weight: float = 1.0
    aux_reply_weight: float = 0.3   # the seed ckpt's aux head is fresh (0.0 in
                                    # supervised) — run 1 turns the head on
    buffer_games: int = 45000       # rolling window (plan §3: 40–50k games)

    # --- loop ------------------------------------------------------------
    iters: int = 8
    save_every: int = 4             # round snapshots net-iter-XXXX.pt
    seed: int = 0
    seed_ckpt: str = "runs/policy_value_net_rope_25k.pt"
    out_dir: str = "runs/chess_pilot"
    device: str = "auto"            # trainer device (auto → MPS → CUDA → CPU)
    eval_device: str = ""           # "" = same as the trainer's resolved device
    hf_repo: str = ""               # "" = no cloud sync (Mac default). When set,
                                    # every round pushes latest.pt + unit shards
                                    # to this HF repo (§4.3), and a fresh box
                                    # pulls them back before resuming.
    push_every: int = 1             # push cadence in ROUNDS (1 commit/round;
                                    # HF caps commits at 128/h — back off here
                                    # if round time ever drops under ~30 s)
    eval_bf16: bool = False         # evaluator inference autocast (the net
                                    # trained under bf16; ~2x on the 3090)
    eval_compile: bool = False      # torch.compile(dynamic) the evaluator
                                    # (warmup stall once per process start)
    train_bf16: bool = False        # training autocast (TrainConfig.bf16)


def _derive(*parts: int) -> int:
    """Deterministic small-int derivation for per-round seeds (stable across
    processes — hash() of tuples is NOT, str hashing needs PYTHONHASHSEED)."""
    h = 0
    for p in parts:
        h = (h * 1_000_003 + p) & 0xFFFFFFFFFFFF
    return h


def build_start_pool(pool_seed: int, size: int, max_prefix: int) -> list[str]:
    """Random legal opening prefixes played through our own rules engine —
    the chess twin of tinygame's random start cells. Zero human knowledge:
    moves come from the engine's legal-move generator under a seeded RNG
    (random play is not a label, only a start). Deterministic per pool_seed.

    A random walk can reach mate/stalemate within the prefix window (seen
    live once per ~800 entries); terminal positions are resampled — a
    start position with no legal moves would produce a zero-ply game."""
    rng = random.Random(pool_seed)
    fens = []
    for _ in range(size):
        while True:
            board = Board()
            for _ in range(rng.randint(1, max_prefix)):
                moves = board.legal_moves
                if not moves:              # walked into mate/stalemate
                    break
                board.push(moves[rng.randrange(len(moves))])
            if board.legal_moves:          # non-terminal → a usable start
                fens.append(board.fen)
                break
    return fens


def _write_done(unit_dir: Path, manifest: dict) -> None:
    """The driver's twin of chess_selfplay_main.c's write_done: atomic
    tmp+rename, same schema — read_unit consumes both interchangeably."""
    tmp = unit_dir / "DONE.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp, unit_dir / "DONE.json")


def _apply_overrides(trainer: Trainer, cfg: LoopConfig) -> list[str]:
    """The checkpoint's TrainConfig is what the supervised run used; the
    loop's knobs take effect from here on (logged so the log file tells
    the whole story)."""
    applied = []
    t = trainer.cfg
    for name, val in (("lr", cfg.lr),
                      ("policy_weight", cfg.policy_weight),
                      ("value_weight", cfg.value_weight),
                      ("aux_reply_weight", cfg.aux_reply_weight),
                      ("batch_size", cfg.batch_size),
                      ("bf16", cfg.train_bf16)):
        if getattr(t, name) != val:
            setattr(t, name, val)
            applied.append(f"{name}={val}")
    if applied:                       # lr lives in the optimizer's groups too
        for group in trainer.opt.param_groups:
            group["lr"] = cfg.lr
    return applied


def run_round(trainer: Trainer, buffer: SelfPlayBuffer, cfg: LoopConfig,
              it: int, latest: Path) -> dict:
    """One iteration: self-play → ingest → train. Returns the metrics dict
    (also the log.jsonl line's content)."""
    out = Path(cfg.out_dir)
    unit_dir = out / "units" / f"iter-{it:04d}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_seed = _derive(cfg.seed, it)

    pool = (build_start_pool(_derive(cfg.seed, 7919, it),
                             cfg.start_pool_size, cfg.start_plies)
            if cfg.start_plies > 0 and cfg.start_pool_size > 0 else None)
    eval_spec = (f"torch:ckpt={latest}"
                 + (f",device={cfg.eval_device}" if cfg.eval_device else "")
                 + (",bf16=1" if cfg.eval_bf16 else "")
                 + (",compile=1" if cfg.eval_compile else ""))

    t0 = time.perf_counter()
    run = SpRun(
        SelfPlayConfig(n=cfg.n, m_cap=cfg.m_cap, c_visit=cfg.c_visit,
                       c_scale=cfg.c_scale, c_fpu=cfg.c_fpu,
                       sample_plies=(None if cfg.sample_plies < 0
                                     else cfg.sample_plies),
                       resign_value=cfg.resign_value),
        unit_seed=unit_seed, game_budget=cfg.games_per_iter,
        n_workers=cfg.workers, buffer_requests=cfg.buffer_requests,
        max_plies=(None if cfg.max_plies < 0 else cfg.max_plies),
        eval_spec=eval_spec, record_dir=unit_dir, start_fens=pool)
    records = run.play()
    tele = run.telemetry()
    run.close()
    sp_s = time.perf_counter() - t0

    _write_done(unit_dir, {
        "format": "SPSHARD1", "version": 1,
        "unit_seed": unit_seed, "game_budget": cfg.games_per_iter,
        "n_workers": cfg.workers, "games_done": len(records),
        "evals": tele["evals"], "epochs": tele["epochs"],
        "max_batch": tele["max_batch"], "elapsed_s": round(sp_s, 3),
        "eval_spec": eval_spec,
        "n": cfg.n, "m_cap": cfg.m_cap, "c_visit": cfg.c_visit,
        "c_scale": cfg.c_scale, "c_fpu": cfg.c_fpu,
        "sample_plies": cfg.sample_plies, "resign_value": cfg.resign_value,
        "max_plies": cfg.max_plies, "interleave": 0,
        "start_fens": len(pool) if pool else 0,
        "writer": "selfplay_loop",
    })
    buffer.add_all(records)

    t1 = time.perf_counter()
    sample_rng = random.Random(_derive(cfg.seed, it, trainer.steps))
    agg: dict[str, float] = {}
    for _ in range(cfg.train_steps):
        batch = tensors_to_torch(buffer.sample_tensors(sample_rng,
                                                       cfg.batch_size),
                                 trainer.device)
        for k, v in trainer.train_step(batch).items():
            agg[k] = agg.get(k, 0.0) + v
    train_s = time.perf_counter() - t1
    train = {k: v / cfg.train_steps for k, v in agg.items()}

    draws = sum(1 for r in records if r.z_white == 0)
    resigns = sum(1 for r in records if r.outcome_kind == "resign")
    plies = [playable_plies(r) for r in records]
    metrics = {
        "iter": it,
        "games_total": len(buffer),
        "buffer_positions": buffer.total_positions,
        "steps": trainer.steps,
        "train": train,
        "sp": {
            "games": len(records), "evals": tele["evals"],
            "epochs": tele["epochs"], "max_batch": tele["max_batch"],
            "draw_rate": draws / len(records),
            "resign_rate": resigns / len(records),
            "mean_plies": sum(plies) / len(plies),
            "start_pool": len(pool) if pool else 0,
            "sp_s": round(sp_s, 2),
            "games_per_hour": len(records) / sp_s * 3600 if sp_s > 0 else 0.0,
        },
        "train_s": round(train_s, 2),
    }
    return metrics


# ---------------------------------------------------------------------------
# HF sync (4.3) — non-blocking pushes, pull on a fresh box. Mirrors the
# supervised trainer's _hf_push: synchronous uploads stalled the training
# loop by ~25-60 s per push (user-noticed 2026-09-25), so uploads run in a
# daemon thread and coalesce (a skip on busy; the next round pushes newer
# state anyway). Trainer.save's tmp+rename keeps an in-flight upload on a
# stable inode even as latest.pt is replaced underneath it.
# ---------------------------------------------------------------------------

def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


_hf_state = {"busy": False, "lock": threading.Lock()}


def hf_push_async(paths: list[tuple[Path, str]], repo: str, log=print,
                  message: str = "self-play round artifacts") -> None:
    """Push (local_path, repo_path) pairs to the HF model repo as ONE
    commit. One commit per round is a hard operational need: HF caps
    repository commits at 128/hour, and the per-file upload_file pattern
    (~10 commits/round) hit the cap after 25 rounds (429s, live 2026-09-30)."""
    with _hf_state["lock"]:
        if _hf_state["busy"]:
            return
        _hf_state["busy"] = True

    def _work():
        try:
            from huggingface_hub import CommitOperationAdd, HfApi
            api = HfApi(token=_hf_token())
            api.create_repo(repo, private=True, exist_ok=True)
            ops = [CommitOperationAdd(path_in_repo=name,
                                      path_or_fileobj=str(path))
                   for path, name in paths if path.exists()]
            if ops:
                api.create_commit(repo_id=repo, operations=ops,
                                  commit_message=message)
                log(f"pushed {len(ops)} file(s) -> hf:{repo} (1 commit)")
        except Exception as exc:            # noqa: BLE001 — never kill the run
            log(f"WARN: hf push failed ({exc}); continuing (files on disk)")
        finally:
            with _hf_state["lock"]:
                _hf_state["busy"] = False

    threading.Thread(target=_work, daemon=True).start()


def hf_pull_prefix(repo: str, prefix: str, dest: Path, log=print) -> bool:
    """Download `<prefix>/**` from the HF repo into `dest` (a fresh box
    resumes from remote state alone, §5 rule 1). Files land at
    dest/prefix/… — pass dest = the out dir's PARENT so the layout
    reproduces. Returns True when anything was fetched."""
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo, allow_patterns=f"{prefix}/**",
                          local_dir=str(dest), token=_hf_token())
    except Exception as exc:              # noqa: BLE001 — a missing repo or
        log(f"WARN: hf pull failed ({exc}); starting fresh")  # no net = fresh
        return False
    got = dest / prefix
    ok = got.exists() and any(p.is_file() for p in got.rglob("*"))
    if ok:
        log(f"pulled {prefix}/** from hf:{repo}")
    return ok


def run_loop(cfg: LoopConfig, log=print) -> list[dict]:
    """Bootstrap-or-resume, then cfg.iters rounds. Returns the per-round
    history (this run's rounds only after a resume)."""
    out = Path(cfg.out_dir)
    (out / "units").mkdir(parents=True, exist_ok=True)
    latest = out / "latest.pt"

    if cfg.hf_repo and not latest.exists():
        # Fresh box: remote state alone must reconstruct the run (§5 rule 1).
        hf_pull_prefix(cfg.hf_repo, out.name, out.parent, log)

    if latest.exists():
        trainer = Trainer.load(latest)
        log(f"resumed trainer from {latest} (steps {trainer.steps})")
    else:
        trainer = Trainer.load(cfg.seed_ckpt)
        trainer.save(latest)      # round 1's evaluator reads latest.pt
        log(f"bootstrapped from {cfg.seed_ckpt} "
            f"(steps {trainer.steps}) → {latest}")
    overrides = _apply_overrides(trainer, cfg)
    if overrides:
        log(f"loop config overrides: {', '.join(overrides)}")
    eval_dev = cfg.eval_device or str(trainer.device)
    log(f"trainer device {trainer.device}, evaluator device {eval_dev}, "
        f"n={cfg.n}, {cfg.workers} workers, buffer {cfg.buffer_requests}/worker, "
        f"{cfg.games_per_iter} games/iter, start prefixes 1..{cfg.start_plies}")

    done_units = sorted(d for d in (out / "units").glob("iter-*")
                        if (d / "DONE.json").exists())
    buffer = SelfPlayBuffer(cfg.buffer_games)
    if done_units:
        records = ingest_units(done_units)
        buffer.add_all(records)
        log(f"ingested {len(done_units)} completed unit(s), "
            f"{len(records)} games, {buffer.total_positions} positions")
    first_iter = (max(int(d.name.split("-")[1]) for d in done_units) + 1
                  if done_units else 1)

    log_path = out / "log.jsonl"
    history: list[dict] = []
    for it in range(first_iter, first_iter + cfg.iters):
        metrics = run_round(trainer, buffer, cfg, it, latest)
        trainer.save(latest)
        snapshot = None
        if cfg.save_every and it % cfg.save_every == 0:
            snapshot = trainer.save(out / f"net-iter-{it:04d}.pt")
        with log_path.open("a") as f:
            f.write(json.dumps(metrics) + "\n")
        if cfg.hf_repo and it % max(1, cfg.push_every) == 0:
            prefix = out.name
            unit = out / "units" / f"iter-{it:04d}"
            paths = [(latest, f"{prefix}/latest.pt"),
                     (log_path, f"{prefix}/log.jsonl")]
            paths += [(f, f"{prefix}/units/{unit.name}/{f.name}")
                      for f in sorted(unit.iterdir())]
            if snapshot is not None:
                paths.append((snapshot, f"{prefix}/{snapshot.name}"))
            hf_push_async(paths, cfg.hf_repo, log,
                          message=f"round {it}: ckpt, log, {unit.name}")
        history.append(metrics)
        t, sp = metrics["train"], metrics["sp"]
        log(f"it {metrics['iter']:3d}  steps {metrics['steps']:6d}  "
            f"sp {sp['sp_s']:7.1f}s ({sp['games_per_hour']:6.0f} g/h)  "
            f"train {metrics['train_s']:6.1f}s  "
            f"loss {t['loss']:.4f}  policy {t['policy']:.4f}  "
            f"value {t['value']:.4f}  "
            f"v_acc {t['value_acc']:.3f}  "
            f"draws {sp['draw_rate']:4.0%}  plies {sp['mean_plies']:5.1f}")
    return history


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m chess_ai.train.selfplay_loop",
        description="Chess self-play training loop (Gumbel engine → "
                    "dense-π′ training → checkpoint, one round at a time).")
    p.add_argument("--seed-ckpt", default=LoopConfig.seed_ckpt)
    p.add_argument("--out", default=LoopConfig.out_dir)
    p.add_argument("--iters", type=int, default=LoopConfig.iters)
    p.add_argument("--games-per-iter", type=int,
                   default=LoopConfig.games_per_iter)
    p.add_argument("--train-steps", type=int, default=LoopConfig.train_steps)
    p.add_argument("--batch", type=int, default=LoopConfig.batch_size)
    p.add_argument("--lr", type=float, default=LoopConfig.lr)
    p.add_argument("--n", type=int, default=LoopConfig.n)
    p.add_argument("--workers", type=int, default=LoopConfig.workers)
    p.add_argument("--buffer-requests", type=int,
                   default=LoopConfig.buffer_requests)
    p.add_argument("--start-plies", type=int, default=LoopConfig.start_plies,
                   help="diverse init: prefix depth 1..N (0 = startpos only)")
    p.add_argument("--start-pool-size", type=int,
                   default=LoopConfig.start_pool_size)
    p.add_argument("--max-plies", type=int, default=LoopConfig.max_plies)
    p.add_argument("--resign", type=float, default=LoopConfig.resign_value)
    p.add_argument("--c-visit", type=float, default=LoopConfig.c_visit)
    p.add_argument("--buffer-games", type=int, default=LoopConfig.buffer_games)
    p.add_argument("--aux-weight", type=float,
                   default=LoopConfig.aux_reply_weight)
    p.add_argument("--save-every", type=int, default=LoopConfig.save_every)
    p.add_argument("--seed", type=int, default=LoopConfig.seed)
    p.add_argument("--device", default=LoopConfig.device)
    p.add_argument("--eval-device", default=LoopConfig.eval_device,
                   help="evaluator device ('' = trainer's device; e.g. mps, "
                        "cuda, cpu)")
    p.add_argument("--hf-repo", default=LoopConfig.hf_repo,
                   help="HF model repo for round pushes + fresh-box pull "
                        "(empty = off)")
    p.add_argument("--push-every", type=int, default=LoopConfig.push_every,
                   help="push cadence in rounds (HF caps commits at 128/h)")
    p.add_argument("--eval-bf16", action="store_true",
                   help="evaluator inference autocast (CUDA)")
    p.add_argument("--eval-compile", action="store_true",
                   help="torch.compile the evaluator (dynamic shapes)")
    p.add_argument("--train-bf16", action="store_true",
                   help="training autocast (CUDA)")
    a = p.parse_args(argv)

    cfg = LoopConfig(
        seed_ckpt=a.seed_ckpt, out_dir=a.out, iters=a.iters,
        games_per_iter=a.games_per_iter, train_steps=a.train_steps,
        batch_size=a.batch, lr=a.lr, n=a.n, workers=a.workers,
        buffer_requests=a.buffer_requests, start_plies=a.start_plies,
        start_pool_size=a.start_pool_size, max_plies=a.max_plies,
        resign_value=a.resign, c_visit=a.c_visit, buffer_games=a.buffer_games,
        aux_reply_weight=a.aux_weight, save_every=a.save_every, seed=a.seed,
        device=a.device, eval_device=a.eval_device, hf_repo=a.hf_repo,
        push_every=a.push_every, eval_bf16=a.eval_bf16,
        eval_compile=a.eval_compile, train_bf16=a.train_bf16)
    if cfg.n < 2:
        p.error("--n must be ≥ 2 (the SelfPlayConfig contract)")
    run_loop(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
