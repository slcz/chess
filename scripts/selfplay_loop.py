#!/usr/bin/env python3
"""selfplay_loop.py — one self-contained train/eval loop on a single machine.

Generate games with the engine, train on them, gate the result against the
current best, adopt if it wins, repeat. No cloud, no uploads, no fixed
directory layout: everything is an argument and the work directory is
self-contained (best.pt, per-iteration subdirs, loop.jsonl).

    python scripts/selfplay_loop.py \
        --work-dir /data/az-run --book book.epd \
        --iterations 20 --games 30000 --gate-games 400 --adopt-min 0.55

`--init-ckpt` is optional: pass it to seed/resume from an existing net, or
omit it to create a random-init `best.pt` (the zero-knowledge start).

Per iteration, in <work-dir>/itNNN/:
    selfplay/     the engine's shard (+ DONE.json)
    train/        the trained checkpoint(s)
    gate.log      the 400-game gate against the current best
and one JSONL record per iteration appended to <work-dir>/loop.jsonl.

Assumes engine/build/selfplay is built and the interpreter has the repo's
deps (torch, cffi...). Delete an iteration's selfplay/ to regenerate it; the
loop skips any iteration whose trained checkpoint already exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ENGINE = REPO / "engine" / "build" / "selfplay"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def random_init_checkpoint(path: Path, seed: int) -> None:
    """Write a Trainer-format checkpoint with random weights.

    This is the loop's zero-knowledge start: default ChessTransformer
    architecture, so selfplay, sp_round_train, uci_mux and sf_gauntlet all
    load it through the same Trainer.load path.
    """
    import torch
    from chess_ai.net.model import ChessTransformer, NetConfig
    from chess_ai.train import Trainer, TrainConfig

    torch.manual_seed(seed)
    model = ChessTransformer(NetConfig())
    Trainer(model, TrainConfig(device="cpu", compile=False,
                               seed=seed)).save(path)
    log(f"created random-init {path.name} "
        f"({sum(p.numel() for p in model.parameters()):,} params)")


def count_games(unit_dir: Path) -> int:
    """Games in a shard directory, via the shard reader (0 if absent)."""
    if not (unit_dir / "shard-0.spbin").exists():
        return 0
    sys.path.insert(0, str(REPO))
    from chess_ai.selfplay.shard import read_shard
    return len(read_shard(unit_dir / "shard-0.spbin"))


def ensure_done(unit_dir: Path, seed: int) -> bool:
    """Write DONE.json for a shard that has no marker (a killed generation).

    The engine only writes DONE.json on a clean finish, so a crashed run
    leaves a readable shard that read_unit() would refuse. Returns True if a
    marker was written.
    """
    if not (unit_dir / "shard-0.spbin").exists():
        return False
    if (unit_dir / "DONE.json").exists():
        return False
    n = count_games(unit_dir)
    if n == 0:
        return False
    (unit_dir / "DONE.json").write_text(json.dumps(
        {"n_workers": 1, "unit_seed": seed, "games_done": n}) + "\n")
    log(f"recovered {n} games from an unfinished shard in {unit_dir}")
    return True


def pump(cmd: list[str], log_path: Path, env: dict[str, str] | None = None,
         poll: int = 60) -> int:
    """Run a command, streaming its log tail so a long stage is not silent."""
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env=env, cwd=str(REPO))
        stop = threading.Event()

        def tail() -> None:
            while not stop.wait(poll):
                lines = [l for l in log_path.read_text().splitlines() if l.strip()]
                if lines:
                    log(f"  … {lines[-1][:110]}")

        t = threading.Thread(target=tail, daemon=True)
        t.start()
        rc = proc.wait()
        stop.set()
        t.join(timeout=1)
    if rc != 0:
        log(f"FAILED ({rc}); tail of {log_path}:")
        for line in log_path.read_text().splitlines()[-15:]:
            print("    " + line)
    return rc


def generate(args, best: Path, unit_dir: Path, seed: int) -> int:
    env = dict(os.environ, NN_BF16="1", NN_COMPILE="1")
    cmd = [str(ENGINE), str(best), str(unit_dir), str(args.games),
           str(args.sims), str(args.pool), str(args.rows),
           str(args.sample_plys), str(args.max_plys), "4",
           args.device, str(seed), str(args.fast_sims), str(args.fast_p)]
    return pump(cmd, unit_dir.parent / "generate.log", env=env)


def train(args, unit_dirs: list[Path], buffer_dirs: list[Path],
          base: Path, out_prefix: Path) -> int:
    """unit_dirs train at full weight; buffer_dirs (NEWEST first) replay
    through the trainer's deterministic keep^age trim."""
    cmd = [sys.executable, "scripts/sp_round_train.py",
           "--data-dir", *[str(d) for d in unit_dirs],
           *[x for b in buffer_dirs for x in ("--buffer", str(b))],
           "--buffer-keep", str(args.buffer_keep),
           "--trim-seed", str(args.seed),
           "--base-ckpt", str(base), "--out", str(out_prefix),
           "--epochs", str(args.epochs), "--batch", str(args.batch),
           "--lr", str(args.lr), "--fast-weight", str(args.fast_weight),
           "--seed", str(args.seed), "--device", args.device]
    return pump(cmd, out_prefix.parent / "train.log")


def gate(args, candidate: Path, best: Path, eval_json: Path) -> float | None:
    """Run the 400-game gate; the machine-readable result lands in
    eval.json (the per-round eval artifact), the human log beside it."""
    log_path = eval_json.with_suffix(".log")
    cmd = [sys.executable, "scripts/gate_match.py",
           "--ckpt-a", str(candidate), "--ckpt-b", str(best),
           "--games", str(args.gate_games // 2),
           "--a-sims", str(args.gate_sims), "--b-sims", str(args.gate_sims),
           "--book", str(args.book), "--device", args.device,
           "--rows", str(args.rows), "--json", str(eval_json)]
    if pump(cmd, log_path) != 0:
        return None
    if eval_json.exists():
        return float(json.loads(eval_json.read_text())["score_a"])
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--init-ckpt",
                    help="starting checkpoint; omit to create a random-init "
                         "best.pt in the work directory")
    ap.add_argument("--book", required=True,
                    help="FEN-per-line opening book for the gate")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--games", type=int, default=30000)
    ap.add_argument("--sims", type=int, default=300)
    ap.add_argument("--fast-sims", type=int, default=50)
    ap.add_argument("--fast-p", type=float, default=0.75)
    ap.add_argument("--sample-plys", type=int, default=10)
    ap.add_argument("--pool", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=2048)
    ap.add_argument("--max-plys", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fast-weight", type=float, default=0.15)
    ap.add_argument("--history", type=int, default=3,
                    help="train on the last N rounds' shards at full weight "
                         "(1 = newest only); older rounds replay trimmed")
    ap.add_argument("--buffer-keep", type=float, default=2 / 3,
                    help="fraction of each older round kept per round of age "
                         "(deterministic trim; 1 = replay everything)")
    ap.add_argument("--gate-games", type=int, default=400)
    ap.add_argument("--gate-sims", type=int, default=200)
    ap.add_argument("--adopt-min", type=float, default=0.5,
                    help="adopt only if the gate score is strictly above this; "
                         "a tie (exactly 0.5) is left for a human call")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-recover", action="store_true",
                    help="do not write DONE.json for an unfinished shard")
    args = ap.parse_args()

    if not ENGINE.exists():
        log(f"engine not built: {ENGINE} (run `make -C engine all build/selfplay`)")
        return 1
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    best = work / "best.pt"
    if not best.exists():
        if args.init_ckpt:
            init = Path(args.init_ckpt).expanduser()
            if not init.exists():
                log(f"init checkpoint not found: {init}")
                return 1
            shutil.copy(init, best)
            log(f"seeded best.pt from {init}")
        else:
            log("no --init-ckpt given; creating a random-init best.pt")
            random_init_checkpoint(best, args.seed)
    log(f"work {work} | best {best.name} sha {sha256(best)[:12]} | "
        f"{args.iterations} x ({args.games} games @ {args.sims}/{args.fast_sims} sims "
        f"-> {args.epochs} epoch(s) -> {args.gate_games}-game gate @ {args.adopt_min})")

    for it in range(1, args.iterations + 1):
        d = work / f"it{it:03d}"
        unit = d / "selfplay"
        d.mkdir(parents=True, exist_ok=True)
        ckpt = d / f"train-e{args.epochs}.pt"
        t0 = time.time()

        if not ckpt.exists():
            if not args.no_recover:
                ensure_done(unit, args.seed + it)
            games = count_games(unit)
            if games < args.games:
                if games:
                    log(f"it{it}: shard has {games}/{args.games}, regenerating")
                    shutil.rmtree(unit, ignore_errors=True)
                log(f"it{it}: generating {args.games} games")
                rc = generate(args, best, unit, args.seed + it)
                if rc != 0:
                    return rc
                games = count_games(unit)
                log(f"it{it}: generated {games} games in {time.time() - t0:.0f}s")
            else:
                log(f"it{it}: reusing existing shard ({games} games)")
            first = max(1, it - args.history + 1)
            round_dirs = [work / f"it{j:03d}" / "selfplay"
                          for j in range(first, it + 1)
                          if (work / f"it{j:03d}" / "selfplay" / "DONE.json").exists()]
            # older rounds replay through the keep^age trim, newest first
            buffer_dirs = [work / f"it{j:03d}" / "selfplay"
                           for j in range(first - 1, 0, -1)
                           if (work / f"it{j:03d}" / "selfplay" / "DONE.json").exists()]
            log(f"it{it}: training on {len(round_dirs)} round(s) + "
                f"{len(buffer_dirs)} buffer pool(s)")
            rc = train(args, round_dirs, buffer_dirs, best, d / "train")
            if rc != 0:
                return rc

        log(f"it{it}: gating {ckpt.name} against best.pt")
        score = gate(args, ckpt, best, d / "eval.json")
        if score is None:
            log(f"it{it}: gate failed; keeping the current best")
            adopted = False
        else:
            adopted = score > args.adopt_min      # ties are a judgement call
            if adopted:
                shutil.copy(ckpt, work / "best.pt.next")
                os.replace(work / "best.pt.next", best)
            log(f"it{it}: gate {score:.3f} -> "
                f"{'ADOPTED' if adopted else 'kept previous best'}")

        rec = {"iteration": it, "games": games, "epochs": args.epochs,
               "gate_score": score, "adopted": adopted,
               "adopt_min": args.adopt_min, "elapsed_s": round(time.time() - t0),
               "ckpt": str(ckpt.relative_to(work)),
               "ckpt_sha256": sha256(ckpt) if ckpt.exists() else None,
               "train_json": str((d / "train-train.json").relative_to(work)),
               "eval_json": str((d / "eval.json").relative_to(work)),
               "best_sha256": sha256(best)[:16],
               "params": {k: getattr(args, k) for k in
                          ("sims", "fast_sims", "fast_p", "history", "lr",
                           "batch", "epochs", "fast_weight", "gate_games",
                           "buffer_keep")}}
        with open(work / "loop.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    log(f"done — best.pt sha {sha256(best)[:12]} | records in {work / 'loop.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
