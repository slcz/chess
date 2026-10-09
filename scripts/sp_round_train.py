#!/usr/bin/env python3
"""
scripts/sp_round_train.py — one training round over self-play units on
HuggingFace: download → ingest (global (seed, game_index) dedup) →
on-the-fly batch sampling → Trainer.train_step for --epochs epochs with
an atomic checkpoint saved AND pushed to the ckpt repo at each epoch
boundary.

    python3 scripts/sp_round_train.py --data-repo chriszhang01/chess-selfplay \
        --prefix units/sp-42k-d1 --base-ckpt runs/ckpt_a/net.pt \
        --out runs/iter1 --ckpt-repo chriszhang01/chess-value-net \
        --epochs 2 --lr 2e-4 --batch 512 [--max-games 0] [--limit-steps 0]

Batch sampling is on-the-fly (selfplay_sample per drawn position): 10K
games ≈ 860K positions ≈ 25 GB as stacked tensors — only the batch is
ever materialized. LR is set post-load on the optimizer's param groups
(fine-tune default 2e-4; the base checkpoint's saved cfg otherwise
stands).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_ai.selfplay.shard import SpGame, read_unit  # noqa: E402
from chess_ai.train.selfplay_data import selfplay_sample, stack_selfplay  # noqa: E402
from chess_ai.train.trainer import Trainer, tensors_to_torch  # noqa: E402


def load_games(repo: str, prefix: str, token: str | None,
               max_games: int) -> list[SpGame]:
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo, repo_type="dataset", token=token,
                              allow_patterns=[f"{prefix}/part-*/**"])
    root = Path(local) / prefix
    unit_dirs = sorted(p for p in root.glob("part-*") if (p / "DONE.json").exists())
    if not unit_dirs:
        raise SystemExit(f"no complete parts under {prefix} in {repo}")
    games: list[SpGame] = []
    seen: set[tuple[int, int]] = set()
    for ud in unit_dirs:
        for g in read_unit(ud):
            key = (json_seed(ud), g.game_index)
            if key not in seen:
                seen.add(key)
                games.append(g)
    if max_games:
        games = games[:max_games]
    print(f"[sp_round_train] ingested {len(games)} games, "
          f"{sum(g.n_plies for g in games)} positions "
          f"from {len(unit_dirs)} parts")
    return games


def json_seed(unit_dir: Path) -> int:
    return int(__import__("json").loads(
        (unit_dir / "DONE.json").read_text())["unit_seed"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-repo", default="chriszhang01/chess-selfplay")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--ckpt-repo", default="chriszhang01/chess-value-net")
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True, help="output checkpoint prefix")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--value-weight", type=float, default=None,
                    help="override the ckpt's value loss weight (train_step "
                         "reads cfg per step); e.g. 0.25 for a gentle "
                         "value-head fine-tune")
    ap.add_argument("--policy-weight", type=float, default=None)
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--limit-steps", type=int, default=0,
                    help="cap total steps (smoke runs)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")     # env wins over .env
    env = Path(__file__).resolve().parent.parent / ".env"
    if token is None and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    if token == "":                        # explicit empty = anonymous:
        token = None                       # public download only, no push

    games = load_games(args.data_repo, args.prefix, token, args.max_games)
    positions = [(gi, ply) for gi, g in enumerate(games)
                 for ply in range(g.n_plies)]
    steps_per_epoch = -(-len(positions) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    if args.limit_steps:
        total_steps = min(total_steps, args.limit_steps)
    print(f"[sp_round_train] {len(positions)} positions, batch {args.batch} "
          f"-> {steps_per_epoch} steps/epoch x {args.epochs} epochs")

    trainer = Trainer.load(args.base_ckpt)
    trainer.model.eval()  # noqa: pure training below re-enables grad; eval()
    trainer.model.train()
    for grp in trainer.opt.param_groups:
        grp["lr"] = args.lr
    if args.value_weight is not None:
        trainer.cfg.value_weight = args.value_weight
    if args.policy_weight is not None:
        trainer.cfg.policy_weight = args.policy_weight
    print(f"[sp_round_train] lr {args.lr} | policy_w {trainer.cfg.policy_weight} "
          f"| value_w {trainer.cfg.value_weight} "
          f"| aux_w {trainer.cfg.aux_reply_weight}")
    rng = random.Random(args.seed)

    step = 0
    for epoch in range(args.epochs):
        sums = {"loss": 0.0, "policy": 0.0, "value": 0.0, "aux": 0.0}
        for _ in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = [positions[i] for i in
                   rng.sample(range(len(positions)),
                              min(args.batch, len(positions)))]
            samples = [selfplay_sample(games[gi], ply) for gi, ply in idx]
            batch = tensors_to_torch(stack_selfplay(samples), trainer.device)
            m = trainer.train_step(batch)
            for k in sums:
                sums[k] += m[k]
            step += 1
            if step % 100 == 0:
                print(f"[sp_round_train] step {step}/{total_steps} "
                      f"loss {sums['loss'] / 100:.4f} "
                      f"policy {sums['policy'] / 100:.4f} "
                      f"value {sums['value'] / 100:.4f}", flush=True)
                sums = dict.fromkeys(sums, 0.0)
        name = f"{Path(args.out).name}-e{epoch + 1}.pt"
        out = Path(args.out).parent / name
        trainer.save(out)
        print(f"[sp_round_train] saved {out}", flush=True)
        if token:
            from huggingface_hub import HfApi
            HfApi(token=token).upload_file(
                path_or_fileobj=str(out), path_in_repo=f"selfplay/{name}",
                repo_id=args.ckpt_repo, repo_type="model")
            print(f"[sp_round_train] pushed selfplay/{name} -> "
                  f"{args.ckpt_repo}", flush=True)
        else:
            print("[sp_round_train] no HF_TOKEN — checkpoint kept local",
                  flush=True)


if __name__ == "__main__":
    main()
