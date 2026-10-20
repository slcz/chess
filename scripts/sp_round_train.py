#!/usr/bin/env python3
"""sp_round_train.py — train on a directory of self-play shards.

The smallest useful version of the training step, meant to be read: local
files only, no network, no cache, no uploads. Point it at shard directories,
start from a checkpoint, get a new checkpoint.

    python scripts/sp_round_train.py --data-dir data/round1 \
        --base-ckpt runs/gen1.pt --out runs/gen2

--data-dir is searched recursively for shard directories (a DONE.json next to
shard-0.spbin) — that is the only layout it requires, and it is what
engine/build/selfplay writes.

The heavy lifting lives in chess_ai/train/: selfplay_data.PackedRound ingests
a round into flat arrays, selfplay_data.stack_selfplay turns sampled rows
into tensors, trainer.Trainer owns the model, optimizer and losses. This file
is just the recipe that connects them.

Every ply is trained (the "all plies" recipe): plies searched with the fast
cap get --fast-weight on the policy, value and aux losses; slow-cap plies get
1.0. One epoch by default, sampling positions with replacement.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_ai.selfplay.shard import read_unit                       # noqa: E402
from chess_ai.train.selfplay_data import (                         # noqa: E402
    PackedRound, stack_selfplay)
from chess_ai.train.trainer import Trainer, tensors_to_torch       # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", nargs="+", required=True,
                    help="one or more directories to search for shards")
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True,
                    help="checkpoint prefix; epoch N is saved as <out>-eN.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fast-weight", type=float, default=0.15,
                    help="loss weight on fast-cap plies (1.0 = no down-weighting)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # 1. ingest every shard directory under the given roots
    units = sorted({p.parent for root in args.data_dir
                    for p in Path(root).rglob("DONE.json")})
    if not units:
        raise SystemExit(f"no shard directories (DONE.json) under {args.data_dir}")
    round_ = PackedRound()
    for unit in units:
        for game in read_unit(unit):
            round_.add_game(game, all_plies=True)
    round_.finalize()
    print(f"[sp_round_train] ingested {len(units)} shards, {len(round_)} positions",
          flush=True)

    # 2. load the net; the checkpoint's config carries the architecture
    trainer = Trainer.load(args.base_ckpt, device=args.device)
    for group in trainer.opt.param_groups:
        group["lr"] = args.lr
    trainer.cfg.lr = args.lr
    trainer.cfg.compile = True
    trainer._maybe_compile()
    trainer.model.train()

    # 3. sample positions with replacement, one epoch at a time
    steps = math.ceil(len(round_) / args.batch) * args.epochs
    rng = random.Random(args.seed)
    for step in range(1, steps + 1):
        batch = tensors_to_torch(
            stack_selfplay([round_.sample(rng.randrange(len(round_)),
                                          fast_w=args.fast_weight,
                                          fast_aux=args.fast_weight)
                            for _ in range(args.batch)]),
            trainer.device)
        metrics = trainer.train_step(batch)
        if step % 100 == 0:
            print(f"[sp_round_train] step {step}/{steps} "
                  + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()),
                  flush=True)
        if step % math.ceil(len(round_) / args.batch) == 0:
            epoch = step // math.ceil(len(round_) / args.batch)
            out = Path(f"{args.out}-e{epoch}.pt")
            trainer.save(out)
            print(f"[sp_round_train] saved {out}", flush=True)


if __name__ == "__main__":
    main()
