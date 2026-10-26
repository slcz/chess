#!/usr/bin/env python3
"""sp_round_train.py — train on a directory of self-play shards, with an
optional replay buffer and a per-round record.

The smallest useful version of the training step, meant to be read: local
files only, no network, no cache, no uploads. Point it at shard directories,
start from a checkpoint, get a new checkpoint — plus a `rec.json` recording
what the round did.

    python scripts/sp_round_train.py --data-dir data/round2 \
        --base-ckpt runs/gen1.pt --out runs/gen2

    # with a replay buffer: --buffer repeats, NEWEST pool first; the k-th
    # one has aged k rounds and keeps (2/3)^k of its games via the
    # deterministic buffer_keep trim (same kept set on any box):
    python scripts/sp_round_train.py --data-dir data/round3 \
        --buffer data/round2 --buffer data/round1 \
        --base-ckpt runs/gen2.pt --out runs/gen3

--data-dir / --buffer are searched recursively for shard directories (a
DONE.json next to shard-0.spbin) — that is the only layout they require,
and it is what engine/build/selfplay writes.

The heavy lifting lives in chess_ai/train/: selfplay_data.PackedRound ingests
a round into flat arrays, selfplay_data.stack_selfplay turns sampled rows
into tensors, trainer.Trainer owns the model, optimizer and losses, and
selfplay_data.buffer_keep is the trim. This file is just the recipe that
connects them.

Every ply is trained (the "all plies" recipe): plies searched with the fast
cap get --fast-weight on the policy, value and aux losses; slow-cap plies get
1.0. One epoch by default, sampling positions with replacement.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_ai.selfplay.shard import read_unit                       # noqa: E402
from chess_ai.train.selfplay_data import (                          # noqa: E402
    PackedRound, buffer_keep, stack_selfplay)
from chess_ai.train.trainer import Trainer, tensors_to_torch        # noqa: E402


def new_tally() -> dict:
    return {"w": 0, "d": 0, "l": 0, "games": 0}


def wdl_out(t: dict) -> dict:
    return {"games": t["games"], "wdl": [t["w"], t["d"], t["l"]]}


def ingest(units, all_plies: bool, age: int | None, trim_seed: int,
           tally: dict, round_: PackedRound) -> None:
    """Parse shard units into `round_`; age None = fresh (all games),
    otherwise the deterministic (2/3)^age trim applies per game."""
    for unit in units:
        useed = None
        if age is not None:
            useed = json.loads((unit / "DONE.json").read_text())["unit_seed"]
        for game in read_unit(unit):
            if age is not None and not buffer_keep(useed, game.game_index,
                                                   age, trim_seed):
                continue
            tally[{"1.0": "w", "0.5": "d"}.get(str(game.z_white), "l")] += 1
            tally["games"] += 1
            round_.add_game(game, all_plies=all_plies)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", nargs="+", required=True,
                    help="one or more directories to search for shards")
    ap.add_argument("--buffer", action="append", default=[],
                    help="older pool directory to replay (repeatable; "
                         "NEWEST first — the k-th --buffer has age k)")
    ap.add_argument("--trim-seed", type=int, default=0,
                    help="seed of the deterministic buffer trim")
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--out", required=True,
                    help="checkpoint prefix; epoch N is saved as <out>-eN.pt")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--fast-weight", type=float, default=0.15,
                    help="loss weight on fast-cap plies (1.0 = no down-weighting)")
    ap.add_argument("--slow-only", action="store_true",
                    help="train only on slow-cap plies: the fast ones are "
                         "dropped at ingest (implies --fast-weight is unused)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()

    # 1. ingest: fresh pools take every game; each --buffer (newest first)
    #    keeps (2/3)^age of its games via the deterministic trim
    def shard_units(roots):
        units = sorted({p.parent for root in roots
                        for p in Path(root).rglob("DONE.json")})
        if not units:
            raise SystemExit(f"no shard directories (DONE.json) under {roots}")
        return units

    all_plies = not args.slow_only
    round_ = PackedRound()
    fresh = new_tally()
    ingest(shard_units(args.data_dir), all_plies, None, 0, fresh, round_)
    buffers = []
    for age, root in enumerate(args.buffer, start=1):
        b = new_tally()
        ingest(shard_units([root]), all_plies, age, args.trim_seed, b, round_)
        buffers.append({"dir": str(root), "age": age, **wdl_out(b)})
    round_.finalize()
    print(f"[sp_round_train] ingested fresh {fresh['games']} + buffer "
          f"{sum(b['games'] for b in buffers)} games, {len(round_)} positions",
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
    per_epoch = math.ceil(len(round_) / args.batch)
    steps = per_epoch * args.epochs
    rng = random.Random(args.seed)
    history = []
    for step in range(1, steps + 1):
        batch = tensors_to_torch(
            stack_selfplay([round_.sample(rng.randrange(len(round_)),
                                          fast_w=args.fast_weight,
                                          fast_aux=args.fast_weight)
                            for _ in range(args.batch)]),
            trainer.device)
        metrics = trainer.train_step(batch)
        history.append(metrics)
        if step % 100 == 0:
            print(f"[sp_round_train] step {step}/{steps} "
                  + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()),
                  flush=True)
        if step % per_epoch == 0:
            epoch = step // per_epoch
            out = Path(f"{args.out}-e{epoch}.pt")
            trainer.save(out)
            print(f"[sp_round_train] saved {out}", flush=True)

    # 4. the round record: what this round was and how it went
    def avg(rows):
        rows = rows or history
        return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}

    ckpts = [str(Path(f"{args.out}-e{e}.pt")) for e in range(1, args.epochs + 1)]
    rec = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {"data_dir": args.data_dir, "buffer": args.buffer,
                 "trim_seed": args.trim_seed, "base_ckpt": args.base_ckpt,
                 "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                 "fast_weight": args.fast_weight, "slow_only": args.slow_only,
                 "seed": args.seed, "device": args.device},
        "fresh": wdl_out(fresh), "buffer": buffers,
        "steps": steps, "positions": len(round_),
        "metrics": {"begin": avg(history[:100]), "end": avg(history[-100:])},
        "checkpoints": ckpts,
        "elapsed_s": round(time.time() - t0, 1),
    }
    rec_path = Path(f"{args.out}-rec.json")
    rec_path.write_text(json.dumps(rec, indent=2))
    print(f"[sp_round_train] record -> {rec_path}", flush=True)


if __name__ == "__main__":
    main()
