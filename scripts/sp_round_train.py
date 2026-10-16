#!/usr/bin/env python3
"""
scripts/sp_round_train.py — one training round over self-play units on
HuggingFace: download → ingest (global (seed, game_index) dedup) →
on-the-fly batch sampling → Trainer.train_step for --epochs epochs with
an atomic checkpoint saved AND pushed to the ckpt repo at each epoch
boundary.

    python3 scripts/sp_round_train.py --data-repo chriszhang01/chess-selfplay \
        --prefix units/sp-42k-d1 --base-ckpt runs/gen1.pt \
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

from chess_ai.selfplay.shard import read_unit  # noqa: E402
from chess_ai.train.selfplay_data import (  # noqa: E402
    PackedRound,
    stack_selfplay,
)
from chess_ai.train.trainer import Trainer, tensors_to_torch  # noqa: E402


def _pack_unit(ud: str):
    """Worker for load_round's pool: part dir -> (unit_seed, game idx, fragment)."""
    seed = json_seed(Path(ud))
    packed = PackedRound()
    idx = []
    for g in read_unit(ud):
        packed.add_game(g)
        idx.append(g.game_index)
    packed.finalize()
    return seed, np.array(idx, dtype=np.int64), packed


def load_round(repo: str, prefixes: list, token: str | None,
               max_games: int, workers: int = 8,
               cache_dir: str = "") -> PackedRound:
    """Stream every complete part under the prefixes into a PackedRound:
    each game is converted to flat arrays on arrival, so peak memory
    tracks the packed size (~2 GB for 250k games), not the record
    objects (~55 GB — that OOM killed the first 250k-game ingest).

    Units are packed in a process pool (the struct parsing is
    single-core-bound; 12 workers cut a 532-part round from ~11 min to
    ~2 min). The result is cached to cache_dir keyed by the exact unit
    listing, so sweep re-launches skip the parse entirely."""
    import time

    from huggingface_hub import snapshot_download
    # fnmatch globs: '*' crosses '/', so "<prefix>*" matches BOTH unit
    # layouts — bare-node units/<run>/ and container units/<run>-i<stamp>/
    # (the old "<prefix>/**" matched only the former and silently
    # downloaded nothing for container rounds).
    local = snapshot_download(repo, repo_type="dataset", token=token,
                              allow_patterns=[f"{p}*" for p in prefixes])
    root = Path(local)
    unit_dirs = sorted(set(
        p.parent for p in root.glob("**/DONE.json")
        if any(pref in str(p) for pref in prefixes)))
    if not unit_dirs:
        raise SystemExit(f"no complete parts under {prefixes} in {repo}")

    if cache_dir:
        import hashlib
        # v2 retires caches written by the pre-fix dedup
        key = hashlib.sha1(("v2|" + "\n".join(sorted(map(str, unit_dirs)))
                            + f"|{max_games}").encode()).hexdigest()[:16]
        cache = Path(cache_dir) / f"packed-{key}.npz"
        if cache.exists():
            t0 = time.time()
            packed = PackedRound.load(cache)
            print(f"[sp_round_train] cache hit {cache.name}: "
                  f"{len(packed)} trainable positions "
                  f"({time.time() - t0:.0f}s)")
            return packed
    else:
        cache = None

    if max_games:
        workers = 1    # keep the 'first N games in unit order' semantics
    if workers > 1 and len(unit_dirs) > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(workers) as ex:
            results = list(ex.map(_pack_unit, [str(u) for u in unit_dirs]))
        seen: dict[int, set] = {}      # seed -> game indices already kept
        parts = []
        for seed, idx, frag in results:
            have = seen.setdefault(seed, set())
            fresh = [i for i in idx.tolist() if i not in have]
            if not fresh:
                print(f"[sp_round_train] skipping duplicate unit seed "
                      f"{seed} ({len(idx)} games already ingested)")
                continue
            assert len(fresh) == len(idx), (
                f"unit {seed}: partial overlap ({len(fresh)}/{len(idx)} "
                f"games new) — corrupt or re-sliced parts")
            have.update(fresh)
            parts.append(frag)
        packed = PackedRound.concat(parts)
        n_games = packed.n_games
    else:
        packed = PackedRound()
        seen: set[tuple[int, int]] = set()
        n_games = 0
        for ud in unit_dirs:
            seed = json_seed(ud)
            for g in read_unit(ud):
                key = (seed, g.game_index)
                if key in seen:
                    continue
                seen.add(key)
                packed.add_game(g)
                n_games += 1
                if max_games and n_games >= max_games:
                    break
            if max_games and n_games >= max_games:
                break
        packed.finalize()

    if cache is not None:
        t0 = time.time()
        packed.save(cache)
        print(f"[sp_round_train] packed cache written: {cache} "
              f"({cache.stat().st_size / 1e9:.1f} GB, "
              f"{time.time() - t0:.0f}s)")
    print(f"[sp_round_train] ingested {n_games} games, "
          f"{packed.n_recorded_plies} positions "
          f"from {len(unit_dirs)} parts")
    return packed


def json_seed(unit_dir: Path) -> int:
    return int(__import__("json").loads(
        (unit_dir / "DONE.json").read_text())["unit_seed"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-repo", default="chriszhang01/chess-selfplay")
    ap.add_argument("--prefix", required=True,
                    help="HF path prefix(es), comma-separated for parallel "
                         "instance pools, e.g. units/sp-42k-d1,units/sp-42k-r3-i1,...")
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
    ap.add_argument("--aux-weight", type=float, default=None,
                    help="override the ckpt's aux-reply loss weight; the net must have the aux head (aux_reply=True)")
    ap.add_argument("--soft-weight", type=float, default=None,
                    help="KataGo auxiliary soft-policy-target weight (default 8, KataGo's value); the net must have the soft head")
    ap.add_argument("--soft-temp", type=float, default=4.0,
                    help="temperature T for the soft target pi^(1/T) (KataGo: 4)")
    ap.add_argument("--max-games", type=int, default=0)
    ap.add_argument("--limit-steps", type=int, default=0,
                    help="cap total steps (smoke runs)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save <out>-snap<step>.pt every N steps (early "
                         "verification snapshots; 0 = only at epoch end)")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="CUDA bf16 autocast (no-op off CUDA)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="torch.compile the net (CUDA only)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel packers during ingest (the parse is "
                         "one-core per part; 8-12 cuts a 532-part round "
                         "from ~11 min to ~2 min)")
    ap.add_argument("--cache-dir",
                    default=str(Path(__file__).resolve().parent.parent
                                / "packed-cache"),
                    help="packed-dataset cache ('' disables; keyed by the "
                         "exact unit listing, so sweep re-launches skip "
                         "the parse)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")     # env wins over .env
    env = Path(__file__).resolve().parent.parent / ".env"
    if token is None and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                token = line.split("=", 1)[1].strip()
    if token == "":                        # explicit empty = anonymous:
        token = None                       # public download only, no push

    prefixes = [p.strip() for p in args.prefix.split(",") if p.strip()]
    packed = load_round(args.data_repo, prefixes, token, args.max_games,
                        workers=args.workers, cache_dir=args.cache_dir)
    positions = len(packed)
    steps_per_epoch = -(-positions // args.batch)
    total_steps = steps_per_epoch * args.epochs
    if args.limit_steps:
        total_steps = min(total_steps, args.limit_steps)
    print(f"[sp_round_train] {positions} trainable positions "
          f"(of {packed.n_recorded_plies} recorded plies), "
          f"batch {args.batch} -> {steps_per_epoch} steps/epoch "
          f"x {args.epochs} epochs")

    trainer = Trainer.load(args.base_ckpt, device=args.device)
    print(f"[sp_round_train] device {trainer.device}")
    trainer.model.eval()  # noqa: pure training below re-enables grad; eval()
    trainer.model.train()
    for grp in trainer.opt.param_groups:
        grp["lr"] = args.lr
    trainer.cfg.lr = args.lr          # else the ckpt records the BASE's lr/batch
    trainer.cfg.batch_size = args.batch
    trainer.cfg.bf16 = args.bf16      # load() forces compile=False, so the perf
    trainer.cfg.compile = args.compile  # knobs need re-applying like --lr
    trainer._maybe_compile()
    print(f"[sp_round_train] bf16 {args.bf16} | compile {args.compile}")
    if args.value_weight is not None:
        trainer.cfg.value_weight = args.value_weight
    if args.policy_weight is not None:
        trainer.cfg.policy_weight = args.policy_weight
    if args.aux_weight is not None:
        trainer.cfg.aux_reply_weight = args.aux_weight
    if args.soft_weight is not None:
        trainer.cfg.soft_policy_weight = args.soft_weight
    print(f"[sp_round_train] lr {args.lr} | policy_w {trainer.cfg.policy_weight} "
          f"| value_w {trainer.cfg.value_weight} "
          f"| aux_w {trainer.cfg.aux_reply_weight} "
          f"| soft_w {trainer.cfg.soft_policy_weight} (T {args.soft_temp}, "
          f"head {'yes' if trainer.model.soft is not None else 'no'})")
    rng = random.Random(args.seed)

    step = 0
    for epoch in range(args.epochs):
        sums = {"loss": 0.0, "policy": 0.0, "value": 0.0, "aux": 0.0,
                "soft": 0.0}
        n = 0          # window length: an epoch boundary can shorten it
        for _ in range(steps_per_epoch):
            if step >= total_steps:
                break
            idx = rng.sample(range(positions),
                             min(args.batch, positions))
            samples = [packed.sample(i, args.soft_temp) for i in idx]
            batch = tensors_to_torch(stack_selfplay(samples), trainer.device)
            m = trainer.train_step(batch)
            for k in sums:
                sums[k] += m[k]
            step += 1
            n += 1
            if args.save_every and step % args.save_every == 0:
                snap = f"{args.out}-snap{step}.pt"
                trainer.save(snap)
                print(f"[sp_round_train] snapshot {snap}", flush=True)
                if token:      # snapshots ARE the data-scaling curve — push
                    from huggingface_hub import HfApi      # them off the box
                    HfApi(token=token).upload_file(
                        path_or_fileobj=snap,
                        path_in_repo=f"selfplay/{Path(snap).name}",
                        repo_id=args.ckpt_repo, repo_type="model")
                    print(f"[sp_round_train] pushed "
                          f"selfplay/{Path(snap).name}", flush=True)
            if step % 100 == 0:
                print(f"[sp_round_train] step {step}/{total_steps} "
                      f"loss {sums['loss'] / n:.4f} "
                      f"policy {sums['policy'] / n:.4f} "
                      f"value {sums['value'] / n:.4f} "
                      f"aux {sums['aux'] / n:.4f} "
                      f"soft {sums['soft'] / n:.4f}", flush=True)
                sums = dict.fromkeys(sums, 0.0)
                n = 0
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
