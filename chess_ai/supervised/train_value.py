"""
chess_ai/supervised/train_value.py — the side-track driver CLI.

Four subcommands, one per stage of the value-net validation run:

    python -m chess_ai.supervised.train_value download  --month 2013-01
    python -m chess_ai.supervised.train_value preprocess --pgn data/x.pgn.zst \
        --out data/lichess_2013-01.pack --min-elo 1600 --max-games 20000
    python -m chess_ai.supervised.train_value train --pack data/x.pack \
        --steps 2000 --out runs/value_pilot
    python -m chess_ai.supervised.train_value eval    --ckpt runs/value_pilot/final.pt \
        --pack data/x.pack
    python -m chess_ai.supervised.train_value match   --ckpt runs/value_pilot/final.pt \
        --opponent greedy --n-games 10

Training defaults are the VALUE-ONLY regime this track agreed on:
policy_weight 0, aux 0 — the trunk and WDL head learn from real game
outcomes; the untouched policy head's accuracy is printed as a control
(it should sit at chance ≈ 1/avg-legal-moves, proving nothing leaks
into it). Checkpoints save atomically (Trainer.save) and optionally
push to HuggingFace (HF_TOKEN in .env) — the salad churn rule: no
durable state on the instance.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from chess_ai.engine import Board
from chess_ai.net.model import ChessTransformer, NetConfig
from chess_ai.train.trainer import (
    TrainConfig,
    Trainer,
    pick_device,
    tensors_to_torch,
)
from chess_ai.supervised.dataset import (
    Game,
    PositionSampler,
    load_pack,
    preprocess,
    replay_position,
)

LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{m}.pgn.zst"


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_download(args) -> None:
    """Fetch one monthly dump into data/ (curl, resumable via -C -)."""
    out = Path(args.out or f"data/lichess_{args.month}.pgn.zst")
    out.parent.mkdir(parents=True, exist_ok=True)
    url = LICHESS_URL.format(m=args.month)
    print(f"downloading {url} -> {out}")
    rc = os.system(f"curl -C - -L --retry 3 -o {out} {url}")
    if rc != 0:
        raise SystemExit(f"download failed (rc={rc})")
    print(f"done: {out} ({out.stat().st_size / 1e6:.1f} MB)")


def cmd_preprocess(args) -> None:
    if args.workers > 1:
        from chess_ai.supervised.fastprep import preprocess_parallel
        preprocess_parallel(
            args.pgn, args.out, workers=args.workers,
            min_elo=args.min_elo, min_base_sec=args.min_base_sec,
            min_plies=args.min_plies, max_plies=args.max_plies,
            drop_time_forfeits=args.drop_time_forfeits,
            max_games=args.max_games, max_positions=args.max_positions,
        )
        return
    preprocess(
        args.pgn, args.out,
        min_elo=args.min_elo, min_base_sec=args.min_base_sec,
        min_plies=args.min_plies, max_plies=args.max_plies,
        drop_time_forfeits=args.drop_time_forfeits,
        max_games=args.max_games, max_positions=args.max_positions,
    )


def _val_batches(games: List[Game], picks: List[Tuple[int, int]],
                 batch: int):
    """Replay a fixed (game, ply) list into yieldable eval batches."""
    board = Board()
    buf = []
    for gi, ply in picks:
        buf.append(replay_position(games[gi], ply, board))
        if len(buf) == batch:
            yield _stack(buf)
            buf = []
    if buf:
        yield _stack(buf)


def _stack(buf) -> Dict[str, np.ndarray]:
    toks, masks, idxs, wdls = zip(*buf)
    n = len(toks)
    return {
        "tokens": np.stack(toks), "mask": np.stack(masks),
        "pol_idx": np.stack(idxs), "wdl": np.stack(wdls),
        "aux_idx": np.zeros(n, dtype=np.int64),
        "aux_mask": np.stack(masks),
        "aux_w": np.zeros(n, dtype=np.float32),
    }


def _hf_push(path: Path, repo: str, name: str) -> None:
    """Non-blocking checkpoint push to the HF repo (churn rule §5).

    Runs in a daemon thread: the 105 MB upload must never stall the
    training loop (it did, ~25-60 s per push, user-noticed 2026-09-25).
    Safe against concurrent saves: Trainer.save is atomic (tmp+rename),
    so an in-flight upload reads a stable inode even if latest.pt is
    replaced underneath it. One upload at a time (lock); later pushes
    coalesce (the newest file wins on the next call)."""
    import threading
    with _hf_push._lock:
        if _hf_push._busy:
            return                  # previous upload still running — skip;
        _hf_push._busy = True       # the next save pushes the newer state
    def _work():
        try:
            from huggingface_hub import HfApi
            token = os.environ.get("HF_TOKEN")
            if not token:
                g = Path(".env")
                if g.exists():
                    token = next((l.split("=", 1)[1].strip() for l in
                                  g.read_text().splitlines()
                                  if l.startswith("HF_TOKEN=")), None)
            api = HfApi(token=token)
            api.create_repo(repo, private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                            repo_id=repo)
            print(f"  pushed {name} -> hf:{repo}")
        except Exception as exc:            # noqa: BLE001 — never kill the run
            print(f"  WARN: hf push failed ({exc}); continuing (checkpoint is "
                  f"still on-disk)")
        finally:
            with _hf_push._lock:
                _hf_push._busy = False
    threading.Thread(target=_work, daemon=True).start()


_hf_push._lock = threading.Lock()
_hf_push._busy = False


def cmd_train(args) -> None:
    games = load_pack(args.pack)
    n_val_games = max(1, int(len(games) * args.val_frac))
    if n_val_games >= len(games):        # tiny packs: overlap, never crash
        n_val_games = max(1, len(games) // 2)
    val_games, train_games = games[-n_val_games:], games[:-n_val_games]
    print(f"pack: {len(games)} games ({sum(len(g[1]) for g in games)} "
          f"positions); train {len(train_games)}, val {len(val_games)}")

    net_cfg = (NetConfig(n_layers=2, d_model=64, n_heads=4, d_ff=128,
                         aux_reply=False)
               if args.net == "small" else NetConfig())
    model = ChessTransformer(net_cfg)
    tcfg = TrainConfig(
        lr=args.lr, batch_size=args.batch_size,
        policy_weight=args.policy_weight, value_weight=1.0,
        aux_reply_weight=args.aux_weight, bf16=args.bf16, compile=args.compile,
        device=args.device, seed=args.seed,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "latest.pt"

    if args.resume and ckpt.exists():
        trainer = Trainer.load(ckpt)      # model + Adam moments round-trip
        trainer.cfg = tcfg                # allow LR/batch/bf16/compile tweaks
        for group in trainer.opt.param_groups:
            group["lr"] = tcfg.lr         # keep moments, retarget LR only
        trainer.model.to(trainer.device)
        trainer._maybe_compile()          # the cfg swap must re-wrap (2c bug)
        print(f"resumed from {ckpt} at step {trainer.steps}")
    else:
        trainer = Trainer(model, tcfg)

    sampler = PositionSampler(train_games, seed=args.seed,
                              n_workers=args.n_workers)
    val_sampler = PositionSampler(val_games, seed=args.seed + 7)
    val_picks = val_sampler.fixed_positions(args.val_positions)

    if args.prefetch:
        from chess_ai.supervised.dataset import BatchPrefetcher
        feeder: object = BatchPrefetcher(sampler, args.batch_size)
        next_batch = feeder.next
    else:
        next_batch = lambda: sampler.sample_batch(args.batch_size)  # noqa: E731

    log_path = out_dir / "log.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")
    t0 = time.time()
    stop = trainer.steps + args.steps
    while trainer.steps < stop:
        batch = tensors_to_torch(next_batch(), trainer.device)
        m = trainer.train_step(batch)

        if (trainer.steps + 1) % args.log_every == 0:
            rate = (time.time() - t0) / max(1, trainer.steps - (stop - args.steps))
            print(f"step {trainer.steps:6d}  loss {m['loss']:.4f}  "
                  f"value {m['value']:.4f} (acc {m['value_acc']:.3f})  "
                  f"policy-ctrl {m['policy_acc']:.3f}  "
                  f"({rate:.2f} s/step)")
            log_f.write(json.dumps({"step": trainer.steps, **m}) + "\n")
            log_f.flush()

        if trainer.steps % args.val_every == 0 or trainer.steps == stop - 1:
            vm = evaluate_value(trainer, val_games, val_picks,
                                args.batch_size)
            print(f"  VAL step {trainer.steps}: acc {vm['value_acc']:.4f} "
                  f"(prior {vm['prior_acc']:.4f})  nll {vm['value_nll']:.4f} "
                  f"(prior {vm['prior_nll']:.4f})")
            log_f.write(json.dumps({"step": trainer.steps, "val": vm}) + "\n")
            log_f.flush()

        if trainer.steps % args.save_every == 0 or trainer.steps == stop - 1:
            trainer.save(ckpt)
            if args.hf_repo and (trainer.steps % args.push_every == 0
                                 or trainer.steps == stop - 1):
                _hf_push(ckpt, args.hf_repo, f"{out_dir.name}/latest.pt")

    trainer.save(out_dir / "final.pt")
    if args.hf_repo:
        _hf_push(out_dir / "final.pt", args.hf_repo, f"{out_dir.name}/final.pt")
    log_f.close()
    print(f"final checkpoint: {out_dir / 'final.pt'}")


@torch.no_grad()
def evaluate_value(trainer: Trainer, val_games: List[Game],
                   picks: List[Tuple[int, int]], batch: int = 512) -> Dict[str, float]:
    """Held-out metrics vs the class prior (the bar to beat).

    Value: accuracy + NLL vs prior. Policy (side track round 3+): top-1
    accuracy of the played move among LEGAL moves — the imitation
    baseline is 1/avg-legal ≈ 3%; human-move prediction from 1600+ Elo
    games lands ~25-35% for a net this size."""
    trainer.model.eval()
    n = correct = 0
    nll = 0.0
    pol_correct = 0
    counts = np.zeros(3, dtype=np.int64)        # W/D/L label counts
    import torch.nn.functional as F
    for tensors in _val_batches(val_games, picks, batch):
        t = tensors_to_torch(tensors, trainer.device)
        out = trainer.model(t["tokens"], t["mask"])
        wdl = out["wdl"]
        logp = F.log_softmax(wdl, dim=-1)
        lab = t["wdl"].argmax(-1)
        counts += np.bincount(lab.cpu().numpy(), minlength=3)
        correct += (wdl.argmax(-1) == lab).sum().item()
        nll -= (logp * t["wdl"]).sum().item()
        pol_correct += (out["policy_logp"].argmax(-1) == t["pol_idx"]).sum().item()
        n += lab.numel()
    trainer.model.train()
    prior = counts / counts.sum()
    return {
        "value_acc": correct / n,
        "value_nll": nll / n,
        "prior_acc": float(prior.max()),
        "prior_nll": float(-(prior * np.log(np.maximum(prior, 1e-12))).sum()),
        "label_prior": prior.round(3).tolist(),
        "policy_acc": pol_correct / n,
        "n": n,
    }


def cmd_eval(args) -> None:
    trainer = Trainer.load(args.ckpt)
    games = load_pack(args.pack)
    n_val_games = max(1, int(len(games) * args.val_frac))
    val_games = games[-n_val_games:]
    sampler = PositionSampler(val_games, seed=args.seed + 7)
    picks = sampler.fixed_positions(args.n_positions)
    m = evaluate_value(trainer, val_games, picks, 512)
    print(json.dumps(m, indent=2))
    edge = m["value_acc"] - m["prior_acc"]
    print(f"lift over prior: {edge:+.4f}")


def cmd_match(args) -> None:
    from chess_ai.match import play_match
    from chess_ai.strategies import GreedyMaterialStrategy, RandomStrategy
    from chess_ai.supervised.play import PolicyStrategy, ValueStrategy

    def load_player(kind: str):
        if kind == "value":
            return ValueStrategy(args.ckpt, device=args.device)
        if kind == "policy":
            return PolicyStrategy(args.ckpt, device=args.device)
        if kind == "random":
            return RandomStrategy(seed=args.seed)
        return GreedyMaterialStrategy(seed=args.seed)

    me = load_player(args.player)
    opp = load_player(args.opponent)
    score = play_match(me, opp, n_games=args.n_games)
    plies = [g.plies for g in score["games"]]
    print(f"{args.player}-net vs {args.opponent}: "
          f"W {score['a_wins']}  L {score['b_wins']}  D {score['draws']}  "
          f"of {args.n_games}  (avg {sum(plies)/len(plies):.0f} plies)")
    for i, g in enumerate(score["games"][:5]):
        print(f"  game {i}: {g.describe()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--month", required=True, help="e.g. 2013-01")
    d.add_argument("--out", default=None)
    d.set_defaults(fn=cmd_download)

    d = sub.add_parser("preprocess")
    d.add_argument("--pgn", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--workers", type=int, default=1,
                   help=">1 = parallel pack build (fastprep). QUARANTINED "
                        "default 1: fastprep starved workers on the 4.5 GB "
                        "node dump (8 games/shard, 2026-09-25) — pending a "
                        "real fix + big-file test; serial is trusted")
    d.add_argument("--min-elo", type=int, default=1600)
    d.add_argument("--min-base-sec", type=int, default=180)
    d.add_argument("--min-plies", type=int, default=10)
    d.add_argument("--max-plies", type=int, default=300)
    d.add_argument("--drop-time-forfeits", action="store_true")
    d.add_argument("--max-games", type=int, default=0)
    d.add_argument("--max-positions", type=int, default=0)
    d.set_defaults(fn=cmd_preprocess)

    d = sub.add_parser("train")
    d.add_argument("--pack", required=True)
    d.add_argument("--out", default="runs/value_pilot")
    d.add_argument("--steps", type=int, default=2000)
    d.add_argument("--batch-size", type=int, default=256)
    d.add_argument("--lr", type=float, default=3e-4)
    d.add_argument("--device", default="auto")
    d.add_argument("--net", choices=["small", "full"], default="full")
    d.add_argument("--policy-weight", type=float, default=0.0)
    d.add_argument("--aux-weight", type=float, default=0.0)
    d.add_argument("--bf16", action="store_true",
                   help="CUDA autocast (perf A/B; no-op on MPS/CPU)")
    d.add_argument("--compile", action="store_true",
                   help="torch.compile reduce-overhead (CUDA; first steps pay "
                        "a compile warm-up)")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--n-workers", type=int, default=2)
    d.add_argument("--no-prefetch", dest="prefetch", action="store_false")
    d.add_argument("--val-frac", type=float, default=0.05)
    d.add_argument("--val-positions", type=int, default=8192)
    d.add_argument("--val-every", type=int, default=250)
    d.add_argument("--save-every", type=int, default=250)
    d.add_argument("--push-every", type=int, default=1000,
                   help="HF upload cadence (non-blocking; local latest.pt "
                        "still saves every --save-every)")
    d.add_argument("--log-every", type=int, default=25)
    d.add_argument("--resume", action="store_true")
    d.add_argument("--hf-repo", default=None)
    d.set_defaults(fn=cmd_train)

    d = sub.add_parser("eval")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--pack", required=True)
    d.add_argument("--n-positions", type=int, default=20000)
    d.add_argument("--val-frac", type=float, default=0.05)
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_eval)

    d = sub.add_parser("match")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--player", choices=["value", "policy"], default="value",
                   help="how the checkpoint plays: 1-ply value greed vs "
                        "policy imitation (no search)")
    d.add_argument("--opponent",
                   choices=["random", "greedy", "value", "policy"],
                   default="greedy",
                   help="value/policy = the SAME checkpoint, other head — "
                        "the no-search self-demo")
    d.add_argument("--n-games", type=int, default=10)
    d.add_argument("--device", default="auto")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_match)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
