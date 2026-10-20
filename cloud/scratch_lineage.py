#!/usr/bin/env python3
"""Scratch-lineage tooling: genesis, approval, and the iteration log.

The lineage lives in the model repo under LINEAGE_DIR (default
selfplay-scratch/) and is driven by one atomic pointer, approved.json:

    {generation, file, sha256, created_at, seed, params, gate}

The self-play container reads that pointer, downloads the checkpoint it
names, and verifies the sha256 before generating — so bumping the pointer
(not copying files) is what "approving a new checkpoint" means.

    scripts/scratch_lineage.py genesis [--seed 0] [--dry-run]
    scripts/scratch_lineage.py approve --ckpt runs/g2-e1.pt --gen 2 \\
        --score 0.53 --wdl 210 20 170 [--games 400] [--adopt] \\
        --train-json '{"batch": 4096, "lr": 2e-4, ...}' \\
        --loss-json '[{"step": 100, "loss": 3.4}, ...]'

approve computes the relative Elo of the new checkpoint against the previous
one from the gate score, appends a YAML record to ITERATIONS.md (local and on
HF), and — with --adopt — uploads the checkpoint FIRST and only then switches
the pointer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODEL_REPO = "${HF_MODEL_REPO:?set the HF model repo, e.g. <owner>/<models>}"
LINEAGE_DIR = "selfplay-scratch"
LOG_NAME = "ITERATIONS.md"


def token() -> str:
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        if line.startswith("HF_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("HF_TOKEN not found (env or .env)")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def elo(score: float, games: int) -> tuple[float, float]:
    """Relative Elo of A vs B from A's match score, and its binomial sigma.

        400*log10(s/(1-s)),  sigma = 400/ln10 / sqrt(n*s*(1-s))

    At 400 games sigma is ~17-18 Elo; our measured gen1-vs-gen1 null control
    (0.500, waves 0.468/0.532) implies a systematic floor nearer 25-30 Elo,
    so treat only |delta| >~ 40 as real.
    """
    s = min(max(score, 1e-6), 1.0 - 1e-6)
    d = 400.0 * math.log10(s / (1.0 - s))
    sigma = 400.0 / (math.log(10.0) * math.sqrt(games * s * (1.0 - s)))
    return d, sigma


def hf():
    from huggingface_hub import HfApi
    return HfApi(token=token())


def read_approved(api) -> dict | None:
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(MODEL_REPO, f"{LINEAGE_DIR}/approved.json",
                            repo_type="model", token=token())
    except Exception:
        return None
    return json.loads(Path(p).read_text())


def cmd_genesis(args) -> int:
    import torch
    from chess_ai.net.model import ChessTransformer, NetConfig
    from chess_ai.train.trainer import Trainer, TrainConfig

    torch.manual_seed(args.seed)
    cfg = NetConfig(pos_scheme="rope2d", policy_head="attention",
                    rope_dirs=4, rope_pairs=2)
    trainer = Trainer(ChessTransformer(cfg),
                      TrainConfig(device="cpu", bf16=False, compile=False,
                                  aux_reply_weight=0.3))
    out = ROOT_SCRATCH / "scratch-gen1.pt"   # never collide with legacy runs/gen1.pt
    trainer.save(out)
    digest = sha256_of(out)
    n = sum(p.numel() for p in trainer.model.parameters())
    print(f"genesis: {out} | params {n} | sha256 {digest[:16]}…")
    if args.dry_run:
        return 0
    api = hf()
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"{LINEAGE_DIR}/gen1.pt",
                    repo_id=MODEL_REPO, repo_type="model")
    approved = {
        "lineage": "scratch",
        "generation": 1,
        "file": "gen1.pt",
        "sha256": digest,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "params": n,
        "gate": None,
    }
    api.upload_file(path_or_fileobj=json.dumps(approved, indent=2).encode(),
                    path_in_repo=f"{LINEAGE_DIR}/approved.json",
                    repo_id=MODEL_REPO, repo_type="model")
    print(f"approved.json -> gen1 (sha256 {digest[:16]}…)")
    return 0


def cmd_approve(args) -> int:
    ckpt = Path(args.ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"no such checkpoint: {ckpt}")
    api = hf()
    prev = read_approved(api)
    delta, sigma = elo(args.score, args.games)
    w, d, l = args.wdl
    assert w + d + l == args.games, (w, d, l, args.games)
    digest = sha256_of(ckpt)
    name = f"gen{args.gen}.pt"

    print(f"gate: score {args.score:.3f} ({w}/{d}/{l}) -> "
          f"Elo {delta:+.0f} ± {sigma:.0f} vs the previous approved net")
    if args.dry_run:
        print("dry run: not uploading")
        return 0

    api.upload_file(path_or_fileobj=str(ckpt), path_in_repo=f"{LINEAGE_DIR}/{name}",
                    repo_id=MODEL_REPO, repo_type="model")

    log_path = REPO_ROOT / LOG_NAME
    log = log_path.read_text() if log_path.exists() else HEADER
    rec = {
        "generation": args.gen,
        "data": json.loads(args.data_json) if args.data_json else None,
        "train": json.loads(args.train_json) if args.train_json else None,
        "loss": json.loads(args.loss_json) if args.loss_json else None,
        "eval": {"games": args.games, "score": args.score,
                 "wdl": [w, d, l],
                 "elo_delta": round(delta, 1), "elo_sigma": round(sigma, 1)},
        "adopted": bool(args.adopt),
        "checkpoint": {"file": name, "sha256": digest},
    }
    import yaml
    block = yaml.safe_dump([rec], sort_keys=False, default_flow_style=False)
    if "```yaml" in log and "\n```" in log:
        head, rest = log.split("```yaml\n", 1)
        body, tail = rest.split("\n```", 1)
        log = (head + "```yaml\n" + body.rstrip("\n") + "\n" + block
               + "```" + tail)
    else:
        log = log.rstrip("\n") + "\n\n```yaml\n" + block + "```\n"
    log_path.write_text(log)
    api.upload_file(path_or_fileobj=log.encode(),
                    path_in_repo=f"{LINEAGE_DIR}/{LOG_NAME}",
                    repo_id=MODEL_REPO, repo_type="model")

    if args.adopt:
        approved = {
            "lineage": "scratch",
            "generation": args.gen,
            "file": name,
            "sha256": digest,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "params": prev.get("params") if prev else None,
            "gate": {"games": args.games, "score": args.score,
                     "wdl": [w, d, l], "elo_delta": round(delta, 1),
                     "vs_generation": prev["generation"] if prev else None},
        }
        # checkpoint uploaded above; switching the pointer is the atomic step
        api.upload_file(path_or_fileobj=json.dumps(approved, indent=2).encode(),
                        path_in_repo=f"{LINEAGE_DIR}/approved.json",
                        repo_id=MODEL_REPO, repo_type="model")
        print(f"ADOPTED gen{args.gen} (pointer moved)")
    else:
        print(f"rejected gen{args.gen}; pointer unchanged")
    return 0


HEADER = """# Scratch lineage iterations

One YAML record per iteration. `elo_delta` is the new checkpoint's relative
strength against the previously approved net, from the 400-game gate score
(200 sims both sides, fixed opening book): `400*log10(s/(1-s))`, sigma by
binomial propagation (~17-18 Elo at 400 games). Our measured null control is
0.500 with waves 0.468/0.532, so treat |delta| below ~40 Elo as noise.

"""

ROOT_SCRATCH = REPO_ROOT / "runs"


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("genesis", help="create + publish the random gen1")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--dry-run", action="store_true")
    g.set_defaults(fn=cmd_genesis)

    a = sub.add_parser("approve", help="log an iteration + optionally adopt it")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--gen", type=int, required=True)
    a.add_argument("--score", type=float, required=True)
    a.add_argument("--wdl", type=int, nargs=3, required=True, metavar=("W", "D", "L"))
    a.add_argument("--games", type=int, default=400)
    a.add_argument("--adopt", action="store_true")
    a.add_argument("--data-json", help="games/buffer/manifest summary")
    a.add_argument("--train-json", help="training parameters")
    a.add_argument("--loss-json", help="brief loss history rows")
    a.add_argument("--dry-run", action="store_true")
    a.set_defaults(fn=cmd_approve)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
