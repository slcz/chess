#!/usr/bin/env python3
"""
scripts/sp_pusher.py — slice a running self-play unit into ~1000-game
parts and push each to HuggingFace, without touching the generator.

One process runs `engine/build/selfplay` for the whole batch (pool 1024,
uninterrupted — no long-tail waits between units); it appends finished
games to ONE shard (shard-0.spbin, self-delimiting records). This
process watches that file, and whenever ≥ --every new complete games
have appeared (or the run's DONE.json showed up), it cuts a byte-exact
part — header + complete games only — writes its DONE.json, uploads the
part directory to the HF dataset repo, and records progress in
pushed.json (crash-safe: an interrupted pusher resumes from there; the
final part goes out when the run finishes).

    python3 scripts/sp_pusher.py <run_dir> <repo> <prefix> \
        [--every 1000] [--interval 60] [--token-env HF_TOKEN] [--once]

`run_dir` holds shard-0.spbin (+ DONE.json once finished); parts land in
<prefix>/part-XXXX under `repo` (a dataset repo; created if missing).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_ai.selfplay.shard import scan_shard  # noqa: E402

_HDR = struct.Struct("<8sIIQ")


def shard_seed(path: Path) -> int:
    with path.open("rb") as f:
        magic, version, _worker, seed = _HDR.unpack(f.read(_HDR.size))
    assert magic == b"SPSHARD1" and version == 1, "not a v1 SPSHARD1 shard"
    return seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("repo", help="HF dataset repo (created if missing)")
    ap.add_argument("prefix", help="path prefix in the repo, e.g. units/sp-42k-d1")
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--interval", type=float, default=60)
    ap.add_argument("--token-env", default="HF_TOKEN")
    ap.add_argument("--once", action="store_true",
                    help="one pass (testing), not a watcher loop")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    token = os.environ.get(args.token_env)
    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", exist_ok=True)

    run_dir = Path(args.run_dir)
    shard = run_dir / "shard-0.spbin"
    state_path = run_dir / "pushed.json"
    state = {"pushed": 0, "parts": 0}
    if state_path.exists():
        state = json.loads(state_path.read_text())

    while True:
        if not shard.exists():
            time.sleep(args.interval)
            continue
        seed = shard_seed(shard)
        games = list(scan_shard(shard))
        done = (run_dir / "DONE.json").exists()
        total, new = len(games), len(games) - state["pushed"]
        if new >= args.every or (done and new > 0):
            k = state["parts"]
            part = run_dir / f"part-{k:04d}"
            part.mkdir(exist_ok=True)
            first_start = games[state["pushed"]][1]
            end = games[-1][2]
            with shard.open("rb") as src, (part / "shard-0.spbin").open("wb") as dst:
                src.seek(0)
                dst.write(src.read(_HDR.size))        # header (worker/seed)
                src.seek(first_start)
                dst.write(src.read(end - first_start))
            (part / "DONE.json").write_text(json.dumps(
                {"n_workers": 1, "unit_seed": seed, "games_done": new}))
            api.upload_folder(folder_path=str(part), repo_id=args.repo,
                              repo_type="dataset",
                              path_in_repo=f"{args.prefix}/part-{k:04d}")
            state.update(pushed=total, parts=k + 1)
            state_path.write_text(json.dumps(state))
            print(f"[sp_pusher] part-{k:04d}: {new} games "
                  f"(total {total}, done={done})", flush=True)
        if done and state["pushed"] == total:
            print(f"[sp_pusher] run complete: {total} games in "
                  f"{state['parts']} parts", flush=True)
            return
        if args.once:
            print(f"[sp_pusher] once: pushed {state['pushed']}/{total}", flush=True)
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
