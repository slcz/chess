#!/usr/bin/env python3
"""
scripts/sp_progress.py — watch an in-flight self-play round from the Mac:
walk the HF dataset repo under a run prefix (units/<run>-i*/part-*/DONE.json),
sum each instance's pushed games, and compare the total against the round
target. Works while instances are still generating (parts land every
--every 1000 games) and after they stop.

    python3 scripts/sp_progress.py --run sp-42k-pcr [--target 25000] [--watch 300]

Uses HF_TOKEN from the environment or ./.env (read-only access).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip() or None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO", ""))
    ap.add_argument("--run", default="sp-42k-pcr",
                    help="run name; the scanned prefix is units/<run>-i*")
    ap.add_argument("--target", type=int, default=25000)
    ap.add_argument("--root", default="units",
                    help="dataset dir holding the units (e.g. "
                         "selfplay-scratch/units for the scratch lineage)")
    ap.add_argument("--watch", type=int, default=0,
                    help="poll every N seconds instead of one report")
    args = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=hf_token())

    unit_re = re.compile(rf"^{re.escape(args.root)}/{re.escape(args.run)}-i(.+?)"
                         r"/part-\d+/DONE\.json$")

    def report() -> bool:
        """Print one progress line set; return True when target is met."""
        per: dict[str, dict] = {}
        try:
            # ROOT listing WITH recursion: prefix listings 404 after a full
            # wipe (no such folder), and a non-recursive root listing only
            # shows the folder entry — never the parts inside it.
            paths = [f.path for f in
                     api.list_repo_tree(args.repo, repo_type="dataset",
                                        recursive=True)
                     if unit_re.match(f.path)]
        except Exception as e:                     # transient HF errors must
            print(f"[progress] repo listing failed: {e}")   # not kill --watch
            return False
        for path in paths:
            inst = unit_re.match(path).group(1)
            local = hf_hub_download(args.repo, path, repo_type="dataset",
                                    token=api.token)
            done = json.loads(Path(local).read_text())
            acc = per.setdefault(inst, {"games": 0, "parts": 0, "seeds": set()})
            acc["games"] += done["games_done"]
            acc["parts"] += 1
            acc["seeds"].add(done["unit_seed"])
        total = sum(a["games"] for a in per.values())
        for inst in sorted(per):
            a = per[inst]
            print(f"  i{inst}: {a['games']:6d} games "
                  f"({a['parts']} parts, seed {sorted(a['seeds'])})")
        flag = " TARGET MET" if total >= args.target else ""
        print(f"[{time.strftime('%H:%M:%S')}] {args.run}: {total}/{args.target} "
              f"games across {len(per)} instance(s){flag}", flush=True)
        return total >= args.target

    if report():
        return
    while args.watch and not report():
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
