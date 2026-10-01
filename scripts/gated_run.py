"""
scripts/gated_run.py — champion–challenger gated self-play (the Phase-5
iteration structure, user-directed 2026-10-01).

Replaces the monolithic fixed-round run with an evaluated iteration cycle:

    repeat:
      1. GENERATE  rounds_per_iter × games_per_iter self-play games,
                   training as usual (the loop resumes from latest.pt)
      2. EVALUATE  32 searched games (n=160, paired colors, bf16 both
                   seats): challenger (latest.pt) vs CHAMPION
      3. GATE      score ≥ threshold → challenger is promoted: it becomes
                   the new champion and the next iteration trains from it.
                   Otherwise: keep training (another block) — the champion
                   is unchanged.

False accepts are self-correcting (the next gate re-tests the lucky
champion against its own continuation); false rejects only cost time, so
the threshold can be conservative. State is files-only (champion.pt +
state.json), so the driver is crash-safe: relaunch and it continues.

    python3 -m scripts.gated_run --out runs/eq9_lr2e5 \
        --champion-init runs/policy_value_net_rope_25k.pt \
        --policy-target eq9 --lr 2e-5 --train-steps 768 \
        --gate-games 32 --gate-threshold 0.60 --iterations 20
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=REPO, **kw)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scripts.gated_run")
    p.add_argument("--out", required=True,
                   help="the training loop's out dir (latest.pt lives here)")
    p.add_argument("--champion-init", required=True,
                   help="checkpoint the first champion is seeded from")
    p.add_argument("--rounds-per-iter", type=int, default=10)
    p.add_argument("--games-per-iter", type=int, default=1024)
    p.add_argument("--train-steps", type=int, default=768)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--policy-target", default="eq9")
    p.add_argument("--n", type=int, default=16, help="self-play sims/move")
    p.add_argument("--gate-games", type=int, default=32)
    p.add_argument("--gate-threshold", type=float, default=0.60)
    p.add_argument("--n-eval", type=int, default=160, help="gate sims/move")
    p.add_argument("--iterations", type=int, default=20,
                   help="max gated iterations")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true", default=True)
    a = p.parse_args(argv)

    out = Path(a.out)
    gated = out / "gated"
    gated.mkdir(parents=True, exist_ok=True)
    champion = gated / "champion.pt"
    state_path = gated / "state.json"
    if champion.exists() and state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"resuming gated run: iteration {state['iteration']}, "
              f"{state['accepted']} accepted of {state['gates']}")
    else:
        shutil.copy(a.champion_init, champion)
        state = {"iteration": 0, "gates": 0, "accepted": 0,
                 "threshold": a.gate_threshold}
        state_path.write_text(json.dumps(state, indent=2))
        print(f"new gated run; champion seeded from {a.champion_init}")

    log_path = gated / "gated_log.jsonl"
    for it in range(state["iteration"] + 1, a.iterations + 1):
        t0 = time.time()
        # 1. GENERATE + TRAIN: the loop resumes from latest.pt and appends
        #    its own log.jsonl (cron-compatible).
        r = run(["python3", "-m", "chess_ai.train.selfplay_loop",
                 "--out", a.out, "--seed-ckpt", str(champion),
                 "--policy-target", a.policy_target, "--lr", str(a.lr),
                 "--train-steps", str(a.train_steps),
                 "--games-per-iter", str(a.games_per_iter),
                 "--iters", str(a.rounds_per_iter), "--n", str(a.n),
                 "--device", a.device, "--eval-device", a.device,
                 "--eval-bf16", "--eval-compile", "--train-bf16",
                 "--save-every", str(a.rounds_per_iter)],
                capture_output=True, text=True)
        if r.returncode != 0:
            print(f"training loop failed (rc={r.returncode}); tail:")
            print(r.stdout[-2000:], r.stderr[-2000:])
            return 1
        latest = out / "latest.pt"
        games_total = json.loads(
            (out / "log.jsonl").read_text().splitlines()[-1])["games_total"]

        # 2. EVALUATE vs the champion (searched) — 32 games, bf16 both seats
        r = run(["python3", "-m", "chess_ai.selfplay.ladder",
                 "--a", str(latest), "--b", str(champion),
                 "--seeds", str(a.gate_games // 2), "--n", str(a.n_eval),
                 "--device", a.device, "--bf16", "--workers", "6",
                 "--jsonl", str(gated / "evals.jsonl")],
                capture_output=True, text=True)
        evals = [json.loads(l) for l in
                 (gated / "evals.jsonl").read_text().splitlines()]
        score = evals[-1]["score_a"] if evals else 0.0

        # 3. GATE
        accepted = score >= a.gate_threshold
        if accepted:
            shutil.copy(latest, champion)
        state.update(iteration=it, gates=state["gates"] + 1,
                     accepted=state["accepted"] + int(accepted))
        state_path.write_text(json.dumps(state, indent=2))
        entry = {"iteration": it, "games_total": games_total,
                 "gate_score": score, "threshold": a.gate_threshold,
                 "accepted": accepted, "train_s": round(time.time() - t0)}
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"== iteration {it}: score {score:.3f} vs champion "
              f"(threshold {a.gate_threshold}) -> "
              f"{'ACCEPTED' if accepted else 'keep training'} "
              f"({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
