"""
scripts/gated_run.py — champion–challenger gated self-play (the Phase-5
iteration structure, user-directed 2026-10-01).

Replaces the monolithic fixed-round run with an evaluated iteration cycle:

    repeat:
      1. GENERATE  rounds_per_iter × games_per_iter self-play games,
                   training as usual (the loop resumes from latest.pt)
      2. EVALUATE  gate-games searched games (paired colors, bf16 both
                   seats): challenger (latest.pt) vs CHAMPION
      3. GATE      score ≥ threshold → challenger is promoted: it becomes
                   the new champion and the next iteration trains from it.
                   Otherwise: keep training (another block) — the champion
                   is unchanged.

    The gate's search family is selectable (--gate-search): the default
    Gumbel gate measures "calibration under full σ-commitment" — the
    2026-10-02 finding is that this DECOUPLES from strength (a net can
    gain +83 Elo raw / +70 PUCT while its Gumbel gate slides). The
    --gate-search az PUCT gate (the C AZ-PUCT walk, n = PUCT sims) tracks
    policy quality instead — "training in Gumbel, evaluating in PUCT".

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
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="decoupled AdamW weight decay for training "
                        "(AGZ Eq. 1 / KataGo c_L2 precedent; 0 = off)")
    p.add_argument("--outlook-downweight", action="store_true",
                   help="KataGo App. D: downweight self-play samples in "
                        "hopeless positions (dilutes saturated ±1 labels)")
    p.add_argument("--c-visit", type=float, default=50.0,
                   help="self-play σ constant (data-generation side; the "
                        "gate/ladder keeps its own --n-eval-side knob)")
    p.add_argument("--policy-target", default="eq9")
    p.add_argument("--n", type=int, default=16, help="self-play sims/move")
    p.add_argument("--max-plies", type=int, default=256,
                   help="adjudicate games as draws beyond this (bounds the "
                        "shuffler tail: 1.8%% of games were >196 plies, up "
                        "to 350, costing ~13x a median game)")
    p.add_argument("--gate-games", type=int, default=32)
    p.add_argument("--gate-threshold", type=float, default=0.60)
    p.add_argument("--n-eval", type=int, default=160, help="gate sims/move")
    p.add_argument("--gate-search", choices=["gumbel", "az"],
                   default="gumbel",
                   help="gate selector: the Gumbel search (default, "
                        "calibration-under-commitment) or the C AZ-PUCT "
                        "walk (az_eval.py's; --n-eval = PUCT sims/move, "
                        "the fleet-measured strength instrument)")
    p.add_argument("--c-puct", type=float, default=1.25,
                   help="az gate: PUCT exploration constant")
    p.add_argument("--q-view", choices=["child", "parent"],
                   default="child",
                   help="az gate: selection Q's sign — child = verbatim "
                        "az_eval.py (the fleet-measured 0.6125 config); "
                        "parent = AlphaGo paper's Q(s,a)")
    p.add_argument("--gate-max-plies", type=int, default=128,
                   help="gate games adjudicate as draws beyond this ply "
                        "(both seats symmetric, self-play's own "
                        "convention; 0 = play to natural termination). "
                        "Gate 1 measured: 42 min wall with the last "
                        "games at 457-459 plies — the draw-tail sets "
                        "the gate's clock, not the median game")
    p.add_argument("--iterations", type=int, default=20,
                   help="max gated iterations")
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--hf-repo", default="",
                   help="push round artifacts + gated state to this HF repo "
                        "(churn rule: nothing durable lives on the instance)")
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
        prev = state.get("gate_search", "gumbel")
        if prev != a.gate_search:
            print(f"WARNING: gate instrument changed mid-run "
                  f"({prev} -> {a.gate_search}); the champion was gated "
                  f"under '{prev}' — gate scores are not comparable "
                  f"across the break")
    else:
        shutil.copy(a.champion_init, champion)
        state = {"iteration": 0, "gates": 0, "accepted": 0,
                 "threshold": a.gate_threshold,
                 "gate_search": a.gate_search}
        state_path.write_text(json.dumps(state, indent=2))
        print(f"new gated run ({a.gate_search} gate); champion seeded "
              f"from {a.champion_init}")

    log_path = gated / "gated_log.jsonl"
    for it in range(state["iteration"] + 1, a.iterations + 1):
        t0 = time.time()
        # 1. GENERATE + TRAIN: the loop resumes from latest.pt and appends
        #    its own log.jsonl (cron-compatible). Resignation is OFF (the
        #    recorded finding: resign_value -0.90 collapses self-play with
        #    the supervised seed — loop-induced value sharpening), and the
        #    gate matches hardcode it off too.
        r = run(["python3", "-m", "chess_ai.train.selfplay_loop",
                 "--out", a.out, "--seed-ckpt", str(champion),
                 "--policy-target", a.policy_target, "--lr", str(a.lr),
                 "--train-steps", str(a.train_steps),
                 "--games-per-iter", str(a.games_per_iter),
                 "--iters", str(a.rounds_per_iter), "--n", str(a.n),
                 "--device", a.device, "--eval-device", a.device,
                 "--resign", "-1",
                 "--max-plies", str(a.max_plies),
                 *(["--eval-bf16", "--eval-compile", "--train-bf16"]
                   if a.device == "cuda" else []),
                 *(["--weight-decay", str(a.weight_decay)]
                   if a.weight_decay else []),
                 *(["--outlook-downweight"] if a.outlook_downweight else []),
                 *(["--c-visit", str(a.c_visit)] if a.c_visit != 50.0 else []),
                 *(["--hf-repo", a.hf_repo] if a.hf_repo else []),
                 "--save-every", str(a.rounds_per_iter)],
                capture_output=True, text=True)
        if r.returncode != 0:
            print(f"training loop failed (rc={r.returncode}); tail:")
            print(r.stdout[-2000:], r.stderr[-2000:])
            return 1
        latest = out / "latest.pt"
        games_total = json.loads(
            (out / "log.jsonl").read_text().splitlines()[-1])["games_total"]

        # 2. EVALUATE vs the champion (searched) — all gate games run
        #    CONCURRENTLY (workers = gate_games): the batched service then
        #    sees B≈games forwards instead of B≈6 — measured ~8x faster.
        #    bf16+compile both seats (the net trained under bf16) — CUDA
        #    only: Mac smokes run mps/cpu eager so dev gates stay cheap.
        #    The ladder prints one progress line per finished game; its
        #    stdout streams to eval_live.log so the gate is watchable live
        #    (capture_output would swallow it until the iteration ends).
        eval_live = gated / "eval_live.log"
        with eval_live.open("w") as lf:
            r = run(["python3", "-m", "chess_ai.selfplay.ladder",
                     "--a", str(latest), "--b", str(champion),
                     "--seeds", str(a.gate_games // 2), "--n", str(a.n_eval),
                     "--search", a.gate_search,
                     *(["--c-puct", str(a.c_puct), "--q-view", a.q_view]
                       if a.gate_search == "az" else []),
                     *(["--max-plies", str(a.gate_max_plies)]
                       if a.gate_max_plies and a.gate_max_plies > 0 else []),
                     "--device", a.device,
                     *(["--bf16", "--compile"] if a.device == "cuda" else []),
                     "--workers", str(a.gate_games),
                     "--jsonl", str(gated / "evals.jsonl")],
                    stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"gate match failed (rc={r.returncode}); {eval_live} tail:")
            print(eval_live.read_text()[-2000:])
            return 1
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
                 "gate_search": a.gate_search, "n_eval": a.n_eval,
                 "gate_max_plies": a.gate_max_plies,
                 **({"c_puct": a.c_puct, "q_view": a.q_view}
                    if a.gate_search == "az" else {}),
                 "accepted": accepted, "train_s": round(time.time() - t0)}
        with log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if a.hf_repo:
            # Churn rule (PLAN §5): the gated state itself must leave the
            # instance after every gate — champion, lineage, eval history.
            from chess_ai.train.selfplay_loop import hf_push_async
            hf_push_async(
                [(champion, f"{out.name}/gated/champion.pt"),
                 (state_path, f"{out.name}/gated/state.json"),
                 (log_path, f"{out.name}/gated/gated_log.jsonl"),
                 (gated / "evals.jsonl", f"{out.name}/gated/evals.jsonl"),
                 (eval_live, f"{out.name}/gated/eval_live.log")],
                a.hf_repo, log=lambda *_: None,
                message=f"gated iteration {it}: score {score:.3f}"
                        f"{' (accepted)' if accepted else ''}")
        print(f"== iteration {it}: score {score:.3f} vs champion "
              f"(threshold {a.gate_threshold}) -> "
              f"{'ACCEPTED' if accepted else 'keep training'} "
              f"({time.time() - t0:.0f} s)")
    from chess_ai.train.selfplay_loop import hf_push_drain
    hf_push_drain()   # the final iteration's gated push must survive exit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
