"""
scripts/status_render.py — regenerate the run's human-readable STATUS.md
(the in-progress report the 45-min cron refreshes; view it with
`scripts/salad.sh ssh cat /workspace/chess/runs/<out>/STATUS.md`).

    python3 -m scripts.status_render --out runs/eq9_lr2e5
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m scripts.status_render")
    p.add_argument("--out", default="runs/eq9_lr2e5")
    a = p.parse_args(argv)
    out = Path(a.out)

    lines = ["# Chess gated self-play — live status", "",
             f"_updated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_",
             ""]
    log = out / "log.jsonl"
    if log.exists():
        rows = [json.loads(l) for l in log.read_text().splitlines()]
        if rows:
            m = rows[-1]
            t, sp = m["train"], m["sp"]
            lines += ["## Training", "",
                      f"- round **{m['iter']}**, games **{m['games_total']}**, "
                      f"steps {m['steps']} · target {m.get('policy_target', '?')}",
                      f"- policy_acc {t['policy_acc']:.3f} · value_acc "
                      f"{t['value_acc']:.3f} · aux {t['aux']:.3f}",
                      f"- throughput {sp['games_per_hour']:.0f} games/h · "
                      f"plies {sp['mean_plies']:.0f} · draws {sp['draw_rate']:.0%}",
                      ""]
            if len(rows) >= 10:
                lines.append(f"- pol_acc trend (last 10 rounds): "
                             f"{rows[-10]['train']['policy_acc']:.3f} → "
                             f"{t['policy_acc']:.3f}")

    lines += ["## Gated iterations (champion–challenger, threshold 0.60)", ""]
    glog = out / "gated" / "gated_log.jsonl"
    gates = ([json.loads(l) for l in glog.read_text().splitlines()]
             if glog.exists() else [])
    for g in gates:
        lines.append(f"- iteration {g['iteration']}: gate score "
                     f"**{g['gate_score']:.3f}** → "
                     f"{'ACCEPTED ✅' if g['accepted'] else 'keep training'}")
    if not gates:
        lines.append("- first gate in flight")

    lines += ["", "## Eval history vs RoPE seed", ""]
    pm = out.parent / "eval" / "policy_match.jsonl"
    if pm.exists():
        for l in pm.read_text().splitlines()[-3:]:
            e = json.loads(l)
            lines.append(f"- {e.get('a', '?')} (no-search): "
                         f"score {e['score_a']:.3f} ({e['elo_a_minus_b']:+.0f} "
                         f"Elo, {e['games']} games)")
    le = out.parent / "eval" / "ladder_eq9.jsonl"
    if le.exists():
        for l in le.read_text().splitlines()[-2:]:
            e = json.loads(l)
            lines.append(f"- {e.get('a', '?')} (searched n={e.get('n', '?')}): "
                         f"score {e['score_a']:.3f} ({e['elo_a_minus_b']:+.0f} "
                         f"Elo, {e['games']} games)")

    (out / "STATUS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'STATUS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
