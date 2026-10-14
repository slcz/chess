#!/usr/bin/env python3
"""
probe_labels/head_swap.py — TEMPORARY diagnostic (not pipeline code).

The head-to-head says pcr2-iter1-e2 scores 0.158 vs its own base, but the
value calibration said pcr2's value head predicts self-play outcomes BETTER
(ECE 0.241 vs 0.343). Those two facts together mean the regression is not
"the value head got worse at predicting" — so which component actually
carries the damage?

This builds the two hybrids and measures them head-to-head:

  value_head_only   = base trunk+policy, fine-tune's value head
  policy_head_only  = fine-tune trunk+policy, base's value head

Interpretation:
  * if value_head_only also collapses -> the value head is the damage
    (and it is "better calibrated" yet plays worse: over-confident values
    poisoning the Q term of the search).
  * if policy_head_only collapses -> the policy head (or the trunk features
    the policy reads) is the damage, even though its update was small.
  * if neither collapses but the full fine-tune does -> the interaction of
    trunk + both heads, i.e. genuine representation drift, not a single head.

Usage: .venv/bin/python probe_labels/head_swap.py build
       (then run scripts/gate_match.py on the two hybrids it prints)
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "runs/gen1.pt"
FT = ROOT / "runs/pcr2-iter1-e2.pt"
OUT = ROOT / "probe_labels"


def load(p: Path):
    return torch.load(p, map_location="cpu", weights_only=False)


def build(dst_name: str, policy_from: str, value_from: str) -> Path:
    base, ft = load(BASE), load(FT)
    src = {"base": base["model"], "ft": ft["model"]}
    out = {k: v.clone() for k, v in base["model"].items()}
    for k in out:
        if k.startswith("policy"):
            out[k] = src[policy_from][k].clone()
        elif k.startswith("value"):
            out[k] = src[value_from][k].clone()
    payload = dict(base)
    payload["model"] = out
    # provenance, so a stray file in runs/ is self-describing
    payload["head_swap"] = {"policy_from": policy_from,
                            "value_from": value_from,
                            "base": BASE.name, "fine_tune": FT.name}
    dst = OUT / dst_name
    torch.save(payload, dst)
    n = sum(1 for k in out if k.startswith(("policy", "value")))
    print(f"  wrote {dst}  (policy<={policy_from}, value<={value_from}, "
          f"{n} head tensors replaced)")
    return dst


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] != "build":
        print(__doc__)
        return 2
    print("building head-swap hybrids from")
    print(f"  base {BASE.name}\n  ft   {FT.name}")
    a = build("hs_value_only.pt", policy_from="base", value_from="ft")
    b = build("hs_policy_only.pt", policy_from="ft", value_from="base")
    # sanity: a swap that takes everything from base must equal the base
    check = build("hs_all_base_check.pt", policy_from="base", value_from="base")
    base = load(BASE)["model"]
    got = load(check)["model"]
    same = all(torch.equal(base[k], got[k]) for k in base)
    print(f"  self-check 'all from base' equals the base state dict: {same}")
    if not same:
        return 1
    print("\nnow gate both, same protocol as the base-vs-pcr2 run:")
    print(f"  .venv/bin/python scripts/gate_match.py --ckpt-a {a.name} "
          f"--ckpt-b runs/gen1.pt --games 20 --a-sims 200 "
          f"--b-sims 200 --book data/noob_3moves.epd --device mps "
          f"--rows 160 --seed 11")
    print(f"  .venv/bin/python scripts/gate_match.py --ckpt-a {b.name} "
          f"--ckpt-b runs/gen1.pt --games 20 --a-sims 200 "
          f"--b-sims 200 --book data/noob_3moves.epd --device mps "
          f"--rows 160 --seed 11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
