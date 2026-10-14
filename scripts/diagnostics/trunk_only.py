#!/usr/bin/env python3
"""
probe_labels/trunk_only.py — TEMPORARY diagnostic (not pipeline code).

Builds the transplant the head_swap.py helper CANNOT express:
    FINE-TUNE trunk (input_proj + blocks + ln_f)  +  BASE policy + BASE value head

Why it is the missing cell. There are only two swappable pieces in the
state dict — the head tensors (policy.*, value.*) and everything else (the
trunk). So the full 2x2 is:

  trunk   policy   value      = what                      measured score vs base
  ------------------------------------------------------------------------------
  base    base     base       base                       0.500 (control)
  base    base     ft         value head damage          0.637
  base    ft       base       policy head damage         0.550
  ft      base     base       trunk/feature damage        <- THIS FILE
  ft      ft       ft         the real fine-tune         0.158

Note head_swap.py's `build(dst, policy_from, value_from)` always starts from
the base state dict, so its (base, base) entry IS the base net — the fourth
row above cannot be produced by it and needs this script.

What the row means:
  * near 0.5  -> the trunk still reads the same features the base heads
                 expect; the damage is confined to the head weights, so the
                 repair is a recipe fix (lr / epochs / anchor / head freeze).
  * collapses -> the shared representation itself was re-purposed, and no
                 amount of head fixing recovers it: the data/targets are
                 what moved the features, i.e. a data-side problem.

Usage: .venv/bin/python probe_labels/trunk_only.py build
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "runs/gen1.pt"
FT = ROOT / "runs/pcr2-iter1-e2.pt"
OUT = ROOT / "probe_labels"


def is_head(k: str) -> bool:
    return k.startswith(("policy", "value"))


def main() -> int:
    base = torch.load(BASE, map_location="cpu", weights_only=False)
    ft = torch.load(FT, map_location="cpu", weights_only=False)
    assert set(base["model"]) == set(ft["model"])

    out = {}
    for k, v in ft["model"].items():
        # trunk from the fine-tune, both heads from the base
        out[k] = base["model"][k].clone() if is_head(k) else v.clone()

    # self-checks: the heads must be bit-identical to the base, and the
    # trunk must be bit-identical to the fine-tune, or the transplant is
    # measuring something other than what it claims.
    heads_same = all(torch.equal(out[k], base["model"][k])
                     for k in out if is_head(k))
    trunk_same = all(torch.equal(out[k], ft["model"][k])
                     for k in out if not is_head(k))
    trunk_moved = sum(1 for k in out if not is_head(k)
                      and not torch.equal(out[k], base["model"][k]))
    n_trunk = sum(1 for k in out if not is_head(k))
    print(f"  heads identical to base:      {heads_same}")
    print(f"  trunk identical to fine-tune: {trunk_same}")
    print(f"  trunk tensors actually differing from base: "
          f"{trunk_moved}/{n_trunk}")
    if not (heads_same and trunk_same and trunk_moved > 0):
        return 1

    payload = dict(base)
    payload["model"] = out
    payload["head_swap"] = {"trunk": "ft", "policy": "base", "value": "base"}
    dst = OUT / "hs_trunk_only.pt"
    torch.save(payload, dst)
    print(f"  wrote {dst}")
    print("\ngate it, same protocol as the other three rows:")
    print(f"  .venv/bin/python scripts/gate_match.py --ckpt-a {dst.name} "
          f"--ckpt-b runs/gen1.pt --games 20 --a-sims 200 "
          f"--b-sims 200 --book data/noob_3moves.epd --device mps "
          f"--rows 160 --seed 11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
