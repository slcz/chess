#!/usr/bin/env python3
"""
scripts/overfit_check.py — end-to-end memorization probe over a handful of
self-play games: drive train_step to ~0 loss on a tiny sample set, then
re-score the same boards through the forward pass and verify every head
agrees with its label.

    python3 scripts/overfit_check.py --unit /tmp/spbench \
        --base-ckpt runs/policy_value_net_42k.pt [--steps 400] [--save ckpt.pt]

Expected end state (the accept criteria):
  value CE        -> ~0      (one-hot target, learnable exactly)
  pol_idx CE      -> ~0      (sparse played-move target, learnable exactly)
  dense pi CE     -> H(pi)   (the visit distribution's own entropy — the
                              FLOOR; a distributional target cannot be
                              beaten, only matched)
  value argmax    -> 100% of samples
Convention errors (frame flips, slot mismatches) show up as a CE that
plateaus ABOVE these floors — contradictory targets cannot be memorized.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_ai.selfplay.shard import read_unit  # noqa: E402
from chess_ai.train.selfplay_data import selfplay_sample, stack_selfplay  # noqa: E402
from chess_ai.train.trainer import Trainer, tensors_to_torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unit", default="/tmp/spbench")
    ap.add_argument("--base-ckpt", default="runs/policy_value_net_42k.pt")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--save", default=None, help="save the overfit ckpt")
    args = ap.parse_args()

    games = read_unit(args.unit)
    picked = []
    for want in (1.0, 0.0, 0.5):          # one game per outcome class
        picked.append(next(g for g in games if g.z_white == want))
    print(f"overfit set: {len(picked)} games "
          f"({[g.n_plies for g in picked]} plies, outcomes "
          f"{[g.z_white for g in picked]})", flush=True)

    samples = [s for g in picked for s in
               (selfplay_sample(g, p) for p in range(g.n_plies))]
    batch = tensors_to_torch(stack_selfplay(samples),
                             torch.device("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"samples: {len(samples)} (full-batch steps)", flush=True)

    trainer = Trainer.load(args.base_ckpt)
    trainer.model.to(batch["tokens"].device).train()
    for grp in trainer.opt.param_groups:
        grp["lr"] = args.lr

    tr = trainer.train_step(batch)        # step 0: the starting point
    print(f"step   0: loss {tr['loss']:.4f} policy {tr['policy']:.4f} "
          f"value {tr['value']:.4f}", flush=True)
    for step in range(1, args.steps + 1):
        tr = trainer.train_step(batch)
        if step % 50 == 0:
            print(f"step {step:4d}: loss {tr['loss']:.4f} "
                  f"policy {tr['policy']:.4f} value {tr['value']:.4f}",
                  flush=True)

    print("\n== final agreement (same boards, forward pass) ==")
    with torch.no_grad():
        out = trainer.model(batch["tokens"], batch["mask"], batch["aux_mask"])
        wdl = torch.softmax(out["wdl"], -1)
        vacc = (out["wdl"].argmax(-1) == batch["wdl"].argmax(-1)).float().mean().item()
        vce = -(torch.log_softmax(out["wdl"], -1) * batch["wdl"]).sum(-1).mean().item()
        played_p = torch.softmax(out["policy_logp"].masked_fill(
            batch["mask"].logical_not(), -1e9), -1)
        played_p = played_p.gather(1, batch["pol_idx"][:, None]).squeeze(1)
        slot_logp = out["policy_logp"].gather(1, batch["pi_slots"])
        pi_ce = -(batch["pi_probs"] * slot_logp).sum(-1)
        pi_H = -(batch["pi_probs"] * batch["pi_probs"].clamp_min(1e-12).log()).sum(-1)
    print(f"value CE {vce:.5f}   value argmax agreement {vacc * 100:.1f}%")
    print(f"played-move prob under policy: min {played_p.min():.4f} "
          f"mean {played_p.mean():.4f}")
    print(f"dense pi CE {pi_ce.mean().item():.4f} vs target entropy "
          f"{pi_H.mean().item():.4f} (floor — gap {(pi_ce.mean() - pi_H.mean()).item():+.4f})")

    if args.save:
        trainer.save(args.save)
        print(f"overfit ckpt saved: {args.save}", flush=True)


if __name__ == "__main__":
    main()
