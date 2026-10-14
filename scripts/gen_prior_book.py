#!/usr/bin/env python3
"""gen_prior_book.py — recreate the net-prior sampled opening book (the
old mux_match protocol): `lines` games of `plies` moves sampled
proportional to the raw policy priors of <ckpt>, written as FEN positions
(one per line) for gate_match.py --book.

    python3 scripts/gen_prior_book.py --ckpt runs/gen1.pt \
        --lines 50 --plies 8 --seed 42 --out data/prior_book_42k.epd

Deterministic given (ckpt, seed); the file is the protocol, so both
processes of a gate and any later rerun share the identical book.
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chess_ai.engine import Board
from chess_ai.net.features import uci_to_index
from chess_ai.train.trainer import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lines", type=int, default=50)
    ap.add_argument("--plies", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = Trainer.load(args.ckpt).model.eval()
    rng = random.Random(args.seed)
    dev = next(model.parameters()).device   # the ckpt's device

    lines = []
    resampled = 0
    for k in range(args.lines):
        while True:                      # resample lines that end early
            b = Board()
            ok = True
            for _ in range(args.plies):
                legal = b.legal_moves
                if not legal:            # prior sampling walked into a
                    ok = False           # terminal (noob quick mates)
                    break
                x = torch.from_numpy(b.encode()).unsqueeze(0).to(dev)
                mask = torch.zeros(1, 4240, dtype=torch.bool)
                side = 0 if b.fen.split()[1] == "w" else 1
                for u in legal:
                    mask[0, uci_to_index(u, side)] = True
                with torch.inference_mode():
                    logp = model(x.to(dev), mask.to(dev))["policy_logp"][0]
                prior = torch.softmax(logp[mask[0].bool()], dim=-1).cpu()
                pick = rng.choices(legal, weights=prior.tolist(), k=1)[0]
                b.push(pick)
            if ok:
                break
            resampled += 1
        lines.append(b.fen)
        print(f"book[{k:02d}] {b.fen}", flush=True)
    if resampled:
        print(f"({resampled} lines resampled — terminal mid-line)",
              flush=True)

    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}: {len(lines)} positions")


if __name__ == "__main__":
    main()
