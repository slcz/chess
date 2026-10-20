#!/usr/bin/env python3
"""
scripts/export_onnx.py — export a checkpoint to ONNX for the web GUI and
run the parity gate (torch vs onnxruntime, real C-encoded positions).

    .venv/bin/python scripts/export_onnx.py runs/gen1.pt
    .venv/bin/python scripts/export_onnx.py runs/gen1.pt --out web/gen1.onnx --parity 128

Without --ckpt a random-init default-config net is exported instead — a
graph smoke test (parity is about the graph, not the weights), useful
before any checkpoint exists locally.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chess_ai.net.export_onnx import export_onnx, parity_gate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ckpt", nargs="?", help="torch checkpoint (Trainer format); "
                    "omit for a random-init graph smoke test")
    p.add_argument("--out", help="output .onnx (default: <ckpt>.onnx)")
    p.add_argument("--parity", type=int, default=64,
                   help="positions for the gate (0 skips it)")
    args = p.parse_args()

    if args.ckpt:
        from chess_ai.train.trainer import Trainer
        model = Trainer.load(args.ckpt, device="cpu").model   # same load path
        out = args.out or str(Path(args.ckpt).with_suffix(".onnx"))  # as nn_worker
    else:
        from chess_ai.net.model import ChessTransformer
        model = ChessTransformer()
        out = args.out or "net.onnx"

    path = export_onnx(model, out)
    print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if args.parity:
        stats = parity_gate(model, path, n_positions=args.parity)
        print(f"parity: {stats['positions']} positions, "
              f"max |Δlogp| {stats['max_d_logp_legal']:.2e}, "
              f"max |Δwdl| {stats['max_d_wdl']:.2e}, "
              f"argmax agree {stats['argmax_agree']}/{stats['positions']}")


if __name__ == "__main__":
    main()
