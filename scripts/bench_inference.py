"""
scripts/bench_inference.py — raw NN inference speed for the production net
(user branch puct_mcts_inference_speed, 2026-10-03): forward-pass latency
and throughput vs batch size, no game/search machinery — just the
transformer trunk + heads, the thing every MCTS evaluator parks requests
against.

What it reports per (device, precision, batch size):
  - ms/forward (mean over timed iters, after warmup)
  - forwards/s and tokens/s (B × 64 square-tokens per forward)
  - µs/token — the linear-regime number to compare with the salad 3090
    sweep (~230 µs/token fixed; ~13 ms fixed overhead, 2026-10-01)

Inputs are realistic shapes, not toy: [B, 64, 117] planes and a [B, 4240]
legal-mask with ~35 legal slots per row (typical midgame mobility) — the
masked log-softmax cost depends on mask density.

    .venv/bin/python scripts/bench_inference.py \
        --device mps --ckpt runs/policy_value_net_rope_25k.pt --bf16

bf16 autocast is attempted when --bf16 is passed and skipped with a note
if the torch build doesn't support it on the device (MPS bf16 needs a
recent torch; the 3090 runs bf16 in production).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def load_model(ckpt: str | None, device: str):
    from chess_ai.net.model import ChessTransformer, NetConfig

    if not ckpt:
        model = ChessTransformer()
        name = "default-init"
    else:
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        raw = dict(state.get("net_config") or {})
        if raw:
            raw.setdefault("pos_scheme", "rope2d")
            model = ChessTransformer(NetConfig(**raw))
        else:
            model = ChessTransformer()
        model.load_state_dict(state.get("model", state))
        name = Path(ckpt).stem
    return model.to(device).eval(), name


def make_inputs(B: int, device: str, legal_mean: int = 35):
    x = torch.randn(B, 64, 117, device=device)
    mask = torch.zeros(B, 4240, dtype=torch.bool, device=device)
    # ~35 legal slots per row: slots are ordered from/to within 64 squares,
    # so a midgame position's legal moves cluster in the low slot range —
    # spread the sampled slots across 0..2800 to be representative.
    g = torch.Generator().manual_seed(0)
    idx = torch.randint(0, 2800, (B, legal_mean), generator=g).to(device)
    mask.scatter_(1, idx, True)
    aux = mask.clone()
    return x, mask, aux


def bench_one(model, device, B: int, iters: int, warmup: int,
              bf16: bool, compile: bool) -> dict:
    x, mask, aux = make_inputs(B, device)
    fwd = (lambda: model(x, mask, aux))
    if compile:
        model_c = torch.compile(model, dynamic=True)
        fwd = (lambda: model_c(x, mask, aux))
    autocast = (lambda: torch.autocast(device, dtype=torch.bfloat16)) \
        if bf16 else torch.inference_mode
    # warmup (compile + MPS kernel caches live here)
    with torch.inference_mode():
        for _ in range(warmup):
            with autocast():
                out = fwd()
        if device == "mps":
            torch.mps.synchronize()
    # timed iters
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            with autocast():
                out = fwd()
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return {
        "batch": B,
        "ms_per_forward": round(dt * 1e3, 3),
        "forwards_per_s": round(1.0 / dt, 1),
        "tokens_per_s": round(B * 64 / dt, 0),
        "us_per_token": round(dt / (B * 64) * 1e6, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        prog="python scripts/bench_inference.py",
        description="Raw forward-pass speed of the production net.")
    p.add_argument("--ckpt", default=None,
                   help="checkpoint to load (default: freshly-initialized "
                        "default net — timing is shape-bound, weights "
                        "barely matter)")
    p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    p.add_argument("--batches", default="1,8,32,128,512,2048",
                   help="comma-separated batch sizes")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--bf16", action="store_true",
                   help="also try bf16 autocast (skipped if unsupported)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(dynamic=True) the model")
    p.add_argument("--jsonl", default=None,
                   help="append result rows here")
    a = p.parse_args()

    model, name = load_model(a.ckpt, a.device)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"net: {name}  ({n_params / 1e6:.2f}M params, device {a.device}"
          f"{', compiled' if a.compile else ''})")
    print(f"{'B':>6} {'ms/fwd':>9} {'fwd/s':>9} {'tokens/s':>12} "
          f"{'us/token':>9}  precision")

    rows = []
    for B in [int(b) for b in a.batches.split(",")]:
        row = bench_one(model, a.device, B, a.iters, a.warmup,
                        bf16=False, compile=a.compile)
        row["precision"] = "fp32"
        rows.append(row)
        print(f"{B:>6} {row['ms_per_forward']:>9.2f} "
              f"{row['forwards_per_s']:>9.1f} {row['tokens_per_s']:>12.0f} "
              f"{row['us_per_token']:>9.2f}  fp32")
        if a.bf16:
            try:
                row16 = bench_one(model, a.device, B, a.iters, a.warmup,
                                  bf16=True, compile=a.compile)
            except (RuntimeError, TypeError) as exc:
                print(f"{'':>6} bf16 autocast unsupported: {exc}")
                continue
            row16["precision"] = "bf16"
            rows.append(row16)
            print(f"{B:>6} {row16['ms_per_forward']:>9.2f} "
                  f"{row16['forwards_per_s']:>9.1f} "
                  f"{row16['tokens_per_s']:>12.0f} "
                  f"{row16['us_per_token']:>9.2f}  bf16")

    if a.jsonl:
        out = Path(a.jsonl)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            for r in rows:
                f.write(json.dumps({"net": name, "device": a.device,
                                    "compiled": a.compile, **r}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
