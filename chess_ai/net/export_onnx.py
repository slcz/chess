"""
chess_ai/net/export_onnx.py — export a checkpoint to ONNX + parity-gate it.

Why ONNX: the serverless web GUI (wasm-serverless branch) runs the trained
net in the browser via onnxruntime-web (WebGPU EP, WASM fallback). Torch
checkpoints can't travel there; ONNX is what ORT-web consumes. This is
offline tooling: export once per generation, gate it, publish the .onnx
next to the page's assets.

The exported graph mirrors the C eval seam's contract exactly
(chess_ai/net/nn_worker.py is the native twin):

    inputs : tokens [B, 64, 117] float32
             mask   [B, 4240]    bool  (True = legal)
    outputs: policy_logp [B, 4240]  illegal slots exp→ EXACTLY 0 (the
                                   search's hard contract, model.py §mask)
             wdl        [B, 3]     softmax PROBABILITIES W/D/L, mover's
                                   view — the search backs up w[0] − w[2]

The aux opponent-reply and soft-policy heads are training-time only; they
are dropped from the graph (smaller file, no aux_mask input, and the GUI
never consumes them).

Parity gate: identical inputs through torch and onnxruntime-CPU on REAL
positions — boards encoded by the C encoder itself (Board.encode) and
masks built with uci_to_index over legal moves, so the gate exercises the
true input distribution instead of synthetic noise. Pass criteria:

    max |Δ policy_logp| over legal slots  ≤ 2e-4
    max |Δ wdl prob|                      ≤ 1e-4
    exp(policy_logp) at illegal slots      == 0.0 exactly
    policy argmax agreement                reported (not asserted)

fp32 torch vs fp32 ORT-CPU differences are op-scheduling noise; 2e-4 is
well above observed drift and far below anything PUCT can feel (visit
probabilities shifting in the 4th decimal).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from chess_ai.net.features import POLICY_SIZE, uci_to_index

# Tolerances of the gate (see module docstring for rationale).
TOL_LOGP = 2e-4
TOL_WDL = 1e-4


class InferenceWrapper(nn.Module):
    """Trimmed forward for export: the two heads the search consumes, in
    the seam's semantics. Applying the wdl softmax HERE means the wasm
    glue feeds ORT output straight to C unchanged — same as nn_worker's
    synchronous path."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        x = self.model.trunk(tokens)
        logp = self.model._masked_logp(self.model.policy(x), mask)
        wdl = torch.softmax(self.model.value(x), dim=-1)
        return logp, wdl


def export_onnx(model: nn.Module, out_path: str | Path, opset: int = 17) -> Path:
    """Trace the trimmed wrapper and save `out_path`. Batch axis dynamic —
    the search batches parked eval rows, so B is anything from 1 up."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    tokens = torch.zeros(8, 64, model.cfg.token_dim)
    mask = torch.zeros(8, POLICY_SIZE, dtype=torch.bool)
    mask[:, :64] = True                       # any non-empty mask traces fine
    with torch.inference_mode():
        torch.onnx.export(
            InferenceWrapper(model), (tokens, mask), str(out_path),
            opset_version=opset, dynamo=False,
            input_names=["tokens", "mask"],
            output_names=["policy_logp", "wdl"],
            dynamic_axes={name: {0: "batch"} for name in
                          ("tokens", "mask", "policy_logp", "wdl")},
        )
    return out_path


def random_positions(n: int, seed: int = 0, max_plies: int = 120):
    """n random legal positions as (tokens [n,64,117], mask [n,4240]) numpy
    arrays: random playouts through the FFI Board, encoded by the C encoder
    (Board.encode). Terminal positions are skipped — the engine never
    evaluates those, so the gate shouldn't either."""
    from chess_ai.engine import Board, Termination

    rng = random.Random(seed)
    tokens, masks = [], []
    while len(tokens) < n:
        b = Board()
        for _ in range(rng.randrange(max_plies)):
            if b.termination != Termination.PLAYING:
                break
            b.push(rng.choice(b.legal_moves))
        if b.termination != Termination.PLAYING:
            continue
        m = np.zeros(POLICY_SIZE, dtype=bool)
        side = b.side_to_move
        for u in b.legal_moves:
            m[uci_to_index(u, side)] = True
        tokens.append(b.encode())
        masks.append(m)
    return np.stack(tokens), np.stack(masks)


def parity_gate(model: nn.Module, onnx_path: str | Path,
                n_positions: int = 64, seed: int = 0) -> dict:
    """Torch vs onnxruntime on real positions; asserts the tolerances and
    the exact-zero illegal contract, returns the numbers it checked."""
    import onnxruntime as ort

    tokens, mask = random_positions(n_positions, seed)
    sess = ort.InferenceSession(str(onnx_path),
                                providers=["CPUExecutionProvider"])
    o_logp, o_wdl = sess.run(None, {"tokens": tokens, "mask": mask})

    with torch.inference_mode():
        t_logp, t_wdl = InferenceWrapper(model)(
            torch.from_numpy(tokens), torch.from_numpy(mask))
    t_logp, t_wdl = t_logp.numpy(), t_wdl.numpy()

    d_logp = np.abs(t_logp - o_logp)
    d_wdl = float(np.abs(t_wdl - o_wdl).max())
    with np.errstate(under="ignore"):         # exp(-3.4e38) underflows to 0
        illegal_prob = np.exp(o_logp[~mask])
    assert d_logp[mask].max() <= TOL_LOGP, \
        f"policy_logp drift {d_logp[mask].max():.2e} > {TOL_LOGP}"
    assert d_wdl <= TOL_WDL, f"wdl drift {d_wdl:.2e} > {TOL_WDL}"
    assert illegal_prob.max(initial=0.0) == 0.0, \
        "illegal slot has nonzero probability in the ONNX graph"
    return {
        "positions": n_positions,
        "max_d_logp_legal": float(d_logp[mask].max()),
        "max_d_wdl": d_wdl,
        "argmax_agree": int((t_logp.argmax(-1) == o_logp.argmax(-1)).sum()),
    }
