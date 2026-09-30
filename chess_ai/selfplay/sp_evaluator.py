"""
chess_ai/selfplay/sp_evaluator.py — the Python side of the embedded-CPython
batch evaluator (PARALLEL_DESIGN F8, R7).

The C engine's evaluator thread (engine/src/sp_evaluator.c) imports this
module once per run, calls make_evaluator(spec), then calls the returned
object's `eval_batch` once per epoch — the GIL is held only around that
call; everything else in the engine is pure C. This is the exact seam the
design's §4.7 evaluator pseudo-code describes: unpack → stack → cast →
net() → sparse-extract legal logp + WDL probs.

Contract (all arguments are memoryviews over the engine's staging buffers —
zero-copy; multi-byte data is little-endian):

    eval_batch(planes, slots, offsets, mover, out_logp, out_wdl)

    planes    u8  [B, 64, 117]  quantized feature tensors (SpReq in
                chess_search.h): dims 0–115 are exactly 0/1; dim 116 is the
                RAW halfmove clock — un-quantize with clock/100.0, which
                reproduces nn_encode's f32 output bit-for-bit.
    slots     u16 [total_legal]  legal policy slots, movegen ENUMERATION
                order, concatenated across the batch (the order the net's
                policy_logp is gathered at, and the order sp_deliver_node
                expects the logp back in).
    offsets   i32 [B + 1]  cumulative legal counts (request i's slots are
                slots[offsets[i]:offsets[i+1]]).
    mover     i32 [B]  side to move at each parked position.
    out_logp  f32 [total_legal]  WRITTEN: the net's policy_logp at the
                legal slots, aligned with `slots`.
    out_wdl   f32 [B, 3]  WRITTEN: softmax(wdl logits) = (pW, pD, pL) from
                the parked position's mover's view.

Specs: "fixture" (uniform priors, uniform WDL), "fixture:white_wins" (the
mover-always-winning fixture, for resignation tests), "torch" (random-init
ChessTransformer), "torch:ckpt=<path>" (a trained checkpoint) with an
optional ",device=<dev>" suffix — "torch:ckpt=net.pt,device=mps" — default
cpu. A checkpoint's saved net_config drives the architecture (with
Trainer.load's pre-flip "shaw" pin), so a checkpoint can never load into
the wrong net shape.
"""
from __future__ import annotations

import math

import numpy as np

NN_TOTAL_DIMS = 64 * 117
POLICY_SIZE = 4240
_HALFMOVE_DIM = 116
_EVAL_CHUNK = 4096   # requests per forward; a 16-worker × 2048-buffer epoch
                     # can carry 32k requests — far too much for one device
                     # tensor (B×64×117 f32 = 9.8 GB). The chunk loop keeps
                     # the GPU side ~1.2 GB and overlaps prep with compute.


class FixtureBatchEvaluator:
    """Batch-invariant TableEvaluator twin: uniform priors (logp =
    −log k per legal slot) and a WDL from `white_wins` (False: ⅓⅓⅓;
    True: the mover always wins — black's outlook is −1, so it resigns).
    Values are cast through float32 exactly like the net's tensors, which
    keeps the parallel engine bit-comparable with a sequential driver fed
    by the same fixture."""

    def __init__(self, white_wins: bool = False):
        self.white_wins = white_wins

    def eval_batch(self, planes, slots, offsets, mover, out_logp, out_wdl):
        offsets = np.frombuffer(offsets, dtype="<i4")
        mover = np.frombuffer(mover, dtype="<i4")
        logp_out = np.frombuffer(out_logp, dtype="<f4")
        wdl_out = np.frombuffer(out_wdl, dtype="<f4")
        for i in range(len(offsets) - 1):
            a, b = int(offsets[i]), int(offsets[i + 1])
            k = b - a
            logp_out[a:b] = np.float32(-math.log(k))
            if self.white_wins and int(mover[i]) == 0:
                wdl_out[3 * i:3 * i + 3] = np.float32([1.0, 0.0, 0.0])
            elif self.white_wins:
                wdl_out[3 * i:3 * i + 3] = np.float32([0.0, 0.0, 1.0])
            else:
                wdl_out[3 * i:3 * i + 3] = np.float32([1 / 3, 1 / 3, 1 / 3])


class TorchBatchEvaluator:
    """ChessTransformer behind the batch contract (R7): the net gets the
    un-quantized f32 planes + the legal mask (scattered from `slots`), and
    the outputs are the masked policy_logp gathered at the legal slots
    (enumeration order) plus the WDL softmax. torch.set_num_threads(1) per
    F8 — the evaluator thread must not fight the MCTS workers for cores."""

    def __init__(self, ckpt: str | None = None, device: str = "cpu"):
        import torch

        from chess_ai.net.model import ChessTransformer, NetConfig

        self._torch = torch
        torch.set_num_threads(1)
        if ckpt:
            state = torch.load(ckpt, map_location="cpu")
            # The architecture comes from the checkpoint itself (Trainer.save
            # payloads carry net_config); the setdefault is the same
            # pre-flip legacy pin as Trainer.load — checkpoints saved before
            # pos_scheme existed are Shaw. A bare state_dict ({"model": …}
            # only) keeps the current default config.
            net_cfg = (state.get("net_config")
                       if isinstance(state, dict) else None)
            if isinstance(net_cfg, dict) and net_cfg:
                raw = dict(net_cfg)
                raw.setdefault("pos_scheme", "shaw")
                self.model = ChessTransformer(NetConfig(**raw))
            else:
                self.model = ChessTransformer()
            self.model.load_state_dict(state.get("model", state))
        else:
            self.model = ChessTransformer()
        self.model = self.model.to(device=device, dtype=torch.float32).eval()
        self._device = next(self.model.parameters()).device

    def eval_batch(self, planes, slots, offsets, mover, out_logp, out_wdl):
        torch = self._torch
        offsets = np.frombuffer(offsets, dtype="<i4")
        b = len(offsets) - 1
        slots_np = np.frombuffer(slots, dtype="<u2")
        planes_np = np.frombuffer(planes, dtype=np.uint8)
        if not planes_np.flags.writeable:
            planes_np = planes_np.copy()   # staging is a PyBUF_READ view
        planes_t = torch.from_numpy(planes_np)       # CPU u8, chunked later
        rows_np = np.repeat(np.arange(b, dtype=np.int64), np.diff(offsets))
        cols_np = slots_np.astype(np.int64)

        logp_out = np.frombuffer(out_logp, dtype="<f4")
        wdl_out = np.frombuffer(out_wdl, dtype="<f4")

        # Un-quantize + mask build run ON THE DEVICE, one chunk at a time.
        # The arithmetic is value-identical to the numpy path (u8→f32 is
        # exact; f32 /100.0 is correctly rounded on both backends).
        for a in range(0, b, _EVAL_CHUNK):
            c = min(_EVAL_CHUNK, b - a)
            x = planes_t[a * NN_TOTAL_DIMS:(a + c) * NN_TOTAL_DIMS] \
                .to(self._device).reshape(c, 64, 117).to(torch.float32)
            x[:, :, _HALFMOVE_DIM] /= 100.0
            r = torch.from_numpy(
                rows_np[offsets[a]:offsets[a + c]] - a).to(self._device)
            k = torch.from_numpy(
                cols_np[offsets[a]:offsets[a + c]]).to(self._device)
            mask = torch.zeros(c, POLICY_SIZE, dtype=torch.bool,
                               device=self._device)
            mask[r, k] = True
            with torch.no_grad():
                out = self.model(x, mask)
            logp_out[offsets[a]:offsets[a + c]] = \
                out["policy_logp"][r, k].cpu().numpy()
            w = torch.softmax(out["wdl"], dim=-1).cpu().numpy()
            wdl_out[3 * a:3 * (a + c)] = w.reshape(-1)


def _parse_kv(rest: str) -> dict[str, str]:
    """'ckpt=net.pt,device=mps' → {'ckpt': 'net.pt', 'device': 'mps'}.
    (A path containing a comma is not expressible — document, don't parse.)"""
    out: dict[str, str] = {}
    for part in rest.split(","):
        if not part:
            continue
        k, _, v = part.partition("=")
        out[k.strip()] = v
    return out


def make_evaluator(spec: str):
    """Factory called once per engine run by the C evaluator (F8)."""
    if spec == "fixture":
        return FixtureBatchEvaluator(white_wins=False)
    if spec == "fixture:white_wins":
        return FixtureBatchEvaluator(white_wins=True)
    if spec == "torch":
        return TorchBatchEvaluator()
    if spec.startswith("torch:"):
        kv = _parse_kv(spec[len("torch:"):])
        device = kv.pop("device", "cpu")
        ckpt = kv.pop("ckpt", None)
        if kv:
            raise ValueError(f"unknown torch evaluator keys: {sorted(kv)}")
        return TorchBatchEvaluator(ckpt=ckpt, device=device)
    raise ValueError(f"unknown evaluator spec: {spec!r}")
