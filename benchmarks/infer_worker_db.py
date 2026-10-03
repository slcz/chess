"""
benchmarks/infer_worker_db.py — double-buffered (ping-pong) variant of
infer_worker: TWO input staging buffers + TWO output staging buffers, the
net always working on B=8192. While the GPU computes slot A, the host
fills slot B's input and drains slot A's output.

Streams/events (the standard producer-consumer pipeline):
  s_copy    — H2D and D2H transfers
  s_compute — the forward passes
  ev_free[slot] — recorded on s_copy after the slot's D2H is enqueued;
        the host waits on it before REUSING the slot (refill input AND
        read output — one event covers both).

Host protocol per batch k (slot = k & 1):
  wait_free(slot)   -> block until the slot's previous D2H completed
  argmax(slot)      -> per-position argmax over the delivered policy
                       (the "search reads its move") — pure CPU work
  [C fills the slot's pinned input via raw pointers]
  enqueue(slot)     -> H2D -> forward -> D2H, all async; returns at once

Nothing per batch blocks on the GPU except wait_free of the slot being
reused — by then the GPU has had a full batch of compute to finish it.
"""
import os

import numpy as np
import torch

from chess_ai.net.model import ChessTransformer

_S: dict = {}
NSLOTS = 2


def setup(batch_size: int, compile: int = 0) -> dict:
    torch.backends.cudnn.benchmark = True
    # one INDEPENDENT compiled instance per slot: cudagraph trees require
    # static input/output addresses, and the ping-pong alternates them —
    # a single shared wrapper would re-capture on every address switch
    models = []
    for slot in range(NSLOTS):
        net = ChessTransformer().to("cuda").eval()
        if compile == 1:
            # dynamic compile has no static-address requirement — share
            # one instance across slots (two would double inductor's
            # memory footprint)
            if models:
                break
            net = torch.compile(net, dynamic=True)
        elif compile == 2:
            net = torch.compile(net, dynamic=False,
                                mode="reduce-overhead")
        models.append(net)
    if compile == 1:
        models = [models[0], models[0]]
    _S["models"] = models
    s_h2d = torch.cuda.Stream()
    s_d2h = torch.cuda.Stream()
    s_compute = torch.cuda.Stream()
    _S.update(s_copy=s_h2d, s_h2d=s_h2d, s_d2h=s_d2h,
              s_compute=s_compute,
              ev_free=[torch.cuda.Event() for _ in range(NSLOTS)],
              B=batch_size, tl=[])
    out = {}
    for slot in range(NSLOTS):
        x = torch.empty(batch_size, 64, 117, dtype=torch.float32,
                        pin_memory=True)
        mask = torch.empty(batch_size, 4240, dtype=torch.bool,
                           pin_memory=True)
        pol = torch.empty(batch_size, 4240, dtype=torch.float32,
                          pin_memory=True)
        wdl = torch.empty(batch_size, 3, dtype=torch.float32,
                          pin_memory=True)
        # preallocated device-side conversion targets: copy_ converts
        # bf16->fp32 IN PLACE into these — no per-batch allocations
        # (fresh 139 MB allocs made the allocator sync against the
        # running forward and dominated enqueue)
        _S[f"polf{slot}"] = torch.empty(batch_size, 4240,
                                        dtype=torch.float32, device="cuda")
        _S[f"wdlf{slot}"] = torch.empty(batch_size, 3,
                                        dtype=torch.float32, device="cuda")
        _S[f"x{slot}"], _S[f"mask{slot}"] = x, mask
        _S[f"pol{slot}"], _S[f"wdl{slot}"] = pol, wdl
        out[f"x{slot}_ptr"] = x.data_ptr()
        out[f"mask{slot}_ptr"] = mask.data_ptr()
    return out


def wait_free(slot: int) -> None:
    _S["ev_free"][slot].synchronize()


def argmax(slot: int) -> int:
    """The deliver-side read the search would do (defeats DCE too)."""
    return int(np.argmax(_S[f"pol{slot}"].numpy(), axis=1).sum())


def push(slot: int) -> None:
    """H2D + forward for the slot — everything the COMPUTE side needs.
    No D2H here: the copy engine's FIFO is single-file on GeForce, so
    the next batch's H2D must enter the queue BEFORE this batch's D2H
    (which is gated on the forward) or it head-of-line blocks for the
    whole forward duration."""
    m = _S
    B = m["B"]
    # device-side buffers are allocated lazily on first use
    if f"xd{slot}" not in m:
        m[f"xd{slot}"] = torch.empty(B, 64, 117, dtype=torch.float32,
                                     device="cuda")
        m[f"md{slot}"] = torch.empty(B, 4240, dtype=torch.bool,
                                     device="cuda")
    ev0 = torch.cuda.Event(enable_timing=True)
    ev0.record(m["s_h2d"])                      # push start
    ev_h2d = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(m["s_h2d"]):
        if not os.environ.get("DB_NOH2D"):
            m[f"xd{slot}"].copy_(m[f"x{slot}"], non_blocking=True)
            m[f"md{slot}"].copy_(m[f"mask{slot}"], non_blocking=True)
        ev_h2d.record(m["s_h2d"])
    m["s_compute"].wait_event(ev_h2d)
    if os.environ.get("DB_NOFWD"):
        m[f"fwd_ev{slot}"] = ev_h2d      # diagnostic: nothing to wait on
        return
    with torch.cuda.stream(m["s_compute"]):
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16):
            out = m["models"][slot](m[f"xd{slot}"], m[f"md{slot}"])
        m[f"polf{slot}"].copy_(out["policy_logp"])          # bf16 -> fp32
        m[f"wdlf{slot}"].copy_(out["wdl"])
        ev_fwd = torch.cuda.Event(enable_timing=True)
        ev_fwd.record(m["s_compute"])
    m[f"fwd_ev{slot}"] = ev_fwd
    if os.environ.get("DB_TIMING"):
        _S["tl"].append((ev0, ev_h2d, ev_fwd))


def pull(slot: int) -> None:
    """D2H for the slot — enqueued AFTER the next push so the copy FIFO
    order is H2D(next) -> D2H(current); gated on the forward's event."""
    m = _S
    m["s_d2h"].wait_event(m[f"fwd_ev{slot}"])
    with torch.cuda.stream(m["s_d2h"]):
        if not os.environ.get("DB_NOD2H"):
            m[f"pol{slot}"].copy_(m[f"polf{slot}"], non_blocking=True)
            m[f"wdl{slot}"].copy_(m[f"wdlf{slot}"], non_blocking=True)
        ev_d2h = torch.cuda.Event(enable_timing=True)
        ev_d2h.record(m["s_d2h"])
        m["ev_free"][slot].record(m["s_d2h"])
    if os.environ.get("DB_TIMING"):
        _S["tl"][-1] = _S["tl"][-1] + (ev_d2h,)


def dump_timeline() -> None:
    """GPU-side truth: per batch, ms since the first push for
    [push_start, H2D_end, fwd_end, D2H_end]. Overlap is visible as the
    fwd window of batch k extending past the h2d window of batch k+1."""
    torch.cuda.synchronize()
    tl = _S["tl"]
    e0 = tl[0][0]
    print("  GPU timeline (ms from first push):")
    print("  batch   push_start  H2D_end   fwd_end   D2H_end")
    for k, evs in enumerate(tl):
        ts = [e0.elapsed_time(e) for e in evs]
        print(f"  {k:5d}  " + "  ".join(f"{t:8.1f}" for t in ts))


def sync() -> None:
    torch.cuda.synchronize()
