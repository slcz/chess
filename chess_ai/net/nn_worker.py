"""
nn_worker.py — the Python half of the NN evaluator (engine/src/nn_eval.c).

Owns the model and the device-side tensors. The host staging tensors are
allocated here (pinned where the platform allows) and handed to C by raw
pointer: walkers fill planes/mask directly with nn_encode, the io thread
drives the three-phase protocol, and delivery reads logp/wdl straight out
of the host buffers. One tensor set per landing-zone bank.

Three-phase protocol (the machinery proven in benchmarks/infer_worker_db.py):
  push(bank, n)       H2D + launch compute — ASYNC on CUDA (three streams:
                      s_h2d / s_compute / s_d2h, the forward gated on the
                      H2D event); returns at once so the io thread can push
                      the other bank and keep the forwards back to back.
  pull(bank, n)       enqueue D2H for the results, gated on the forward's
                      event; async.
  wait_ready(bank)    block until the bank's D2H has completed — the only
                      GPU-blocking call; the caller may read logp/wdl after.

On backends without streams (MPS) push runs the whole H2D + forward + D2H
synchronously and pull/wait_ready are no-ops. Runs entirely on the landing
zone's single io thread (the only Python entrant).
"""
import torch

from chess_ai.net.model import ChessTransformer
from chess_ai.train.trainer import Trainer

_S: dict = {}
_MODELS: dict = {}          # (ckpt, device) -> loaded model (shared across games)


def setup(ckpt: str, device: str, rows: int, banks: int = 1) -> dict:
    key = (ckpt, device)
    if key not in _MODELS:
        model = Trainer.load(ckpt).model.to(device).eval()
        _MODELS[key] = model
    model = _MODELS[key]

    cuda = device == "cuda"
    if cuda:
        _S.update(s_h2d=torch.cuda.Stream(), s_compute=torch.cuda.Stream(),
                  s_d2h=torch.cuda.Stream(),
                  ev_fwd=[torch.cuda.Event() for _ in range(banks)],
                  ev_free=[torch.cuda.Event() for _ in range(banks)])

    sets = []
    out = {"planes": [], "mask": [], "logp": [], "wdl": []}
    for _ in range(banks):
        try:
            pin = cuda
            x = torch.zeros(rows, 64, 117, pin_memory=pin)
        except (RuntimeError, NotImplementedError):     # no pinned memory
            pin = False
            x = torch.zeros(rows, 64, 117)
        s = {"x": x,
             "m": torch.zeros(rows, 4240, dtype=torch.bool, pin_memory=pin),
             "lp": torch.empty(rows, 4240, pin_memory=pin),
             "w": torch.empty(rows, 3, pin_memory=pin)}
        if cuda:
            # device-side buffers per bank (static addresses; also what
            # torch.compile would need) and fp32 conversion targets so
            # pull's copy_ does no per-batch allocation
            s["xd"] = torch.empty(rows, 64, 117, device=device)
            s["md"] = torch.empty(rows, 4240, dtype=torch.bool, device=device)
            s["polf"] = torch.empty(rows, 4240, device=device)
            s["wdlf"] = torch.empty(rows, 3, device=device)
        else:
            s["xd"] = s["x"].to(device)
            s["md"] = s["m"].to(device)
        sets.append(s)
        out["planes"].append(s["x"].data_ptr())
        out["mask"].append(s["m"].data_ptr())
        out["logp"].append(s["lp"].data_ptr())
        out["wdl"].append(s["w"].data_ptr())
    _S.update(model=model, device=device, cuda=cuda, sets=sets)
    return out


def push(bank: int, n: int) -> int:
    s = _S["sets"][bank]
    if not _S["cuda"]:
        # synchronous backend: the whole forward at push
        with torch.inference_mode():
            s["xd"][:n].copy_(s["x"][:n])
            s["md"][:n].copy_(s["m"][:n])
            out = _S["model"](s["xd"][:n], s["md"][:n])
            s["lp"][:n].copy_(out["policy_logp"].cpu())
            s["w"][:n] = torch.softmax(out["wdl"], dim=-1).cpu()
        return n
    # CUDA: H2D on the copy stream, forward on the compute stream, gated on
    # the H2D event — everything async, returns before the GPU runs it
    ev_h2d = torch.cuda.Event()
    with torch.cuda.stream(_S["s_h2d"]):
        s["xd"][:n].copy_(s["x"][:n], non_blocking=True)
        s["md"][:n].copy_(s["m"][:n], non_blocking=True)
        ev_h2d.record(_S["s_h2d"])
    _S["s_compute"].wait_event(ev_h2d)
    with torch.cuda.stream(_S["s_compute"]):
        with torch.inference_mode():
            out = _S["model"](s["xd"][:n], s["md"][:n])
        s["polf"][:n].copy_(out["policy_logp"])
        s["wdlf"][:n].copy_(out["wdl"])
        _S["ev_fwd"][bank].record(_S["s_compute"])
    return n


def pull(bank: int, n: int) -> int:
    if not _S["cuda"]:
        return n                      # synchronous backend: already on host
    s = _S["sets"][bank]
    _S["s_d2h"].wait_event(_S["ev_fwd"][bank])
    with torch.cuda.stream(_S["s_d2h"]):
        s["lp"][:n].copy_(s["polf"][:n], non_blocking=True)
        s["w"][:n].copy_(s["wdlf"][:n], non_blocking=True)
        _S["ev_free"][bank].record(_S["s_d2h"])
    return n


def wait_ready(bank: int, n: int = 0) -> int:
    if _S["cuda"]:
        _S["ev_free"][bank].synchronize()
    return bank
