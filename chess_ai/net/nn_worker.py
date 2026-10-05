"""
nn_worker.py — the Python half of the NN evaluator (engine/src/nn_eval.c).

Owns the model and the device-side tensors. The host staging tensors are
allocated here (pinned where the platform allows) and handed to C by raw
pointer: walkers fill planes/mask directly with nn_encode, and forward(n)
is one H2D + forward + D2H round for the batch's filled prefix — no Python
objects per row, no extra copies. Runs entirely on the C search's IO thread
(the only Python entrant; the waker thread is suspended during rounds).
"""
import torch

from chess_ai.net.model import ChessTransformer
from chess_ai.train.trainer import Trainer

_S: dict = {}
_MODELS: dict = {}          # (ckpt, device) -> loaded model (shared across games)


def setup(ckpt: str, device: str, rows: int) -> dict:
    key = (ckpt, device)
    if key not in _MODELS:
        model = Trainer.load(ckpt).model.to(device).eval()
        _MODELS[key] = model
    model = _MODELS[key]

    pin = device == "cuda"
    try:
        x = torch.zeros(rows, 64, 117, pin_memory=pin)
        m = torch.zeros(rows, 4240, dtype=torch.bool, pin_memory=pin)
        lp = torch.empty(rows, 4240, pin_memory=pin)
        w = torch.empty(rows, 3, pin_memory=pin)
    except (RuntimeError, NotImplementedError):     # no pinned memory here
        x = torch.zeros(rows, 64, 117)
        m = torch.zeros(rows, 4240, dtype=torch.bool)
        lp = torch.empty(rows, 4240)
        w = torch.empty(rows, 3)
    xd, md = x.to(device), m.to(device)
    _S.update(model=model, device=device, x=x, m=m, lp=lp, w=w, xd=xd, md=md)
    return {"planes": x.data_ptr(), "mask": m.data_ptr(),
            "logp": lp.data_ptr(), "wdl": w.data_ptr()}


def forward(n: int) -> int:
    s = _S
    with torch.inference_mode():
        s["xd"][:n].copy_(s["x"][:n])
        s["md"][:n].copy_(s["m"][:n])
        out = s["model"](s["xd"][:n], s["md"][:n])
        s["lp"][:n].copy_(out["policy_logp"].cpu())
        s["w"][:n] = torch.softmax(out["wdl"], dim=-1).cpu()
    return n
