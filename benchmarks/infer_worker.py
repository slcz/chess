"""
benchmarks/infer_worker.py — the Python half of the C-side inference bench
(user's speed-of-light-MCTS experiment, 2026-10-03): owns the net and the
pinned staging tensors, exposes their HOST pointers to C, and runs one
forward per C call.

The C program (c_infer_bench.c) fills the pinned CPU buffers directly via
data_ptr() (memcpy — no Python objects per row), then calls step():

    setup(B, device, compile) -> {"x_ptr", "mask_ptr", "x_bytes", "mask_bytes"}
    step(n, consume)          -> one (H2D copy + forward [+ D2H policy])

`consume` simulates the deliver half of the search boundary: the policy
(B x 4240) and WDL come back to the CPU as numpy (implicit device sync) —
what the C search would read per request. Without consume, the loop is
pure produce+forward and only the final sync() bounds the wall clock, so
H2D/compute pipeline across batches (the production evaluator's regime).
"""
import torch

from chess_ai.net.model import ChessTransformer, NetConfig

_S: dict = {}


def setup(batch_size: int, device: str = "cuda", compile: int = 0) -> dict:
    torch.backends.cudnn.benchmark = True
    model = ChessTransformer()          # fresh init: timing is shape-bound
    model = model.to(device).eval()
    if compile:
        model = torch.compile(model, dynamic=True)
    # pinned host staging: C memcpys into these; H2D copies from them
    x = torch.empty(batch_size, 64, 117, dtype=torch.float32,
                    pin_memory=True)
    mask = torch.empty(batch_size, 4240, dtype=torch.bool,
                       pin_memory=True)
    xd = x.to(device)
    md = mask.to(device)
    _S.update(model=model, x=x, mask=mask, xd=xd, md=md, device=device)
    return {"x_ptr": x.data_ptr(), "mask_ptr": mask.data_ptr(),
            "x_bytes": x.numel() * x.element_size(),
            "mask_bytes": mask.numel() * mask.element_size()}


def step(n: int, consume: int) -> int:
    m = _S
    m["xd"].copy_(m["x"][:n], non_blocking=True)
    m["md"].copy_(m["mask"][:n], non_blocking=True)
    with torch.inference_mode(), torch.autocast(
            m["device"], dtype=torch.bfloat16):
        out = m["model"](m["xd"], m["md"])
    if consume:
        # deliver half: policy + WDL back to CPU (syncs the stream), then
        # the per-position read the search would do (argmax over the full
        # move space) — representative use, defeats dead-code elimination
        pol = out["policy_logp"].float().cpu().numpy()
        w = torch.softmax(out["wdl"].float(), dim=-1).cpu().numpy()
        import numpy as np
        _S["argmax_sum"] = float(np.argmax(pol, axis=1).sum())
        _S["pol"], _S["w"] = pol, w        # keep alive for C to read
        return n
    return 0


def sync() -> None:
    torch.cuda.synchronize()
