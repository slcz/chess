"""
chess_ai/train/trainer.py — the training loop (Phase 2.2).

LOSS (the AGZ shape, per the agreed Gumbel design in PLAN.md Phase 3):

    total = policy_loss + value_weight · value_loss + aux_reply_weight · aux_loss

    policy_loss = −Σ_a π′(a) · log p_θ(a)        (target-weighted CE; with
                                                  π′ one-hot this is plain CE
                                                  on the played move. The
                                                  target-entropy term of the
                                                  KL is constant in θ — dropped)
    value_loss  = WDL cross-entropy vs the game outcome z (mover's view)
    aux_loss    = same CE form against the opponent's actual reply,
                  in our frame (KataGo's aux policy target; knob default on)

Adam, batch 512–1024 at scale (256 in tests), gradient clipping on by
default — the memorization gate trains at high LR where clipping keeps
early spikes from blowing up the run.

CHECKPOINTING follows the salad-reliability rules (PLAN §5): atomic
(tmp file + rename) so a kill can never leave a torn checkpoint, and
everything needed to resume in one file (model, optimizer, step count,
config). At cloud scale the periodic push off-instance (4.3) wraps
Trainer.save(); locally this is just a file.

DEVICE: "auto" prefers MPS (the Mac) → CUDA (cloud) → CPU, so the same
code runs everywhere Phase 2 needs it to.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from chess_ai.net.model import ChessTransformer, NetConfig


@dataclass
class TrainConfig:
    lr: float = 3e-4
    batch_size: int = 256
    value_weight: float = 1.0
    aux_reply_weight: float = 0.3     # KataGo aux head knob; 0 disables
    grad_clip: Optional[float] = 1.0
    device: str = "auto"
    seed: int = 0


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# numpy dtypes → torch, per key of the data.build_tensors() schema.
_TORCH_DTYPES = {
    "tokens": torch.float32,
    "mask": torch.bool,
    "pol_idx": torch.int64,
    "wdl": torch.float32,
    "aux_idx": torch.int64,
    "aux_mask": torch.bool,
    "aux_w": torch.float32,
}


def tensors_to_torch(batch: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: torch.from_numpy(np.ascontiguousarray(v)).to(device=device, dtype=_TORCH_DTYPES[k])
        for k, v in batch.items()
    }


class Trainer:
    """Owns model + optimizer + step count; one `train_step` per batch."""

    def __init__(self, model: ChessTransformer, cfg: TrainConfig):
        self.cfg = cfg
        self.device = pick_device(cfg.device)
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.steps = 0
        torch.manual_seed(cfg.seed)

    # --- one optimization step ------------------------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        out = self.model(batch["tokens"], batch["mask"], batch["aux_mask"])

        nll = -out["policy_logp"].gather(1, batch["pol_idx"][:, None]).squeeze(1)
        policy_loss = nll.mean()

        wdl_logp = F.log_softmax(out["wdl"], dim=-1)
        value_loss = -(wdl_logp * batch["wdl"]).sum(-1).mean()

        aux_nll = -out["aux_logp"].gather(1, batch["aux_idx"][:, None]).squeeze(1)
        w = batch["aux_w"]
        denom = w.sum().clamp(min=1.0)
        aux_loss = (aux_nll * w).sum() / denom

        loss = (policy_loss
                + self.cfg.value_weight * value_loss
                + self.cfg.aux_reply_weight * aux_loss)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.steps += 1

        with torch.no_grad():
            policy_acc = (out["policy_logp"].argmax(-1) == batch["pol_idx"]).float().mean()
            value_acc = (out["wdl"].argmax(-1) == batch["wdl"].argmax(-1)).float().mean()
        return {
            "loss": loss.item(),
            "policy": policy_loss.item(),
            "value": value_loss.item(),
            "aux": aux_loss.item(),
            "policy_acc": policy_acc.item(),
            "value_acc": value_acc.item(),
        }

    # --- a whole run over a fixed tensor set ----------------------------

    def fit(
        self,
        tensors: Dict[str, np.ndarray],
        steps: int,
        log_every: int = 0,
        rng: Optional[np.random.Generator] = None,
    ) -> List[Dict[str, float]]:
        """Random-with-replacement minibatches from one fixed tensor set —
        the exact regime the Phase 2 gate needs (memorize a fixed dataset).
        Phase 3's production loop replaces `fit` with streaming batches
        from the replay buffer; train_step is the shared core."""
        n = tensors["tokens"].shape[0]
        rng = rng or np.random.default_rng(self.cfg.seed)
        history: List[Dict[str, float]] = []
        for step in range(steps):
            idx = rng.integers(0, n, size=self.cfg.batch_size)
            batch = tensors_to_torch({k: v[idx] for k, v in tensors.items()}, self.device)
            metrics = self.train_step(batch)
            history.append(metrics)
            if log_every and (step + 1) % log_every == 0:
                m = metrics
                print(f"step {self.steps:6d}  loss {m['loss']:.4f}  "
                      f"policy {m['policy']:.4f} (acc {m['policy_acc']:.3f})  "
                      f"value {m['value']:.4f} (acc {m['value_acc']:.3f})  "
                      f"aux {m['aux']:.4f}")
        return history

    # --- eval without training ------------------------------------------

    @torch.no_grad()
    def evaluate(self, tensors: Dict[str, np.ndarray], batch_size: int = 512) -> Dict[str, float]:
        """Accuracy over a whole tensor set (no grad, eval-safe)."""
        self.model.eval()
        sums = {"policy_acc": 0.0, "value_acc": 0.0, "policy_nll": 0.0}
        n = tensors["tokens"].shape[0]
        for start in range(0, n, batch_size):
            sl = slice(start, min(start + batch_size, n))
            batch = tensors_to_torch({k: v[sl] for k, v in tensors.items()}, self.device)
            out = self.model(batch["tokens"], batch["mask"], batch["aux_mask"])
            sums["policy_acc"] += (out["policy_logp"].argmax(-1) == batch["pol_idx"]).float().sum().item()
            sums["value_acc"] += (out["wdl"].argmax(-1) == batch["wdl"].argmax(-1)).float().sum().item()
            sums["policy_nll"] += (-out["policy_logp"].gather(1, batch["pol_idx"][:, None])).sum().item()
        self.model.train()
        return {k: v / n for k, v in sums.items()}

    # --- checkpointing (atomic; PLAN §5) ----------------------------------

    def save(self, path: os.PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "steps": self.steps,
            "config": asdict(self.cfg),
            "net_config": asdict(self.model.cfg),   # architecture travels with
        }                                            # the weights (load needs it)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)             # atomic on the same filesystem
        return path

    @classmethod
    def load(cls, path: os.PathLike, model: Optional[ChessTransformer] = None) -> "Trainer":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = TrainConfig(**payload["config"])
        if model is None:
            net_cfg = NetConfig(**payload["net_config"])
            model = ChessTransformer(net_cfg)
        model.load_state_dict(payload["model"])
        trainer = cls(model, cfg)
        trainer.opt.load_state_dict(payload["optimizer"])
        trainer.steps = payload["steps"]
        return trainer
