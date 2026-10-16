"""
chess_ai/train/trainer.py — the training loop (Phase 2.2).

LOSS (the AGZ shape, per the agreed Gumbel design in PLAN.md Phase 3):

    total = policy_weight · policy_loss + value_weight · value_loss
            + aux_reply_weight · aux_loss

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
    policy_weight: float = 1.0    # 0 = value-only run (supervised side track:
                                   # trunk+value head learn, policy head rides
                                   # along untrained — its acc is the control)
    aux_reply_weight: float = 0.3     # KataGo aux head knob (on by default)
    soft_policy_weight: float = 1.0   # KataGo soft-policy-target weight; the
                                      # soft head is OFF unless a net sets
                                      # NetConfig.soft_policy, so this only
                                      # bites when a run enables it
    grad_clip: Optional[float] = 1.0
    bf16: bool = True              # CUDA autocast (no-op off CUDA)
    compile: bool = True           # torch.compile (CUDA only)
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


def resolve_device(name: str) -> str:
    """A checkpoint's saved device is a TRAINING-time fact, not a load-time
    requirement: resuming a cloud ("cuda") run on the Mac must not demand
    CUDA. If the saved device isn't available here, fall back to auto."""
    if name == "cuda" and not torch.cuda.is_available():
        return "auto"
    if name == "mps" and not torch.backends.mps.is_available():
        return "auto"
    return name


# numpy dtypes → torch, per key of the data.build_tensors() schema
# (pi_*/sample_w: the self-play dense target, selfplay_data.py).
_TORCH_DTYPES = {
    "tokens": torch.float32,
    "mask": torch.bool,
    "pol_idx": torch.int64,
    "wdl": torch.float32,
    "aux_idx": torch.int64,
    "aux_mask": torch.bool,
    "aux_w": torch.float32,
    "pi_slots": torch.int64,
    "pi_probs": torch.float32,
    "pi_probs_soft": torch.float32,
    "sample_w": torch.float32,
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
        self._raw_model = model          # unwrapped: state_dict stays clean
        self._maybe_compile()
        try:                             # fused Adam: one update kernel for
            self.opt = torch.optim.Adam(   # all ~100 tensors (CUDA only)
                model.parameters(), lr=cfg.lr, fused=self.device.type == "cuda")
        except TypeError:                # older torch without `fused`
            self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        self.steps = 0
        torch.manual_seed(cfg.seed)

    def _maybe_compile(self) -> None:
        """(Re)apply the compile wrapper per cfg. Called from __init__ AND
        after a resume swaps cfg — the 2c run silently trained UNcompiled
        because load built the Trainer under the OLD cfg and the swap
        never re-wrapped."""
        if getattr(self, "cfg").compile and self.device.type == "cuda":
            self.model = torch.compile(self._raw_model, mode="reduce-overhead")
        else:
            self.model = self._raw_model

    # --- one optimization step ------------------------------------------

    def _autocast(self):
        """bf16 autocast context (CUDA only; identity elsewhere so the
        same TrainConfig runs on the Mac's MPS/CPU untouched)."""
        if self.cfg.bf16 and self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        import contextlib
        return contextlib.nullcontext()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        with self._autocast():
            out = self.model(batch["tokens"], batch["mask"], batch["aux_mask"])

            if "pi_slots" in batch:
                # dense search-improved target (self-play data): per-sample
                # −Σ_a π(a)·logp_θ(a), optionally weighted. π slots are legal
                # moves so the gathered logp is finite; padding repeats the
                # first slot at prob 0.
                slot_logp = out["policy_logp"].gather(1, batch["pi_slots"])
                nll = -(batch["pi_probs"] * slot_logp).sum(-1)
                w = batch.get("sample_w")
                policy_loss = ((nll * w).sum() / w.sum().clamp(min=1e-8)
                               if w is not None else nll.mean())
                policy_acc = ((slot_logp.argmax(-1)
                               == batch["pi_probs"].argmax(-1)).float().mean())
            else:
                nll = -out["policy_logp"].gather(
                    1, batch["pol_idx"][:, None]).squeeze(1)
                policy_loss = nll.mean()
                policy_acc = (out["policy_logp"].argmax(-1)
                              == batch["pol_idx"]).float().mean()

            wdl_logp = F.log_softmax(out["wdl"], dim=-1)
            value_loss = -(wdl_logp * batch["wdl"]).sum(-1).mean()

            if self.model.aux is not None:        # net built without the
                aux_nll = -out["aux_logp"].gather(    # aux head has nothing to
                    1, batch["aux_idx"][:, None]).squeeze(1)   # score (side-track
                w = batch["aux_w"]                    # small nets; aux weight
                denom = w.sum().clamp(min=1.0)        # must be 0 there anyway)
                aux_loss = (aux_nll * w).sum() / denom
            else:
                aux_loss = torch.zeros((), device=self.device)

            if self.model.soft is not None:
                soft_logp = out["soft_logp"].gather(1, batch["pi_slots"])
                soft_nll = -(batch["pi_probs_soft"] * soft_logp).sum(-1)
                w = batch.get("sample_w")
                soft_loss = ((soft_nll * w).sum() / w.sum().clamp(min=1e-8)
                             if w is not None else soft_nll.mean())
            else:
                soft_loss = torch.zeros((), device=self.device)

            loss = (self.cfg.policy_weight * policy_loss
                    + self.cfg.value_weight * value_loss
                    + self.cfg.aux_reply_weight * aux_loss
                    + self.cfg.soft_policy_weight * soft_loss)

        self.opt.zero_grad(set_to_none=True)      # backward OUTSIDE autocast
        loss.backward()
        if self.cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()
        self.steps += 1

        # One host sync for ALL metrics (.item() per scalar would be 8
        # GPU->CPU round-trips per step — visible once steps get fast).
        with torch.no_grad():
            scalars = {
                "loss": loss.detach(),
                "policy": policy_loss.detach(),
                "value": value_loss.detach(),
                "aux": aux_loss.detach(),
                "soft": soft_loss.detach(),
                "policy_acc": policy_acc.detach(),
                "value_acc": (out["wdl"].argmax(-1)
                              == batch["wdl"].argmax(-1)).float().mean(),
            }
            values = torch.stack(list(scalars.values())).tolist()
        return dict(zip(scalars.keys(), values))

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
            # _raw_model: torch.compile's wrapper would prefix state_dict
            # keys with "_orig_mod." and poison cross-version loads.
            "model": self._raw_model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "steps": self.steps,
            "config": asdict(self.cfg),
            "net_config": asdict(self._raw_model.cfg),   # architecture travels with
        }                                                  # the weights (load needs it)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)             # atomic on the same filesystem
        return path

    @classmethod
    def load(cls, path: os.PathLike, model: Optional[ChessTransformer] = None,
             device: Optional[str] = None) -> "Trainer":
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = TrainConfig(**payload["config"])
        if device is not None:                    # caller's --device beats the
            cfg.device = device                   # checkpoint's saved device
        cfg.device = resolve_device(cfg.device)   # cloud ckpt on a Mac → auto
        cfg.compile = False    # training-time fact (Triton needs a build
                               # toolchain inference must not require — the
                               # 30.5k ckpt eval crash, 2026-09-25)
        if model is None:
            # A checkpoint saved BEFORE a knob existed must rebuild with
            # the value that was default AT SAVE TIME, not today's: the
            # pos_scheme flip (shaw→rope2d, 2026-09-27) would otherwise
            # turn every legacy Shaw checkpoint into a rope2d net and its
            # state_dict (rel_k/rel_v vs rope.cos/sin) would not match.
            # The setdefault below pins the pre-round-4 era; future knob
            # flips need the same treatment for their own legacy window.
            # rope_dirs/rope_lambda are such a window: checkpoints saved
            # before 2026-10-15 carry 2 banks on the 16·2^i ladder, and the
            # cos/sin buffer SHAPE depends on it — today's 4-bank default
            # would fail to load them outright.
            raw_cfg = dict(payload["net_config"])
            raw_cfg.setdefault("pos_scheme", "shaw")
            raw_cfg.setdefault("rope_dirs", 2)
            raw_cfg.setdefault("rope_lambda", 16.0)
            net_cfg = NetConfig(**raw_cfg)
            model = ChessTransformer(net_cfg)
        model.load_state_dict(payload["model"])
        trainer = cls(model, cfg)
        trainer.opt.load_state_dict(payload["optimizer"])
        trainer.steps = payload["steps"]
        return trainer
