"""chess_ai/tinygame/net.py — the smoke net + its numpy inference mirror.

TinyNet keeps the CHESS NET'S CONTRACT and nothing else:

    forward(tokens[B,18], mask[B,9], aux_mask[B,9]) ->
        {"policy_logp": [B,9]  masked log-softmax — EXACTLY −inf at illegal
                          slots (the search and the dense-π′ CE both rely on
                          this; trainer.py's NaN note applies verbatim),
         "wdl":         [B,3]  raw logits (loss applies log_softmax),
         "aux_logp":    [B,9]  masked log-softmax over the aux mask — the
                          opponent-reply head, same head set as run 1}

A 2×256 MLP is the whole trunk (the point is the loop, not capacity: the
entire game is 5478 positions). Defaults mirror TrainConfig where they
exist (Adam 3e-4, grad clip 1.0 — set in loop.py).

NumpyEvaluator: the SELF-PLAY inference path. The Phase 3 chess engine
calls the net from C; the analogous seam here is a plain-numpy forward of
the same weights, synced once per training round. Rationale (measured
during 3.5 bring-up): a size-1 torch forward pays ~0.3–0.5 ms of dispatch
overhead per call — at ~150 evals/game × hundreds of games per round that
is minutes of pure overhead on the Mac, while the same MLP in numpy is
tens of µs. The math is identical float32 matmul chains; a test pins
agreement with the torch forward to 1e-5.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_ai.tinygame.game import NN_INPUT_DIM

NN_POLICY = 9
NN_WDL = 3


class TinyNet(nn.Module):
    """18 → 256 → 256, three heads (policy / WDL / aux reply)."""

    def __init__(self, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(NN_INPUT_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, NN_POLICY)
        self.wdl = nn.Linear(hidden, NN_WDL)
        self.aux = nn.Linear(hidden, NN_POLICY)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor,
                aux_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.trunk(tokens)
        return {
            "policy_logp": masked_log_softmax(self.policy(h), mask),
            "wdl": self.wdl(h),
            "aux_logp": masked_log_softmax(self.aux(h), aux_mask),
        }


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """log-softmax with illegal slots at exactly −inf (the chess net's
    policy_logp contract — probability exactly zero, never NaN)."""
    return F.log_softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)


class NumpyEvaluator:
    """Same-weights numpy forward behind gumbel.py's `Evaluator` interface:
    `evaluate(planes, legal_slots, mover) -> (logp dict, wdl triple)`.

    One synchronous size-1 call per position — deliberately the 3.2a
    reference shape (no queues, no batching; PLAN 3.5 validates the learn
    loop, not the batched engine). `sync_from` snapshots the torch weights;
    the per-round call order in loop.py keeps self-play on the CURRENT net.
    """

    def __init__(self, net: TinyNet):
        self.net = net
        self.sync_from(net)

    def sync_from(self, net: TinyNet) -> None:
        sd = net.state_dict()
        self._w = [sd[k].detach().cpu().numpy() for k in (
            "trunk.0.weight", "trunk.0.bias",
            "trunk.2.weight", "trunk.2.bias",
            "policy.weight", "policy.bias",
            "wdl.weight", "wdl.bias",
            "aux.weight", "aux.bias")]

    def _forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(policy_logits, wdl_logits, aux_logits) for one [18] input.

        errstate: Apple Accelerate (this numpy's BLAS) raises spurious
        divide/overflow/invalid FP flags inside its matmul blocking —
        outputs are bit-identical to torch's F.linear (pinned by test;
        measured max diff 0.0) — so the flags are masked here rather
        than spamming every search with warnings."""
        (w0, b0, w1, b1, wp, bp, wv, bv, wa, ba) = self._w
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            h = np.maximum(x @ w0.T + b0, 0.0)
            h = np.maximum(h @ w1.T + b1, 0.0)
            return h @ wp.T + bp, h @ wv.T + bv, h @ wa.T + ba

    def evaluate(self, planes: np.ndarray, legal_slots: Sequence[int],
                 mover: int) -> Tuple[Dict[int, float], Tuple[float, float, float]]:
        pol, wdl, _aux = self._forward(planes.astype(np.float32))
        raws = pol[list(legal_slots)]
        m = raws.max()
        lse = m + math.log(np.exp(raws - m).sum())
        logp = {s: float(r) for s, r in zip(legal_slots, raws - lse)}
        e = np.exp(wdl - wdl.max())
        p = e / e.sum()
        return logp, (float(p[0]), float(p[1]), float(p[2]))
