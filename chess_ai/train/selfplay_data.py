"""
chess_ai/train/selfplay_data.py — self-play shard records → training
samples (the trainer side of the selfplay driver).

Sample schema: data.py's Phase 2 schema (tokens/mask/pol_idx/wdl/aux_*)
plus the dense search-improved policy target the driver records:

    pi_slots  [K] i64  — the root moves that earned visits, mover's frame
                          (chess_policy_slot), zero-padded by REPEATING THE
                          FIRST REAL slot — padding with slot 0 would gather
                          −inf·0 = NaN out of policy_logp
    pi_probs  [K] f32  — π ∝ visits renormalized to sum exactly 1 (the shard
                          stores f32-rounded probs; the renorm keeps the CE
                          scale exact), 0 on padding
    sample_w  []  f32  — per-sample weight: KataGo App. D-style downweight
                          of hopeless positions via the recorded mover-view
                          root value (outlook < −0.9 → 0.1..1), else 1

K = PI_SUPPORT_MAX = 256 > the legal-move bound (218), so the driver's
full visit distribution always fits — no truncation, no tail mass lost.

The trainer consumes this via Trainer.train_step's dense-π branch
(target-weighted CE); with a one-hot π that path is equivalent to the
sparse pol_idx CE.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from chess_ai.engine import Board
from chess_ai.net.features import (
    legal_move_indices,
    policy_mask,
    uci_to_index,
    wdl_onehot,
)
from chess_ai.selfplay.shard import SpGame

PI_SUPPORT_MAX = 256

# KataGo's App. D λ scheme adapted (see module docstring): downweight
# positions whose mover is hopelessly lost — outlook < −0.9 maps linearly
# to a 0.1..1 weight, everything above stays 1.
OUTLOOK_WEIGHT_LO = -0.9


def outlook_weight(outlook: float) -> float:
    if outlook >= OUTLOOK_WEIGHT_LO:
        return 1.0
    lam = max(0.0, min(1.0, (outlook + 1.0) / 0.1))
    return 0.1 + 0.9 * lam


def selfplay_samples(game: SpGame) -> List[Dict[str, np.ndarray]]:
    """Replay one shard game and build every sample. One Board walk, the
    same shape as data.py replay_samples: each ply's tensors/π are read at
    that ply, the aux target is the next ply's move in OUR frame."""
    board = Board(game.start_fen)
    per_ply = []
    for ply in game.plies:
        side = board.side_to_move
        legal = board.legal_moves
        mask = policy_mask(legal_move_indices(board))
        pol_idx = uci_to_index(ply.played_uci, side)
        assert mask[pol_idx], f"played {ply.played_uci} not in its legal set"
        assert ply.mover == side, "record mover disagrees with replay"

        slots = [s for s, _ in ply.pi]
        probs = np.array([p for _, p in ply.pi], dtype=np.float64)
        assert len(slots) <= PI_SUPPORT_MAX
        assert (probs >= 0).all() and probs.sum() > 0
        probs /= probs.sum()                    # exact renorm (f32 rounding)

        wdl = wdl_onehot(2.0 * game.z_white - 1.0, side)   # z ∈ {1,.5,0} → {+1,0,−1}
        per_ply.append({
            "tokens": board.encode(),
            "mask": mask,
            "legal": legal,
            "pol_idx": np.int64(pol_idx),
            "wdl": wdl,
            "side": side,
            "pi_slots": slots,
            "pi_probs": probs,
            "sample_w": np.float32(outlook_weight(ply.outlook)),
        })
        board.push(ply.played_uci)

    samples = []
    for i, s in enumerate(per_ply):
        if i + 1 < len(per_ply):
            aux_idx = uci_to_index(game.plies[i + 1].played_uci, s["side"])
            aux_mask = policy_mask(
                [uci_to_index(m, s["side"]) for m in per_ply[i + 1]["legal"]]
            )
            aux_w = 1.0
        else:
            aux_idx, aux_mask, aux_w = 0, s["mask"], 0.0
        pi_slots = np.full(PI_SUPPORT_MAX, s["pi_slots"][0], dtype=np.int64)
        pi_slots[: len(s["pi_slots"])] = s["pi_slots"]
        pi_probs = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
        pi_probs[: len(s["pi_probs"])] = s["pi_probs"]
        samples.append({
            "tokens": s["tokens"],
            "mask": s["mask"],
            "pol_idx": s["pol_idx"],
            "wdl": s["wdl"],
            "aux_idx": np.int64(aux_idx),
            "aux_mask": aux_mask,
            "aux_w": np.float32(aux_w),
            "pi_slots": pi_slots,
            "pi_probs": pi_probs,
            "sample_w": s["sample_w"],
        })
    return samples


def stack_selfplay(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}
