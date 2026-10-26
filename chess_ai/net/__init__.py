"""chess_ai.net — feature encoding + the network (Phase 2).

- `features` — the 4240-slot policy move space (decision D1: 64×64 +
  promotions) and a pure-Python reference encoder (test oracle for the C
  encoder in engine/src/nn_encode.c).
- `model` — the CF-6M-style encoder-only transformer (PLAN.md Phase 2,
  "Agreed net design"): axial 2D RoPE position scheme, config-swappable
  policy head (RopePolicyHead default, pooled legacy), WDL value head,
  aux opponent-reply head.
"""
from chess_ai.net.features import (
    NN_TOKEN_DIM,
    NN_TOKENS,
    POLICY_SIZE,
    legal_move_indices,
    policy_mask,
    uci_to_index,
    wdl_onehot,
)
from chess_ai.net.model import (
    ChessTransformer,
    NetConfig,
    PooledPolicyHead,
    count_parameters,
)

__all__ = [
    "NN_TOKEN_DIM",
    "NN_TOKENS",
    "POLICY_SIZE",
    "legal_move_indices",
    "policy_mask",
    "uci_to_index",
    "wdl_onehot",
    "ChessTransformer",
    "NetConfig",
    "PooledPolicyHead",
    "count_parameters",
]
