"""chess_ai.train — training data + loop.

- `selfplay_data` — SPSHARD1 records to training tensors (PackedRound)
- `trainer` — AGZ-shape loss (policy CE to π′ + WDL value CE + KataGo aux
  reply CE), Adam, atomic checkpointing, device auto-pick (MPS/CUDA/CPU).
"""
from chess_ai.train.trainer import (
    TrainConfig,
    Trainer,
    pick_device,
    tensors_to_torch,
)

__all__ = ["TrainConfig", "Trainer", "pick_device", "tensors_to_torch"]
