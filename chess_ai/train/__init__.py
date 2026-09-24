"""chess_ai.train — data pipeline + training loop (Phase 2).

- `data` — GameRecord (compact self-play unit), ReplayBuffer (rolling
  window of recent games), sample building (encode on demand via the C
  engine), and seeded random-game fixtures for gate tests.
- `trainer` — AGZ-shape loss (policy CE to π′ + WDL value CE + KataGo aux
  reply CE), Adam, atomic checkpointing, device auto-pick (MPS/CUDA/CPU).
"""
from chess_ai.train.data import (
    GameRecord,
    ReplayBuffer,
    build_tensors,
    generate_random_games,
    generate_strategy_games,
    material_sign_wdl,
)
from chess_ai.train.trainer import TrainConfig, Trainer, pick_device

__all__ = [
    "GameRecord",
    "ReplayBuffer",
    "build_tensors",
    "generate_random_games",
    "generate_strategy_games",
    "material_sign_wdl",
    "TrainConfig",
    "Trainer",
    "pick_device",
]
