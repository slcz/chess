# chess_ai/train/ — training data + loop

- **`selfplay_data.py`** — SPSHARD1 records to training tensors.
  `PackedRound.add_game` ingests a shard (planes/mask/π slots/outcome,
  fast-cap plies optionally dropped), `finalize()` packs them, and
  `stack_selfplay` builds the batch the trainer consumes.
- **`trainer.py`** — `TrainConfig` (lr, batch, epoch count, loss weights,
  bf16), the AGZ-shape loss (policy CE to π′ + WDL value CE + KataGo
  aux-reply CE), Adam/AdamW, atomic checkpointing, device auto-pick
  (MPS/CUDA/CPU), and `tensors_to_torch`.

The replay buffer lives outside this package now: it is the set of shard
directories on disk, trimmed by the round scripts.
