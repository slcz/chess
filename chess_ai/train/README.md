# chess_ai/train/ — training loop

Replay buffer (recent 40–50k games), policy target = MCTS visit distribution, value target = game outcome, Adam at batch 512–1024, and frequent checkpointing that pushes off-instance immediately (instances are cattle). Built in Phase 2.
