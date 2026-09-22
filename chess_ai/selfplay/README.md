# chess_ai/selfplay/ — batched self-play

The throughput core (Phase 3): a Python outer loop over N games in lockstep — C advances each game until it needs an NN evaluation and returns packed input planes for pending leaves; Python runs one batched forward pass per round; priors + values return as arrays. Also home to the game recorder, the replay-buffer writer, and the checkpoint-ladder match runner (net@A vs. net@B).
