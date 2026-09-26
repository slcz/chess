"""
chess_ai/supervised — supervised pretraining pipeline (production seed
path since D6, 2026-09-25).

STATUS: this code is mainline — the supervised track validated the
Phase 2.1 architecture on real data and was promoted to the production
initialization path (self-play boots from its strongest checkpoint).
The data half of the old quarantine still binds: datasets, packs, and
checkpoints never enter git (`data/`, `runs/` gitignored; the seed
checkpoint travels via HuggingFace), and human knowledge enters the
production system exactly once — as weight initialization (no human
games in replay buffers, targets, or post-init training data).

What this package provides: train ONLY the WDL value head (value-only
rounds) or both heads jointly (rounds 3/3b) on real human-game outcomes
from Lichess monthly dumps; play via 1-ply value greed or policy
imitation (no search — the search consumer is Phase 3's MCTS).

Modules:
    pgn.py          streaming Lichess-PGN parser + SAN→UCI (own engine,
                    python-chess stays a test-only oracle)
    dataset.py      filtered "games pack" store + on-the-fly position
                    sampler (encode-on-demand, the Phase 3 path)
    play.py         ValueStrategy (1-ply greedy) + PolicyStrategy
                    (imitation), both no-search
    train_value.py  CLI: download / preprocess / train / eval / match
"""
