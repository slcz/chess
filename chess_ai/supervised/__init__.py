"""
chess_ai/supervised — the value-net validation side track (branch-only).

ZERO-KNOWLEDGE QUARANTINE (binding, agreed 2026-09-24):
    Everything in this package exists ONLY on the `supervised_learning`
    branch. The dataset, preprocessed packs, trained checkpoints, and any
    results never merge to main and never seed the production net — only
    bug fixes found here get distilled over. Main-branch development keeps
    the pure zero-knowledge self-play path.

What this side track does (user decision 2026-09-24, narrowed the same
day): train ONLY the WDL value head of the Phase 2.1 transformer on real
human-game outcomes (Lichess monthly dumps). No policy supervision. At
play time we evaluate every legal move with the value net (the position
after the move, sign-flipped back to our perspective) and play the argmax
— a 1-ply greedy value player. If that bot plays sensible chess, the
Phase 2.1 trunk + value architecture is sound before any self-play cloud
spend (mirrors how arXiv 2409.12272 itself validated, supervised).

Why outcome labels and not engine evals: our encoder needs each
position's 8-ply history and repetition flags, so the dataset must be
full move sequences, not a flat FEN+score dump. Human games give us
sequences + outcomes for free; eval dumps would also be engine-labeled,
which this track deliberately avoids.

Modules:
    pgn.py          streaming Lichess-PGN parser + SAN→UCI (own engine,
                    python-chess stays a test-only oracle)
    dataset.py      filtered "games pack" store + on-the-fly position
                    sampler (encode-on-demand, the Phase 3 path)
    play.py         ValueStrategy — the 1-ply greedy value player
    train_value.py  CLI: preprocess / train / eval / match
"""
