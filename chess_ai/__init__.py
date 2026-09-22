"""chess_ai — the Python side of the project.

Modules:
    engine      cffi binding over engine/build/libchess (the C rules engine)
    strategies  dummy players (random, material-greedy) + the Strategy protocol
    uci         minimal UCI client for playing against external engines
    match       game/match runner between two strategies

The C engine is the source of truth for rules; Python orchestrates.
"""
