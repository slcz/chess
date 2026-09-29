"""chess_ai.tinygame — Phase 3.5, the tiny-game smoke test (PLAN §4).

Tic-tac-toe re-instantiates the ENTIRE Phase 3 learn loop — Gumbel search
→ dense-π′/WDL records → replay window → target-weighted-CE training →
net feeds the next self-play round — at a scale where "perfect play" is
exactly checkable against an exhaustive minimax oracle, in minutes, on
the Mac, before any cloud spend (PLAN §1 budget principle).

This package is deliberately free of the C engine (the loop it validates
is the LEARN loop, not the play path; PLAN 3.5) and of `chess_ai.net` /
`chess_ai.train` (their shapes are the chess move space and trunk). What
it shares with the chess pipeline is the ALGORITHM, ported line for line
with the pinned items kept identical — see each module's header for the
mapping to its chess counterpart.
"""
