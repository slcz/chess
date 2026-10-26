"""
tests/test_buffer_trim.py — the replay buffer's game-level trim must be
deterministic (same manifest seed => identical kept set on any box) and
statistically honest: keep**rounds keep fraction (default keep = 2/3).
"""
from __future__ import annotations

from chess_ai.train.selfplay_data import buffer_keep, fnv1a64


def test_fnv_stable() -> None:
    # reference value so a silent hash change (which would re-trim every
    # historical pool differently) trips a test
    assert fnv1a64("loop1:i123-456") == 0xB74C5FE94474849B


def test_keep_is_deterministic() -> None:
    for rounds in (0, 1, 3):
        a = [buffer_keep(u, i, rounds, seed=42) for u in range(50)
             for i in range(20)]
        b = [buffer_keep(u, i, rounds, seed=42) for u in range(50)
             for i in range(20)]
        assert a == b


def test_rounds_zero_keeps_everything() -> None:
    assert all(buffer_keep(u, i, 0, seed=1) for u in range(30)
               for i in range(30))


def test_keep_fraction_matches_parameter() -> None:
    keys = [(u, i) for u in range(40) for i in range(600)]   # 24k draws
    for keep in (2 / 3, 3 / 4, 1 / 2):
        for rounds, want in ((1, keep), (2, keep * keep)):
            kept = sum(buffer_keep(u, i, rounds, seed=7, keep=keep)
                       for u, i in keys)
            assert abs(kept / len(keys) - want) < 0.02, (keep, rounds, kept)
    # keep >= 1 disables the trim entirely
    assert all(buffer_keep(u, i, 3, seed=7, keep=1.0) for u, i in keys[:500])
