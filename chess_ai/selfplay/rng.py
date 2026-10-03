"""
chess_ai/selfplay/rng.py — the RNG spec for search (Phase 3.2a).

Why this module exists (PARALLEL_DESIGN.md R6): the C parallel engine and the
Python reference must consume *identical random streams* from identical seeds
for the exact-diff verification gates (L2: C-sequential vs Python must play
bit-identical games). That is only possible if the generator, the seeding, the
uniform mapping, and the *order of consumption* are all pinned. This module is
that spec — engine/src (3.3b) mirrors it line for line:

  - splitmix64 for seed derivation (Vigna's reference constants),
  - xoshiro256++ as the per-game generator,
  - uniform() maps to (0, 1) exclusive at 53-bit precision via a midpoint
    offset — chosen so -log(-log(u)) (the Gumbel draw) is always finite and
    so an IEEE-double C implementation reproduces it bit-exactly,
  - gumbel() = inverse-CDF Gumbel(0, 1).

Consumption order (enforced in gumbel.py, mirrored in C):
  1. per move, at root-eval time: one gumbel() draw per legal action, in FFI
     movegen enumeration order (a forced position draws exactly one — the
     general path has no k==1 special case);
  2. per sampled move: one uniform() draw, resolved by inverse CDF over the
     candidate set in movegen order (training samples at every ply —
     sample_plies=None; eval seed phases sample a fixed number of plies).
A deterministic (non-sampled) ply consumes nothing beyond its Gumbels.
Nothing else may draw from the stream.
"""

from __future__ import annotations

import math

MASK64 = (1 << 64) - 1

_GOLDEN = 0x9E3779B97F4A7C15          # splitmix64 additive constant
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


def splitmix64_next(state: int) -> tuple[int, int]:
    """One splitmix64 step: (new_state, output). Pure, branch-free — the
    exact reference arithmetic, so C mirrors it trivially."""
    state = (state + _GOLDEN) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * _M1) & MASK64
    z = ((z ^ (z >> 27)) * _M2) & MASK64
    return state, z ^ (z >> 31)


class SplitMix64:
    """Stateful wrapper for seed derivation."""

    def __init__(self, seed: int):
        self._state = seed & MASK64

    def next(self) -> int:
        self._state, out = splitmix64_next(self._state)
        return out


def game_seed(unit_seed: int, game_index: int) -> int:
    """Per-game seed from the work-unit seed and the game's index — the
    PARALLEL_DESIGN R6 derivation (part of the C-mirror contract)."""
    a = SplitMix64(unit_seed).next()
    return SplitMix64(a ^ (game_index & MASK64)).next()


def _rotl(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK64


class Xoshiro256PP:
    """xoshiro256++ — the per-game generator (PARALLEL_DESIGN R6).

    Seeded via four splitmix64 draws from `seed`. The all-zero state is the
    generator's single fixed point; splitmix64 seeding cannot produce it, and
    we guard anyway to keep the guarantee total.
    """

    def __init__(self, seed: int):
        sm = SplitMix64(seed)
        s = [sm.next() for _ in range(4)]
        if not any(s):
            s[0] = 1
        self._s = s

    def next_u64(self) -> int:
        s = self._s
        result = (_rotl((s[0] + s[3]) & MASK64, 23) + s[0]) & MASK64
        t = (s[1] << 17) & MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl(s[3], 45)
        return result

    def uniform(self) -> float:
        """Uniform in (0, 1), 53-bit precision, endpoints excluded.

        ((x >> 11) + 0.5) * 2^-53 maps the 53 high bits to the *midpoints* of
        2^53 equal bins — never 0.0, never 1.0, exactly representable, and
        reproducible by any IEEE-754 double implementation (the C mirror does
        the same arithmetic).
        """
        return ((self.next_u64() >> 11) + 0.5) * (2.0 ** -53)

    def gumbel(self) -> float:
        """Standard Gumbel(0, 1) via inverse CDF: -log(-log(u))."""
        u = self.uniform()
        return -math.log(-math.log(u))
