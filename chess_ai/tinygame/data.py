"""chess_ai/tinygame/data.py — self-play records → training samples + the
replay window (the tiny-game counterpart of train/selfplay_data.py).

Same SAMPLE SCHEMA, same semantics, 9-slot move space:

    tokens    [18]  f32   — board.encode(), mover-relative
    mask      [9]   bool  — legal-move mask
    pol_idx   []    i64   — played cell (metrics stand-in, as in chess)
    pi_slots  [K]   i64   — π′ support (candidate set A₀), padded by
                            REPEATING THE FIRST REAL slot (the NaN trap:
                            a literal 0-pad could point at an illegal slot
                            where policy_logp is −inf and −inf·0 = NaN —
                            identical to selfplay_data._dense_pi)
    pi_probs  [K]   f32   — π′, renormalized to sum exactly 1, 0 on padding
    wdl       [3]   f32   — outcome z, MOVER's view (wdl_onehot convention)
    aux_*                 — opponent-reply target: the reply at ply i+1 in
                            OUR frame (ttt's frame never rotates, so the
                            aux slot is the reply cell itself); aux_w = 0
                            when no reply exists (last ply, or the reply
                            is the resignation)

The resign ply contributes NO sample (no decision completed) — same rule
as chess 3.4. `_dense_pi` and `wdl_onehot` are DUPLICATED from the chess
side (they are 12 lines each and importing train/selfplay_data would
drag the C-engine-backed chess modules into this package's import graph
— 3.5 must run wherever Python+numpy+torch run, PLAN 3.5).

The buffer keeps SelfPlayBuffer's exact semantics: rolling window of
records, positions sampled uniformly WITH replacement (the ply list is a
multiset — one sample build per ply, one row per ask).
"""
from __future__ import annotations

import bisect
import random
from collections import Counter, deque
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from chess_ai.tinygame.game import TicTacToe
from chess_ai.tinygame.search import GameRecord

# Fixed width of the dense π′ arrays — gumbel candidate sets are bounded
# by m_cap (16) in both games; ttt's real bound is ≤ 9.
PI_SUPPORT_MAX = 16


def _dense_pi(slots: Sequence[int],
              probs: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """(pi_slots, pi_probs) fixed-width; identical logic to
    selfplay_data._dense_pi (see module docstring for why it's copied)."""
    assert 0 < len(slots) <= PI_SUPPORT_MAX
    p = np.asarray(probs, dtype=np.float64)
    assert (p >= 0).all() and abs(p.sum() - 1.0) < 1e-6, "π′ not a distribution"
    pi_slots = np.full(PI_SUPPORT_MAX, slots[0], dtype=np.int64)
    pi_probs = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
    pi_slots[: len(slots)] = slots
    pi_probs[: len(slots)] = p / p.sum()
    return pi_slots, pi_probs


def _wdl_onehot(outcome_first_pov: float, side: int) -> np.ndarray:
    """z as W/D/L one-hot from `side`'s view — features.wdl_onehot, copied
    (same import-graph rationale). outcome_first_pov: +1 X win / −1 O / 0."""
    z = outcome_first_pov if side == 0 else -outcome_first_pov
    v = np.zeros(3, dtype=np.float32)
    v[0 if z > 0 else (1 if z == 0 else 2)] = 1.0
    return v


def _mask(cells: Tuple[int, ...]) -> np.ndarray:
    m = np.zeros(9, dtype=bool)
    for s in TicTacToe(cells).legal_moves:
        m[s] = True
    return m


def playable_plies(record: GameRecord) -> int:
    return sum(1 for m in record.moves if m.played_cell >= 0)


def samples_at(record: GameRecord, plies: Iterable[int]) -> List[Dict[str, np.ndarray]]:
    """Samples for the given 0-based positions in record.moves — ONE board
    walk; `plies` is a MULTISET (with-replacement sampling: each ask its
    own row). Mirrors selfplay_data.samples_at."""
    counts = Counter(plies)
    board = TicTacToe(record.start_cells)
    out: List[Dict[str, np.ndarray]] = []
    for i, mv in enumerate(record.moves):
        if mv.played_cell < 0:                 # the resign ply (always last)
            assert i == len(record.moves) - 1, "empty move before game end"
            break
        if counts.get(i):
            side = board.side_to_move
            mask = _mask(board.cells)
            pi_slots, pi_probs = _dense_pi(mv.pi_slots, mv.pi_probs)
            assert mask[pi_slots[: len(mv.pi_slots)]].all()
            assert mask[mv.played_cell]
            wdl = _wdl_onehot(record.z_white, side)
            tokens = board.encode()
            board.push(mv.played_cell)
            # Aux reply: ply i+1's played cell, target mask = the legal set
            # AFTER our move (the reply position's empties) — our frame.
            nxt = record.moves[i + 1] if i + 1 < len(record.moves) else None
            if nxt is not None and nxt.played_cell >= 0:
                aux_idx = nxt.played_cell
                aux_mask = _mask(board.cells)
                assert aux_mask[aux_idx]
                aux_w = 1.0
            else:                               # last ply / reply resigned
                aux_idx, aux_mask, aux_w = 0, mask, 0.0
            sample = {
                "tokens": tokens,
                "mask": mask,
                "pol_idx": np.int64(mv.played_cell),
                "pi_slots": pi_slots,
                "pi_probs": pi_probs,
                "wdl": wdl,
                "aux_idx": np.int64(aux_idx),
                "aux_mask": aux_mask,
                "aux_w": np.float32(aux_w),
            }
            out.extend([sample] * counts[i])
        else:
            board.push(mv.played_cell)
    assert len(out) == sum(counts.values()), "asked for an unplayed ply"
    return out


def selfplay_samples(record: GameRecord) -> List[Dict[str, np.ndarray]]:
    return samples_at(record, range(playable_plies(record)))


class TinyReplayBuffer:
    """Rolling window + uniform-over-positions with-replacement sampling —
    SelfPlayBuffer's semantics at ttt scale (~9 samples/game)."""

    def __init__(self, max_games: int):
        self._records: deque = deque(maxlen=max_games)
        self._snap: List[GameRecord] = []
        self._cum: List[int] = [0]
        self._dirty = False

    def add(self, record: GameRecord) -> None:
        self._records.append(record)
        self._dirty = True

    def add_all(self, records: Iterable[GameRecord]) -> None:
        for r in records:
            self.add(r)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def total_positions(self) -> int:
        self._rebuild()
        return self._cum[-1]

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        self._snap = list(self._records)
        cum = [0]
        for r in self._snap:
            cum.append(cum[-1] + playable_plies(r))
        self._cum = cum
        self._dirty = False

    def sample_tensors(self, rng: random.Random, n: int) -> Dict[str, np.ndarray]:
        """n uniformly-sampled positions stacked into one dense-π′ batch."""
        self._rebuild()
        total = self._cum[-1]
        if total == 0:
            raise RuntimeError("sampled from an empty buffer")
        picks: Dict[int, List[int]] = {}
        for _ in range(n):
            p = rng.randrange(total)
            i = bisect.bisect_right(self._cum, p) - 1
            picks.setdefault(i, []).append(p - self._cum[i])
        samples = [s for i in sorted(picks)
                   for s in samples_at(self._snap[i], picks[i])]
        return {k: np.stack([s[k] for s in samples]) for k in samples[0]}
