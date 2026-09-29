"""
chess_ai/train/selfplay_data.py — self-play records → the training format
(Phase 3.4: "game recorder + replay-buffer writer", the trainer side).

The recorder half of 3.4 already shipped with the parallel engine
(3.3b, PARALLEL_DESIGN §8 deviation 11): finished games are dumped at
collect time to per-worker shard files (engine/include/chess_sp_dump.h),
and `chess_ai.selfplay.spengine.read_records` reads them back as
gumbel.py GameRecords. This module is the other half — those records →
tensors the Phase 2 trainer consumes, plus the rolling replay window.

SAMPLE SCHEMA (extends data.py's Phase 2 schema with the dense target)
=====================================================================
For every ply where a move was PLAYED (the resign ply records a search but
no move — it contributes nothing, mirroring Phase 2's "don't train a
policy where no decision completed"; the value signal of the resignation
is already carried by every earlier ply's z label, which is
resigner-loses):

    tokens    [64, 117]  f32   — C encoder output (replayed via Board)
    mask      [4240]     bool  — legal-move mask
    pol_idx   []         i64   — the PLAYED move's slot (metrics stand-in;
                                 not the loss target for self-play data)
    pi_slots  [K]        i64   — π′ support (the Gumbel candidate set A₀),
                                 zero-padded by REPEATING THE FIRST REAL
                                 slot (padding with slot 0 would gather
                                 −inf·0 = NaN from policy_logp)
    pi_probs  [K]        f32   — π′ aligned with pi_slots, renormalized to
                                 sum exactly 1 (the shard stores f32-rounded
                                 probs; the renorm keeps the CE scale exact),
                                 0 on padding
    wdl       [3]        f32   — game outcome z as W/D/L, MOVER's view
    aux_*                      — opponent-reply target, same convention as
                                 data.py: the reply at ply+1 in OUR frame;
                                 aux_w = 0 when no reply exists (last ply,
                                 or the next ply is the resignation)

K = PI_SUPPORT_MAX = 16 = the C record layout's SP_REC_CANDIDATES bound
(engine/include/chess_search.h; m_cap ≤ 16 by design), so the dense arrays
are fixed-width without inspecting the data.

LOSS side: Trainer.train_step computes the target-weighted CE
−Σ_a π′(a)·log p_θ(a) whenever pi_slots/pi_probs are present
(trainer.py); with a one-hot π′ that is bit-identical to the Phase 2
sparse CE on pol_idx (pinned by test).
"""
from __future__ import annotations

import bisect
import json
import random
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from chess_ai.engine import Board
from chess_ai.net.features import (
    legal_move_indices,
    policy_mask,
    uci_to_index,
    wdl_onehot,
)
from chess_ai.selfplay.gumbel import GameRecord
from chess_ai.selfplay.spengine import read_records

# The dense target's fixed width — the C record layout's π′-support bound
# (SP_REC_CANDIDATES, chess_search.h). The reader asserts n_pi ≤ this on
# every ply, so a C-side raise of the cap fails loudly here, not silently.
PI_SUPPORT_MAX = 16


# ---------------------------------------------------------------------------
# Record → samples (one Board walk per game, same pattern as data.py)
# ---------------------------------------------------------------------------

def playable_plies(record: GameRecord) -> int:
    """Positions this record contributes: plies where a move was played."""
    return sum(1 for m in record.moves if m.played_uci)


def _dense_pi(slots: Sequence[int], probs: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """π′ support → fixed-width (pi_slots, pi_probs), renormalized.

    Padding repeats the first REAL slot with prob 0 — a literal slot-0 pad
    could point at an illegal slot where the net's policy_logp is −inf, and
    (−inf) · 0.0 = NaN would poison the loss."""
    assert 0 < len(slots) <= PI_SUPPORT_MAX, "π′ support outside the record bound"
    p = np.asarray(probs, dtype=np.float64)
    # p >= 0, not p > 0: an extreme-tail candidate's f32 prob may round to
    # exactly 0 in the shard — a legitimate rounded-to-zero store.
    assert (p >= 0).all() and abs(p.sum() - 1.0) < 1e-6, \
        "π′ not a distribution"
    pi_slots = np.full(PI_SUPPORT_MAX, slots[0], dtype=np.int64)
    pi_probs = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
    pi_slots[: len(slots)] = slots
    pi_probs[: len(slots)] = p / p.sum()    # exact renorm over the f32-rounded store
    return pi_slots, pi_probs


def samples_at(record: GameRecord, plies: Iterable[int]) -> List[Dict[str, np.ndarray]]:
    """Build the samples for the given game-ply indices (0-based positions
    in `record.moves`) — ONE board walk, skipping encode/legal for unasked
    plies. Pliers is a MULTISET: sampling with replacement may ask for the
    same ply twice, and each ask is its own row (the sample is built once
    and repeated). The buffer's sampler calls this; selfplay_samples is
    the everything-at-once convenience."""
    counts = Counter(plies)
    board = Board(record.start_fen)
    out: List[Dict[str, np.ndarray]] = []
    for i, mv in enumerate(record.moves):
        if not mv.played_uci:
            # The resign ply (always last — the engine only writes an empty
            # UCI for the resignation that ends the game).
            assert i == len(record.moves) - 1, "empty UCI before the game ended"
            break
        if counts.get(i):
            side = board.side_to_move
            mask = policy_mask(legal_move_indices(board))
            pi_slots, pi_probs = _dense_pi(mv.pi_slots, mv.pi_probs)
            assert mask[pi_slots[: len(mv.pi_slots)]].all(), \
                "π′ support outside the position's legal set"
            assert mask[mv.played_slot], "played move not in its own legal set"
            wdl = wdl_onehot(record.z_white, side)
            tokens = board.encode()
            board.push(mv.played_uci)
            # Aux reply: ply i+1's played move + ply i+1's legal set, both
            # re-indexed into OUR frame (the aux head predicts the enemy's
            # move in our coordinates — same stitch as data.replay_samples).
            nxt = record.moves[i + 1] if i + 1 < len(record.moves) else None
            if nxt is not None and nxt.played_uci:
                aux_idx = uci_to_index(nxt.played_uci, side)
                aux_mask = policy_mask(
                    [uci_to_index(m, side) for m in board.legal_moves])
                assert aux_mask[aux_idx], "reply move not in its own aux mask"
                aux_w = 1.0
            else:                       # last ply, or the reply is a resignation
                aux_idx, aux_mask, aux_w = 0, mask, 0.0
            sample = {
                "tokens": tokens,
                "mask": mask,
                "pol_idx": np.int64(mv.played_slot),
                "pi_slots": pi_slots,
                "pi_probs": pi_probs,
                "wdl": wdl,
                "aux_idx": np.int64(aux_idx),
                "aux_mask": aux_mask,
                "aux_w": np.float32(aux_w),
            }
            out.extend([sample] * counts[i])   # one build, one row per ask
        else:
            board.push(mv.played_uci)
    assert len(out) == sum(counts.values()), \
        "asked for a ply the record does not play"
    return out


def selfplay_samples(record: GameRecord) -> List[Dict[str, np.ndarray]]:
    """Every sample of one record (all played plies)."""
    return samples_at(record, range(playable_plies(record)))


def build_selfplay_tensors(records: Sequence[GameRecord]) -> Dict[str, np.ndarray]:
    """Stack every sample of every record into training arrays (the dense-
    π′ counterpart of data.build_tensors; Trainer accepts either)."""
    samples = [s for r in records for s in selfplay_samples(r)]
    if not samples:
        raise ValueError("no positions in records")
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


# ---------------------------------------------------------------------------
# The replay window (self-play's ReplayBuffer)
# ---------------------------------------------------------------------------

class SelfPlayBuffer:
    """Rolling window of recent self-play games (PLAN §3: 40–50k games at
    full scale; the Mac's 16 GB argues smaller locally — it's a knob).

    Stores gumbel.py GameRecords only (~9 KB/game at ~80 plies: moves carry
    the π′ supports), exactly like Phase 2's ReplayBuffer stores compact
    records — encoded tensors are rebuilt on demand by replaying the record
    through the C encoder (~µs/position, trivial next to the forward +
    backward). Positions are sampled uniformly across the window (same
    convention as Phase 2's buffer); each pick replays its game from the
    start to the picked ply. If profiling ever disagrees, a per-record
    sample cache in front of this is the same-interface upgrade."""

    def __init__(self, max_games: int):
        self._records: deque = deque(maxlen=max_games)
        self._snap: List[GameRecord] = []      # snapshot + prefix sums,
        self._cum: List[int] = [0]             # rebuilt lazily on add
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
        """Playable positions in the window (resign plies excluded)."""
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
        """n uniformly-sampled positions, stacked into a dense-π′ batch."""
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


# ---------------------------------------------------------------------------
# Unit ingestion (the cloud-rules side: small idempotent work units)
# ---------------------------------------------------------------------------

def read_unit(unit_dir: str | Path) -> List[GameRecord]:
    """One work unit's records, via the DONE.json manifest the
    chess_selfplay binary writes on completion (n_workers + unit_seed are
    the reader's two unknowns; games_done cross-checks the shard count)."""
    unit_dir = Path(unit_dir)
    manifest = json.loads((unit_dir / "DONE.json").read_text())
    records = read_records(unit_dir, manifest["n_workers"],
                           manifest["unit_seed"])
    assert len(records) == manifest["games_done"], \
        f"{unit_dir}: shards hold {len(records)} games, " \
        f"manifest says {manifest['games_done']}"
    return records


def ingest_units(unit_dirs: Sequence[str | Path]) -> List[GameRecord]:
    """Union of many units' records, deduped by (seed, game_index) — later
    dirs win (a re-run unit supersedes its own partial earlier copy) — and
    sorted. This is the trainer's whole view of a scattered shard tree."""
    by_key: Dict[Tuple[int, int], GameRecord] = {}
    for d in unit_dirs:
        for r in read_unit(d):
            by_key[(r.seed, r.game_index)] = r
    return [by_key[k] for k in sorted(by_key)]
