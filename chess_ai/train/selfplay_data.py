"""
chess_ai/train/selfplay_data.py — self-play shard records → training
samples (the trainer side of the selfplay driver).

Sample schema: data.py's Phase 2 schema (tokens/mask/pol_idx/wdl/aux_*)
plus the dense search-improved policy target the driver records:

    pi_slots  [K] i64  — the ply's legal moves in engine movegen order,
                          mover's frame (uci_to_index of board.legal_moves:
                          the v3 shard stores only probs, no slot ids),
                          zero-padded by REPEATING THE FIRST REAL slot —
                          padding with slot 0 would gather
                          −inf·0 = NaN out of policy_logp
    pi_probs  [K] f32  — π (pi-bar) renormalized to sum exactly 1 (the shard
                          stores f16-rounded probs; the renorm keeps the CE
                          scale exact), 0 on padding
    sample_w  []  f32  — per-sample weight: KataGo App. D-style downweight
                          of hopeless positions via the recorded mover-view
                          root value (outlook < −0.9 → 0.1..1), else 1

K = PI_SUPPORT_MAX = 256 > the legal-move bound (218), so the driver's
full visit distribution always fits — no truncation, no tail mass lost.

The trainer consumes this via Trainer.train_step's dense-π branch
(target-weighted CE); with a one-hot π that path is equivalent to the
sparse pol_idx CE.

Playout cap randomization (KataGo, Wu 2020 §4): the driver plays ~75% of
moves at a cheap fast cap and the rest at the slow cap (game.n_sims); only
slow-cap plies carry a training-grade π, so they are the only samples.
Fast plies are still replayed through (no per-ply FENs are stored) and
still serve as the aux-reply target of the ply before them.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from chess_ai.engine import Board
from chess_ai.net.features import (
    legal_move_indices,
    policy_mask,
    uci_to_index,
    wdl_onehot,
)
from chess_ai.selfplay.shard import SpGame, SpPly

PI_SUPPORT_MAX = 256

# KataGo's App. D λ scheme adapted (see module docstring): downweight
# positions whose mover is hopelessly lost — outlook < −0.9 maps linearly
# to a 0.1..1 weight, everything above stays 1.
OUTLOOK_WEIGHT_LO = -0.9


def fnv1a64(s: str) -> int:
    """FNV-1a-64 of a string — the repo's standard hash for seeds/stamps."""
    h = 0xCBF29CE484222325
    for b in s.encode():
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def buffer_keep(unit_seed: int, game_index: int, rounds: int, seed: int = 0) -> bool:
    """Replay-buffer trim: True = keep this game. A pool that has aged
    `rounds` generations draws one independent 1-in-3 drop per round
    (keep probability (2/3)**rounds), deterministic in (seed, unit_seed,
    game_index) — so every box re-derives the identical kept set and a
    per-round manifest only has to pin seed + age. rounds=0 keeps all.

    FNV-1a alone is NOT enough here: its low bits correlate across
    strings differing in one early character, which made consecutive
    rounds' drop draws dependent — the fmix64 finalizer decorrelates
    them."""
    for r in range(rounds):
        if _fmix64(fnv1a64(f"{seed}:{r}:{unit_seed}:{game_index}")) % 3 == 0:
            return False
    return True


def _fmix64(h: int) -> int:
    m = (1 << 64) - 1
    h ^= h >> 33
    h = (h * 0xFF51AFD7ED558CCD) & m
    h ^= h >> 33
    h = (h * 0xC4CEB9FE1A85EC53) & m
    h ^= h >> 33
    return h


def ply_trainable(game: SpGame, ply: SpPly) -> bool:
    """PCR filter: a ply trains iff it was searched at the game's slow cap."""
    return ply.sims == game.n_sims


def ply_pi(p: SpPly, legal: List[str]) -> tuple:
    """Dense v3 π → (policy-head slots, probs). `legal` must be the
    replayed position's legal moves in ENGINE movegen order; the count
    assert is the format's order-contract canary — n_pi is written as the
    legal-move count by the driver, so a mismatch means a movegen reorder
    (silent training corruption) and fails loudly here."""
    assert len(p.pi) == len(legal), \
        f"n_pi {len(p.pi)} != {len(legal)} legal moves — π order contract broken"
    slots = [uci_to_index(m, p.mover) for m in legal]
    probs = np.array(p.pi, dtype=np.float64)
    return slots, probs


def outlook_weight(outlook: float) -> float:
    if outlook >= OUTLOOK_WEIGHT_LO:
        return 1.0
    lam = max(0.0, min(1.0, (outlook + 1.0) / 0.1))
    return 0.1 + 0.9 * lam


def soft_target(probs: np.ndarray, temp: float | None) -> np.ndarray:
    """KataGo's auxiliary soft policy target: pi^(1/T), renormalized. T=None
    (or <= 0) leaves the target unchanged, so the key is always present."""
    p = np.asarray(probs, dtype=np.float64)
    if not temp or temp <= 0:
        return p.astype(np.float32)
    soft = np.power(p, 1.0 / temp)
    return (soft / soft.sum()).astype(np.float32)


def selfplay_sample(game: SpGame, ply: int, soft_temp: float | None = None,
                    fast_w: float = 1.0, fast_pol: float | None = None,
                    fast_aux: float = 1.0) -> Dict[str, np.ndarray]:
    """One training sample: the position before `game.plies[ply]`, its
    dense π, aux reply, and outcome label. Same schema and conventions as
    selfplay_samples (which builds every ply of a game); this single-ply
    form is what the training loop's on-the-fly batch sampler calls —
    860K positions must not be materialized up front."""
    p = game.plies[ply]
    assert ply_trainable(game, p), \
        f"ply {ply} is a fast-cap ply (sims {p.sims} != {game.n_sims})"
    board = Board(game.start_fen)
    for i in range(ply):
        board.push(game.plies[i].played_uci)
    p = game.plies[ply]
    side = board.side_to_move
    mask = policy_mask(legal_move_indices(board))
    pol_idx = uci_to_index(p.played_uci, side)
    assert mask[pol_idx], f"played {p.played_uci} not in its legal set"
    assert p.mover == side, "record mover disagrees with replay"
    # tokens BEFORE the aux excursion below: push/pop is not a ring
    # identity once the 8-ply history is full (record evicts, drop only
    # decrements), and encoding after it zeroes the h=7 frame.
    tokens = board.encode()

    slots, probs = ply_pi(p, board.legal_moves)
    assert len(slots) <= PI_SUPPORT_MAX
    probs /= probs.sum()

    pi_slots = np.full(PI_SUPPORT_MAX, slots[0], dtype=np.int64)
    pi_slots[: len(slots)] = slots
    pi_probs = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
    pi_probs[: len(slots)] = probs
    pi_probs_soft = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
    pi_probs_soft[: len(slots)] = soft_target(probs, soft_temp)

    if ply + 1 < len(game.plies):
        nxt = game.plies[ply + 1]
        board.push(p.played_uci)
        legal_after = board.legal_moves
        board.pop()
        aux_idx = uci_to_index(nxt.played_uci, side)   # OUR frame
        aux_mask = policy_mask([uci_to_index(m, side) for m in legal_after])
        aux_w = 1.0
    else:
        aux_idx, aux_mask, aux_w = 0, mask, 0.0

    return {
        "tokens": tokens,
        "mask": mask,
        "pol_idx": np.int64(pol_idx),
        "wdl": wdl_onehot(2.0 * game.z_white - 1.0, side),
        "aux_idx": np.int64(aux_idx),
        "aux_mask": aux_mask,
        "aux_w": np.float32(aux_w),
        "pi_slots": pi_slots,
        "pi_probs": pi_probs,
        "pi_probs_soft": pi_probs_soft,
        "ply_w": np.float32(1.0 if p.sims == game.n_sims else fast_w),
        "pol_w": np.float32(1.0 if p.sims == game.n_sims else
                            (fast_w if fast_pol is None else fast_pol)),
        "sample_w": np.float32(outlook_weight(p.outlook)),
    }


def selfplay_samples(game: SpGame, soft_temp: float | None = None,
                     fast_w: float = 1.0, fast_pol: float | None = None,
                     fast_aux: float = 1.0) -> List[Dict[str, np.ndarray]]:
    """Replay one shard game and build every sample. One Board walk, the
    same shape as data.py replay_samples: each ply's tensors/π are read at
    that ply, the aux target is the next ply's move in OUR frame."""
    board = Board(game.start_fen)
    per_ply = []
    for ply in game.plies:
        side = board.side_to_move
        legal = board.legal_moves
        mask = policy_mask(legal_move_indices(board))
        pol_idx = uci_to_index(ply.played_uci, side)
        assert mask[pol_idx], f"played {ply.played_uci} not in its legal set"
        assert ply.mover == side, "record mover disagrees with replay"

        slots, probs = ply_pi(ply, legal)
        assert len(slots) <= PI_SUPPORT_MAX
        assert (probs >= 0).all() and probs.sum() > 0
        probs /= probs.sum()                    # exact renorm (f16 rounding)

        wdl = wdl_onehot(2.0 * game.z_white - 1.0, side)   # z ∈ {1,.5,0} → {+1,0,−1}
        per_ply.append({
            "tokens": board.encode(),
            "mask": mask,
            "legal": legal,
            "pol_idx": np.int64(pol_idx),
            "wdl": wdl,
            "side": side,
            "pi_slots": slots,
            "pi_probs": probs,
            "pi_probs_soft": soft_target(probs, soft_temp),
            "sample_w": np.float32(outlook_weight(ply.outlook)),
        })
        board.push(ply.played_uci)

    samples = []
    for i, s in enumerate(per_ply):
        if not ply_trainable(game, game.plies[i]):
            continue                  # fast-cap ply: replayed, not trained
        if i + 1 < len(per_ply):
            aux_idx = uci_to_index(game.plies[i + 1].played_uci, s["side"])
            aux_mask = policy_mask(
                [uci_to_index(m, s["side"]) for m in per_ply[i + 1]["legal"]]
            )
            aux_w = 1.0
        else:
            aux_idx, aux_mask, aux_w = 0, s["mask"], 0.0
        pi_slots = np.full(PI_SUPPORT_MAX, s["pi_slots"][0], dtype=np.int64)
        pi_slots[: len(s["pi_slots"])] = s["pi_slots"]
        pi_probs = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
        pi_probs[: len(s["pi_probs"])] = s["pi_probs"]
        pi_probs_soft = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
        pi_probs_soft[: len(s["pi_probs_soft"])] = s["pi_probs_soft"]
        samples.append({
            "tokens": s["tokens"],
            "mask": s["mask"],
            "pol_idx": s["pol_idx"],
            "wdl": s["wdl"],
            "aux_idx": np.int64(aux_idx),
            "aux_mask": aux_mask,
            "aux_w": np.float32(aux_w),
            "pi_slots": pi_slots,
            "pi_probs": pi_probs,
            "pi_probs_soft": pi_probs_soft,
            "ply_w": np.float32(1.0),
            "pol_w": np.float32(1.0),
            "sample_w": s["sample_w"],
        })
    return samples


def stack_selfplay(samples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    return {k: np.stack([s[k] for s in samples]) for k in samples[0]}


class PackedRound:
    """A whole round's games as flat numpy arrays, built incrementally.

    The trainer samples uniformly over ALL trainable plies of the round,
    so the data must be random-accessible for the run's lifetime; but the
    SpGame records are far too heavy to keep (a 250k-game round is ~55 GB
    as Python objects — the memcg OOM that killed the first 250k ingest).
    add_game() therefore converts each record to flat arrays as it is
    streamed in part by part, keeping peak RSS near the packed size
    (~2 GB): per-ply uci as interned strings, per-trainable-ply π as
    int32/float32 runs with offsets, the rest as scalars.

    sample(row) reproduces selfplay_sample's dict exactly (dtype
    included) — parity is pinned by tests/test_packed_round.py.
    """

    def __init__(self) -> None:
        self.start_fens: List[str] = []
        self.z_white: List[float] = []
        self.n_sims: List[int] = []
        self._ucis: List[str] = []          # flat, interned
        self._uci_off: List[int] = [0]      # per-game [start, end) into _ucis
        self._t_game: List[int] = []        # trainable row → game
        self._t_ply: List[int] = []         # … and ply within that game
        self._t_outlook: List[float] = []
        self._t_sims: List[int] = []      # search budget per row (fast vs slow)
        self._slots: List[np.ndarray] = []  # per trainable row
        self._probs: List[np.ndarray] = []

    def add_game(self, g: SpGame, all_plies: bool = False) -> None:
        gi = len(self.start_fens)
        self.start_fens.append(g.start_fen)
        self.z_white.append(g.z_white)
        self.n_sims.append(g.n_sims)
        for p in g.plies:
            self._ucis.append(sys.intern(p.played_uci))
        self._uci_off.append(len(self._ucis))
        # v3 shards store probs only, so slots are derived from the
        # replayed position's legal moves (engine order) — one Board walk.
        board = Board(g.start_fen)
        for i, p in enumerate(g.plies):
            if all_plies or ply_trainable(g, p):
                slots, probs = ply_pi(p, board.legal_moves)
                assert 0 < len(slots) <= PI_SUPPORT_MAX
                self._t_game.append(gi)
                self._t_ply.append(i)
                self._t_outlook.append(p.outlook)
                self._t_sims.append(p.sims)
                self._slots.append(np.array(slots, dtype=np.int32))
                self._probs.append(probs.astype(np.float32))
            board.push(p.played_uci)

    def finalize(self) -> "PackedRound":
        self.z_white = np.array(self.z_white, dtype=np.float64)
        self.n_sims = np.array(self.n_sims, dtype=np.int64)
        self._uci_off = np.array(self._uci_off, dtype=np.int64)
        self.t_game = np.array(self._t_game, dtype=np.int64)
        self.t_ply = np.array(self._t_ply, dtype=np.int64)
        self.t_outlook = np.array(self._t_outlook, dtype=np.float64)
        self.t_sims = np.array(self._t_sims, dtype=np.int32)
        self.slots_off = np.concatenate(
            [[0], np.cumsum([len(s) for s in self._slots])]).astype(np.int64)
        self.slots = (np.concatenate(self._slots) if self._slots
                      else np.zeros(0, np.int32))
        self.probs = (np.concatenate(self._probs) if self._probs
                      else np.zeros(0, np.float32))
        return self

    def __len__(self) -> int:
        t = getattr(self, "t_game", None)
        if isinstance(t, np.ndarray):
            return int(t.shape[0])
        return len(self._t_game)

    @property
    def n_games(self) -> int:
        return len(self.start_fens)

    @property
    def n_recorded_plies(self) -> int:
        # len() works on both the pre-finalize list and the finalized
        # array — truthiness of the array would raise on >1 element
        if not len(self._uci_off):
            return 0
        return int(self._uci_off[-1])

    def sample(self, row: int, soft_temp: float | None = None,
               fast_w: float = 1.0, fast_pol: float | None = None,
               fast_aux: float = 1.0) -> Dict[str, np.ndarray]:
        gi, ply = int(self.t_game[row]), int(self.t_ply[row])
        u0, u1 = int(self._uci_off[gi]), int(self._uci_off[gi + 1])
        board = Board(self.start_fens[gi])
        for k in range(u0, u0 + ply):
            board.push(self._ucis[k])
        side = board.side_to_move
        mask = policy_mask(legal_move_indices(board))
        played = self._ucis[u0 + ply]
        pol_idx = uci_to_index(played, side)
        assert mask[pol_idx], f"played {played} not in its legal set"

        # tokens BEFORE the aux push/pop — same ring caveat as
        # selfplay_sample (see the comment there).
        tokens = board.encode()

        s0, s1 = int(self.slots_off[row]), int(self.slots_off[row + 1])
        slots = self.slots[s0:s1]
        probs = self.probs[s0:s1].astype(np.float64)
        probs /= probs.sum()
        pi_slots = np.full(PI_SUPPORT_MAX, slots[0], dtype=np.int64)
        pi_slots[: len(slots)] = slots
        pi_probs = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
        pi_probs[: len(slots)] = probs
        pi_probs_soft = np.zeros(PI_SUPPORT_MAX, dtype=np.float32)
        pi_probs_soft[: len(slots)] = soft_target(probs, soft_temp)

        if ply + 1 < u1 - u0:
            board.push(played)
            legal_after = board.legal_moves
            board.pop()
            aux_idx = uci_to_index(self._ucis[u0 + ply + 1], side)
            aux_mask = policy_mask(
                [uci_to_index(m, side) for m in legal_after])
            aux_w = 1.0
        else:
            aux_idx, aux_mask, aux_w = 0, mask, 0.0

        return {
            "tokens": tokens,
            "mask": mask,
            "pol_idx": np.int64(pol_idx),
            "wdl": wdl_onehot(2.0 * float(self.z_white[gi]) - 1.0, side),
            "aux_idx": np.int64(aux_idx),
            "aux_mask": aux_mask,
            "aux_w": np.float32(aux_w * (1.0 if self.t_sims[row] == self.n_sims[gi]
                                         else fast_aux)),
            "pi_slots": pi_slots,
            "pi_probs": pi_probs,
            "pi_probs_soft": pi_probs_soft,
            "ply_w": np.float32(1.0 if self.t_sims[row] == self.n_sims[gi]
                                else fast_w),
            "pol_w": np.float32(1.0 if self.t_sims[row] == self.n_sims[gi]
                                else (fast_w if fast_pol is None else fast_pol)),
            "sample_w": np.float32(outlook_weight(float(self.t_outlook[row]))),
        }

    @classmethod
    def concat(cls, parts: List["PackedRound"]) -> "PackedRound":
        """Concatenate finalized fragments (one per unit, from parallel
        packers) into one round. Order is preserved, so row indexing is
        identical to packing serially in unit order."""
        p = cls()
        p.start_fens = [f for q in parts for f in q.start_fens]
        p.z_white = np.concatenate([q.z_white for q in parts])
        p.n_sims = np.concatenate([q.n_sims for q in parts])
        p._ucis = [s for q in parts for s in q._ucis]
        u = 0
        offs = []
        for q in parts:
            offs.append(q._uci_off[:-1] + u)
            u += int(q._uci_off[-1])
        offs.append(np.array([u], dtype=np.int64))
        p._uci_off = np.concatenate(offs)
        g = 0
        tgs = []
        for q in parts:
            tgs.append(q.t_game + g)
            g += q.n_games          # t_game indexes GAMES, not rows
        p.t_game = np.concatenate(tgs)
        p.t_ply = np.concatenate([q.t_ply for q in parts])
        p.t_outlook = np.concatenate([q.t_outlook for q in parts])
        p.t_sims = np.concatenate([q.t_sims for q in parts])
        s = 0
        offs = []
        for q in parts:
            offs.append(q.slots_off[:-1] + s)
            s += int(q.slots_off[-1])
        offs.append(np.array([s], dtype=np.int64))
        p.slots_off = np.concatenate(offs)
        p.slots = np.concatenate([q.slots for q in parts])
        p.probs = np.concatenate([q.probs for q in parts])
        return p

    def save(self, path: str | Path) -> None:
        """Freeze the packed arrays to disk — a launch-time cache so
        hyperparameter sweeps skip the ~10 min parse on re-runs (~2.5 GB
        for a 280k-game round; only labels are stored, never tokens)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            z_white=self.z_white, n_sims=self.n_sims,
            uci_off=self._uci_off, t_game=self.t_game, t_ply=self.t_ply,
            t_outlook=self.t_outlook, t_sims=self.t_sims,
            slots_off=self.slots_off,
            slots=self.slots, probs=self.probs,
            ucis=np.array(self._ucis, dtype="S8"),
            fens=np.array(self.start_fens, dtype="S90"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PackedRound":
        with np.load(path) as z:
            p = cls()
            p.start_fens = [s.decode() for s in z["fens"]]
            p._ucis = [sys.intern(s.decode()) for s in z["ucis"]]
            p.z_white = z["z_white"]
            p.n_sims = z["n_sims"]
            p._uci_off = z["uci_off"]
            p.t_game = z["t_game"]
            p.t_ply = z["t_ply"]
            p.t_outlook = z["t_outlook"]
            p.t_sims = z["t_sims"]
            p.slots_off = z["slots_off"]
            p.slots = z["slots"]
            p.probs = z["probs"]
        return p
