"""
chess_ai/train/data.py — game records, the replay buffer, and sample
building (Phase 2.2).

DATA MODEL
==========
A `GameRecord` is the compact, self-describing unit of self-play data:
start FEN + the UCI move list + the game outcome (white POV). That is ALL
it stores — no feature tensors. Encoded tensors are rebuilt on demand from
the record by replaying it through the C engine (Board.encode()), because
(=30 KB/position) would make a 500k-position replay buffer ~15 GB, while
the raw records for the same window are a few tens of MB. The C encoder
is microseconds per position, so on-the-fly encoding stays trivially
cheap next to the forward/backward pass. (If profiling ever disagrees,
an encode cache can sit in front of this — same interface.)

SAMPLE SCHEMA (what a position contributes to training)
=======================================================
For every ply where a move was chosen (i.e., all but the terminal
position — you don't train a policy on a position with no decision):

    tokens   [64, 117]  float32  — C encoder output (chess_nn.h layout)
    mask     [4240]     bool     — legal-move mask (True = legal)
    pol_idx  []         int64    — played move's slot: the Phase-2 stand-in
                                   for the search-improved target π′; sparse
                                   (index + implicit weight 1) so Phase 3
                                   can swap in real π′ without a schema change
    wdl      [3]        float32  — game outcome z as W/D/L one-hot from the
                                   MOVER's perspective
    aux_*                        — opponent-reply target (KataGo aux head):
                                   the move actually played at ply+1, encoded
                                   in OUR mover-relative frame (see
                                   model.PolicyHead docstring), with the
                                   reply position's legal-move mask. The
                                   game's last ply has aux_w = 0 (no reply
                                   exists) — its loss weight is zeroed
                                   rather than the sample dropped, so the
                                   buffer's positions and the trainer's
                                   bookkeeping stay aligned.

`build_tensors()` materializes a list of records into stacked numpy
arrays ready for torch; `generate_random_games()` produces the fixed
seeded games used by the Phase 2 gate (memorization test) and later smoke
tests. Random-play games are ONLY a pipeline exerciser — real training
data comes from Gumbel self-play in Phase 3.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chess_ai.engine import Board, Termination
from chess_ai.net.features import (
    NN_TOKEN_DIM,
    POLICY_SIZE,
    legal_move_indices,
    policy_mask,
    uci_to_index,
    wdl_onehot,
)

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass(frozen=True)
class GameRecord:
    """One complete game, as compact as it gets (see module docstring)."""

    start_fen: str
    moves: Tuple[str, ...]
    outcome: float            # +1 white win, −1 black win, 0 draw


# ---------------------------------------------------------------------------
# Sample building
# ---------------------------------------------------------------------------

def replay_samples(
    record: GameRecord,
    value_fn: Optional[Callable[[Board, int], np.ndarray]] = None,
) -> List[Dict[str, np.ndarray]]:
    """Replay one game and build every training sample (schema above).

    One Board walk per game: each ply's tokens/mask/policy target are read
    at that ply, and the aux target for ply i is simply ply i+1's move and
    mask — already in hand from the same walk.

    `value_fn(board, side_to_move) -> wdl[3]` overrides the game-outcome
    WDL label per position (used by structured fixtures whose value target
    must be a deterministic function of the position — see
    material_sign_wdl). Default: the game outcome back-propagated to every
    ply, the AGZ convention used by real self-play data (Phase 3).
    """
    board = Board(record.start_fen)
    per_ply: List[Dict[str, np.ndarray]] = []

    for ply, move in enumerate(record.moves):
        side = board.side_to_move
        tokens = board.encode()
        legal = board.legal_moves
        mask = policy_mask(legal_move_indices(board))
        pol_idx = uci_to_index(move, side)
        assert mask[pol_idx], f"played move {move} not in its own legal set"
        wdl = value_fn(board, side) if value_fn is not None \
            else wdl_onehot(record.outcome, side)
        per_ply.append({
            "tokens": tokens,
            "mask": mask,
            "legal": legal,            # raw UCI list; aux masks are built
            "pol_idx": np.int64(pol_idx),   # from it in the PREVIOUS ply's
            "wdl": wdl,                # frame, see below
            "side": side,
        })
        board.push(move)

    # Stitch in the aux-reply targets from the neighboring ply. Both the
    # target index AND its mask must live in OUR (ply-i mover's) frame:
    # the aux head reads our trunk and predicts the enemy's move in our
    # coordinates, so the reply's legal set is re-indexed with OUR side —
    # per_ply[i+1]["mask"] would be the opponent's orientation and wrong.
    samples: List[Dict[str, np.ndarray]] = []
    for i, s in enumerate(per_ply):
        if i + 1 < len(per_ply):
            reply_move = record.moves[i + 1]
            aux_idx = uci_to_index(reply_move, s["side"])   # OUR frame
            aux_mask = policy_mask(
                [uci_to_index(m, s["side"]) for m in per_ply[i + 1]["legal"]]
            )
            assert aux_mask[aux_idx], "reply move not in its own aux mask"
            aux_w = 1.0
        else:                                              # last move: no reply
            aux_idx, aux_mask, aux_w = 0, s["mask"], 0.0
        samples.append({
            "tokens": s["tokens"],
            "mask": s["mask"],
            "pol_idx": s["pol_idx"],
            "wdl": s["wdl"],
            "side": s["side"],      # consumed by build_tensors' mover_filter,
            "aux_idx": np.int64(aux_idx),   # not stacked into the output
            "aux_mask": aux_mask,
            "aux_w": np.float32(aux_w),
        })
    return samples


def build_tensors(
    records: Sequence[GameRecord],
    mover_filter: Optional[int] = None,
    value_fn: Optional[Callable[[Board, int], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Stack every sample of every record into training arrays.

    `mover_filter` (0 = white, 1 = black) keeps only positions where that
    side moved — for fixtures where one side plays a *learnable* strategy
    and the other pure noise (e.g. random-vs-greedy): train on the
    strategist's moves only. NOTE: with a filter, the aux target at a
    kept ply is the *filtered-out* opponent's reply — random noise by
    construction; set TrainConfig.aux_reply_weight=0 for such fixtures.

    `value_fn` overrides the per-position value label (see replay_samples;
    pair with material_sign_wdl for the Phase 2 gate fixture).
    """
    samples = [
        s
        for r in records
        for s in replay_samples(r, value_fn=value_fn)
        if mover_filter is None or s["side"] == mover_filter
    ]
    if not samples:
        raise ValueError("no positions in records")
    return {
        "tokens": np.stack([s["tokens"] for s in samples]),
        "mask": np.stack([s["mask"] for s in samples]),
        "pol_idx": np.stack([s["pol_idx"] for s in samples]),
        "wdl": np.stack([s["wdl"] for s in samples]),
        "aux_idx": np.stack([s["aux_idx"] for s in samples]),
        "aux_mask": np.stack([s["aux_mask"] for s in samples]),
        "aux_w": np.stack([s["aux_w"] for s in samples]),
    }


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Rolling window of recent games (PLAN: 40–50k games at full scale).

    Stores GameRecords only (compact); sampling picks positions uniformly
    across the window — uniform-over-positions matches AGZ's "sample from
    recent games" closely enough at these scales and avoids
    short-game/long-game weighting questions for now.
    """

    def __init__(self, max_games: int):
        self._records: deque = deque(maxlen=max_games)

    def add(self, record: GameRecord) -> None:
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def total_positions(self) -> int:
        return sum(len(r.moves) for r in self._records)

    def sample_positions(self, rng: random.Random, n: int) -> List[Tuple[GameRecord, int]]:
        """n (record, ply) pairs, uniformly over positions in the window."""
        total = self.total_positions
        picks = [rng.randrange(total) for _ in range(n)]
        out: List[Tuple[GameRecord, int]] = []
        for p in picks:
            for r in self._records:
                if p < len(r.moves):
                    out.append((r, p))
                    break
                p -= len(r.moves)
            else:  # only if the window became empty mid-call
                raise RuntimeError("sampled from an empty buffer")
        return out


# ---------------------------------------------------------------------------
# Structured-fixture value target (Phase 2 gate)
# ---------------------------------------------------------------------------

# Classical centipawn values, pawn..king — same table as strategies.py uses.
PIECE_VALUES_CP = [100, 320, 330, 500, 900, 0]


def material_sign_wdl(board: Board, side: int) -> np.ndarray:
    """W/D/L label = sign of the material balance from `side`'s view.

    A *deterministic function of the position* — the learnable-by-design
    value target for the Phase 2 gate fixture. Why not the game outcome
    there: under random-vs-greedy play, whether a game ends in mate or
    shuffles into threefold depends on the random side's FUTURE moves, so
    the outcome label has irreducible noise from any early position (the
    net sat at the class prior; the 2026-09-24 gate calibration). Material
    sign populates all three W/D/L classes and is fully fittable, which is
    what the gate's value pathway check needs. Test fixture ONLY — real
    training targets are game outcomes from self-play (Phase 3).
    """
    balance = 0
    for sq in range(64):
        pc = board.piece_at(sq)
        if pc < 12:  # not EMPTY
            v = PIECE_VALUES_CP[pc % 6]
            balance += v if pc // 6 == side else -v
    if balance > 0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if balance < 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Fixed-data generators (gate tests, smoke tests)
# ---------------------------------------------------------------------------

def generate_random_games(
    n_games: int, seed: int, start_fen: str = START_FEN, max_plies: int = 200
) -> List[GameRecord]:
    """Seeded uniform-random legal play — a deterministic data fixture.

    Adjudication follows the engine's own termination rules (mate,
    stalemate, dead material, claimable threefold/50-move auto-end per the
    Phase 1.3 design); hitting `max_plies` scores the game as a draw.
    Deterministic given `seed`, so a failed gate reproduces exactly.

    NOTE: uniform-random labels are the *maximally unlearnable* case —
    the policy target has no structure, so learning is pure rote and its
    speed scales with exposures per position (the 2026-09-24 gate stall).
    `generate_strategy_games` below is the structured alternative.
    """
    rng = random.Random(seed)
    records = []
    for _ in range(n_games):
        board = Board(start_fen)
        moves: List[str] = []
        while board.termination == Termination.PLAYING and board.ply < max_plies:
            move = rng.choice(board.legal_moves)
            board.push(move)
            moves.append(move)
        outcome = 0.0
        if board.termination == Termination.CHECKMATE:
            outcome = 1.0 if board.winner == 0 else -1.0
        records.append(GameRecord(start_fen, tuple(moves), outcome))
    return records


def generate_strategy_games(
    n_games: int,
    seed: int,
    strategy_factory: Callable[[int], Any],
    start_fen: str = START_FEN,
    max_plies: int = 150,
    opponent_factory: Optional[Callable[[int], Any]] = None,
    strategist_side: int = 0,
) -> List[GameRecord]:
    """Games where one (or both) side(s) play a seeded strategy — structured
    fixtures.

    Unlike uniform-random play, a strategy makes the policy target a
    *learnable function of the position* (e.g. GreedyMaterialStrategy:
    "take the best capture") instead of an arbitrary label — so the net
    can generalize rather than rote-memorize, and learning speed is not
    hostage to exposures-per-position. Still deterministic given `seed`
    (game i uses seed+i per side; the strategies' own rngs provide
    diversity), and still zero-knowledge: our own dummy strategies, no
    external data.

    `opponent_factory` (e.g. RandomStrategy): if given, the OTHER side
    plays it — the strategist supplies the labels, the opponent supplies
    game diversity (the Phase 2 gate fixture). Train on the strategist's
    plies via build_tensors(records, mover_filter=strategist_side).
    The strategist is WHITE by default: the encoder is mover-relative, so
    color balance is immaterial, and a single filter value keeps the gate
    simple.
    """
    records = []
    for i in range(n_games):
        strategist = strategy_factory(seed + i)
        opponent = opponent_factory(seed + i) if opponent_factory else None
        board = Board(start_fen)
        moves: List[str] = []
        while board.termination == Termination.PLAYING and board.ply < max_plies:
            mover = board.side_to_move
            if mover == strategist_side or opponent is None:
                move = strategist.choose_move(board)
            else:
                move = opponent.choose_move(board)
            board.push(move)
            moves.append(move)
        outcome = 0.0
        if board.termination == Termination.CHECKMATE:
            outcome = 1.0 if board.winner == 0 else -1.0
        records.append(GameRecord(start_fen, tuple(moves), outcome))
    return records
