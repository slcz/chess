"""
chess_ai/supervised/dataset.py — filter Lichess PGNs into a compact
"games pack", then stream training positions out of it (side track).

WHY A PACK, NOT TENSORS
=======================
A position's encoded tensor is 64×117 floats ≈ 30 KB — 5M positions
would be ~150 GB. A game's raw record is its move list, ~5 bytes/move.
So the pack stores one line per game:

    1-0<TAB>e2e4 e7e5 g1f3 ...

(the result token + space-separated UCI moves), a few tens of MB for
hundreds of thousands of games, and positions are re-encoded on demand
by replaying through the C engine — the same encode-on-demand design as
the Phase 3 replay buffer, so this side track forces exactly the
streaming-batch trainer path production will need (the plan's stated
bonus). The C encoder is microseconds per position; replay + encode is
noise next to the forward/backward pass.

FILTERS (preprocess knobs, applied with per-reason counters)
============================================================
- both players rated ≥ `min_elo` (provisional '?' tags count as unrated)
- time control base ≥ `min_base_sec` (drops bullet; blitz 180+ and up
  survive — outcome labels in 60-second scrabble are mostly noise)
- ply count within [min_plies, max_plies] (drops aborted 1-2 ply games
  and marathon takeback-fests)
- optional: Termination == "Normal" (drops time-forfeit results, where
  the final positions often aren't objectively decided — noise at the
  label, not at the position; default OFF, it costs a lot of games)

SAMPLING
========
Uniform over POSITIONS (games weighted by length), matching the
ReplayBuffer convention. One replaying Board per worker; optional thread
pool + a one-batch-ahead prefetcher (cffi releases the GIL in the C
calls, so overlap with the GPU step is real). Value-only runs carry the
full trainer schema anyway — real legal-move mask + the played move's
slot — so the control metric "policy head stays at chance" is visible
and the same train_step is reused unchanged.
"""
from __future__ import annotations

import json
import random
import threading
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from chess_ai.engine import Board
from chess_ai.net.features import (
    NN_TOKEN_DIM,
    POLICY_SIZE,
    legal_move_indices,
    policy_mask,
    uci_to_index,
    wdl_onehot,
)
from chess_ai.supervised.pgn import (
    RESULT_SCORES,
    ParseError,
    iter_pgn_games,
    open_pgn,
)

# One game = (white-POV outcome, UCI move tuple).
Game = Tuple[float, Tuple[str, ...]]

# Every pack game starts here: preprocess drops the rare SetUp/FEN
# custom-start games (counted), so replay never needs a per-game start.
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


# ---------------------------------------------------------------------------
# Preprocess: PGN dump → filtered pack (+ stats json)
# ---------------------------------------------------------------------------

class PreprocessStats:
    def __init__(self):
        self.seen = 0
        self.kept = 0
        self.positions = 0
        self.dropped = {
            "parse_error": 0, "no_result": 0, "unrated": 0, "low_elo": 0,
            "time_control": 0, "ply_range": 0, "time_forfeit": 0,
            "custom_start": 0,
        }
        self.outcomes = {"white": 0, "draw": 0, "black": 0}
        self.elo_sum = 0
        self.elo_n = 0
        self.elo_min = 9999
        self.elo_max = 0

    def note_game(self, outcome: float, elos: Tuple[int, int], plies: int):
        self.kept += 1
        self.positions += plies
        self.outcomes["white" if outcome > 0 else
                      "black" if outcome < 0 else "draw"] += 1
        self.elo_sum += sum(elos)
        self.elo_n += 2
        self.elo_min = min(self.elo_min, *elos)
        self.elo_max = max(self.elo_max, *elos)

    def as_dict(self) -> dict:
        d = {k: v for k, v in vars(self).items()}
        d["avg_elo"] = round(self.elo_sum / max(1, self.elo_n), 1)
        d.pop("elo_sum")
        d.pop("elo_n")
        return d


def _parse_elo(tag: str) -> Optional[int]:
    return int(tag) if tag.isdigit() else None


def _parse_base_sec(tc: str) -> Optional[int]:
    """TimeControl '300+2' → 300; '-'/'?' (unrated/correspondence) → None."""
    base = tc.split("+")[0]
    return int(base) if base.isdigit() else None


def preprocess(
    pgn_path: str,
    out_path: str,
    min_elo: int = 1600,
    min_base_sec: int = 180,
    min_plies: int = 10,
    max_plies: int = 300,
    drop_time_forfeits: bool = False,
    max_games: int = 0,
    max_positions: int = 0,
) -> dict:
    """Stream a Lichess dump → filtered pack file + stats (returns stats).

    `max_games`/`max_positions` (>0) stop early — a prefix of the month
    is all a pilot needs. Stats land next to the pack as
    `<out>.stats.json` so the filters that made a pack are never a
    mystery later.
    """
    stats = PreprocessStats()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def on_error(headers, exc):
        stats.dropped["parse_error"] += 1
        if stats.dropped["parse_error"] <= 3:
            print(f"  parse drop: {exc} "
                  f"(game vs {headers.get('WhiteElo', '?')}/{headers.get('BlackElo', '?')})")

    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        with open_pgn(pgn_path) as stream:
            for headers, moves in iter_pgn_games(stream, on_error=on_error):
                stats.seen += 1
                result = headers.get("Result")
                if result not in RESULT_SCORES:
                    stats.dropped["no_result"] += 1
                    continue
                if headers.get("SetUp") == "1" or headers.get("FEN"):
                    stats.dropped["custom_start"] += 1
                    continue
                we, be = (_parse_elo(headers.get("WhiteElo", "?")),
                          _parse_elo(headers.get("BlackElo", "?")))
                if we is None or be is None:
                    stats.dropped["unrated"] += 1
                    continue
                if we < min_elo or be < min_elo:
                    stats.dropped["low_elo"] += 1
                    continue
                base = _parse_base_sec(headers.get("TimeControl", "?"))
                if base is None or base < min_base_sec:
                    stats.dropped["time_control"] += 1
                    continue
                if not (min_plies <= len(moves) <= max_plies):
                    stats.dropped["ply_range"] += 1
                    continue
                if drop_time_forfeits and headers.get("Termination") != "Normal":
                    stats.dropped["time_forfeit"] += 1
                    continue

                token = {1.0: "1-0", -1.0: "0-1", 0.0: "1/2-1/2"}[
                    RESULT_SCORES[result]]
                out.write(f"{token}\t{' '.join(moves)}\n")
                stats.note_game(RESULT_SCORES[result], (we, be), len(moves))
                n += 1
                if n % 5000 == 0:
                    print(f"  {stats.seen} seen, {stats.kept} kept, "
                          f"{stats.positions} positions")
                if max_games and stats.kept >= max_games:
                    break
                if max_positions and stats.positions >= max_positions:
                    break

    meta = {
        "source": pgn_path,
        "filters": {
            "min_elo": min_elo, "min_base_sec": min_base_sec,
            "min_plies": min_plies, "max_plies": max_plies,
            "drop_time_forfeits": drop_time_forfeits,
        },
        "stats": stats.as_dict(),
    }
    Path(str(out_path) + ".stats.json").write_text(json.dumps(meta, indent=2))
    print(f"pack written: {out_path}  {json.dumps(stats.as_dict())}")
    return meta


# ---------------------------------------------------------------------------
# Pack loading + position sampling
# ---------------------------------------------------------------------------

def load_pack(path: str) -> List[Game]:
    games: List[Game] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token, _, moves = line.rstrip("\n").partition("\t")
            mv = tuple(moves.split())
            if mv:
                games.append((RESULT_SCORES[token], mv))
    return games


def replay_position(game: Game, ply: int, board: Board) -> Tuple[np.ndarray, ...]:
    """Encode position at `ply` of `game` on `board` (replayed from start).

    Returns (tokens[64,117], mask[4240], pol_idx, wdl[3]) — trainer
    schema minus aux (value-only runs carry aux dummies added by the
    sampler). `board` is left at the position; the caller reuses it for
    the next sample via the next set_fen.
    """
    outcome, moves = game
    board.set_fen(START_FEN)
    for m in moves[:ply]:
        board.push(m)
    side = board.side_to_move
    tokens = board.encode()
    mask = policy_mask(legal_move_indices(board))
    pol_idx = uci_to_index(moves[ply], side)
    assert mask[pol_idx], f"dataset move {moves[ply]} illegal at ply {ply}"
    return tokens, mask, np.int64(pol_idx), wdl_onehot(outcome, side)


class PositionSampler:
    """Uniform-over-positions batches, encoded on demand from the pack."""

    def __init__(self, games: Sequence[Game], seed: int = 0,
                 n_workers: int = 1):
        self.games = list(games)
        self.prefix = np.cumsum([len(g[1]) for g in self.games])
        self.n_workers = max(1, n_workers)
        self._boards = [Board() for _ in range(self.n_workers)]
        self._rngs = [random.Random(seed + 1000 * w)
                      for w in range(self.n_workers)]

    @property
    def n_positions(self) -> int:
        return int(self.prefix[-1]) if len(self.prefix) else 0

    def _draw(self, rng: random.Random) -> Tuple[int, int]:
        """One uniform position → (game index, ply) via prefix-sum search."""
        p = rng.randrange(self.n_positions)
        gi = int(np.searchsorted(self.prefix, p, side="right"))
        base = self.prefix[gi - 1] if gi else 0
        return gi, p - base

    def sample_batch(self, n: int) -> Dict[str, np.ndarray]:
        toks, masks, idxs, wdls = [], [], [], []
        if self.n_workers == 1:
            parts = [(self._rngs[0], self._boards[0], n)]
        else:
            chunks = [n // self.n_workers] * self.n_workers
            for i in range(n - sum(chunks)):
                chunks[i] += 1
            parts = [(self._rngs[w], self._boards[w], c)
                     for w, c in enumerate(chunks) if c]
        for rng, board, count in parts:
            for _ in range(count):
                gi, ply = self._draw(rng)
                t, m, p, v = replay_position(self.games[gi], ply, board)
                toks.append(t); masks.append(m); idxs.append(p); wdls.append(v)
        return {
            "tokens": np.stack(toks),
            "mask": np.stack(masks),
            "pol_idx": np.stack(idxs),
            "wdl": np.stack(wdls),
            # Value-only runs: aux head contributes nothing (weight 0);
            # dummies keep the trainer schema intact.
            "aux_idx": np.zeros(n, dtype=np.int64),
            "aux_mask": np.stack(masks),
            "aux_w": np.zeros(n, dtype=np.float32),
        }

    def fixed_positions(self, n: int, seed: int = 1234) -> List[Tuple[int, int]]:
        """A deterministic (game, ply) list for the validation split."""
        rng = random.Random(seed)
        return [self._draw(rng) for _ in range(n)]


class BatchPrefetcher:
    """Builds batches on a worker thread, one batch ahead of the GPU.

    Overlaps encode/replay CPU work with the forward/backward step. The
    feed stops on exception (re-raised in the consumer) or sentinel.
    """

    _STOP = object()

    def __init__(self, sampler: PositionSampler, batch_size: int, prefetch: int = 2):
        self.sampler = sampler
        self.batch_size = batch_size
        self._q: Queue = Queue(maxsize=prefetch)
        self._err: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._work, daemon=True)
        self._thread.start()

    def _work(self):
        try:
            while True:
                self._q.put(self.sampler.sample_batch(self.batch_size))
        except BaseException as exc:            # surface in the consumer
            self._err = exc
            self._q.put(self._STOP)

    def next(self) -> Dict[str, np.ndarray]:
        item = self._q.get()
        if item is self._STOP:
            raise self._err or RuntimeError("prefetcher stopped")
        return item
