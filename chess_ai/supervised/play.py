"""
chess_ai/supervised/play.py — the 1-ply greedy VALUE player (side track).

User's play design (2026-09-24): no policy head, no search — evaluate
every legal move with the trained value net and play the best:

    v_us(after a) = +1.0        if a mates
                  =  0.0        if a draws on the spot (stalemate / 75m / ...)
                  = −(pW − pL)  otherwise, where pW/pL are the net's WDL
                                probs for the OPPONENT (the side to move in
                                the position after a — the encoder is always
                                mover-relative, so this is just a sign flip)

Terminal moves are scored EXACTLY (not through the net): mate-in-1 must
be found even by an untrained net, and a stalemating move must never be
confused with a winning one. All non-terminal candidates go through the
net in ONE batched forward pass. Ties break to the first legal move in
generation order (deterministic matches).

This is deliberately the dumbest possible consumer of the value net —
the point of the side track is to validate the Phase 2.1 trunk + value
head on real data, not to build a strong player. If this bot already
plays sensible chess, the architecture is sound and Phase 3's search has
something real to sit on.

Known pathology (measured on the mini fixture, expected on real runs):
greedy value play has NO progress pressure — ahead in material, it will
shuffle a piece forever and take a threefold draw instead of pushing a
winning position. That is a limitation of the 1-ply consumer, not of
the value net; report draw rates in any match, don't hide them.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from chess_ai.engine import Board, Termination
from chess_ai.train.trainer import Trainer


class ValueStrategy:
    """1-ply greedy over legal moves by the checkpoint's WDL value.

    A Strategy in the chess_ai.strategies sense: choose_move(board) -> uci.
    The board passed in is left exactly as found (probe moves are popped);
    it can be the match runner's live board.
    """

    def __init__(self, ckpt_path, device: Optional[str] = None,
                 batch: int = 128):
        trainer = Trainer.load(ckpt_path)          # builds net from saved cfg
        # The checkpoint records the device it TRAINED on ("cuda" from a
        # cloud run) — that is not an inference requirement. Default:
        # auto-pick what THIS machine has (MPS → CUDA → CPU); an explicit
        # arg wins. (Bug found first playtest: loading the cloud net on
        # the Mac tried torch.device("cuda") → "CUDA not enabled".)
        from chess_ai.train.trainer import pick_device
        self.device = pick_device(device or "auto")
        self.model = trainer.model.to(self.device).eval()
        self.batch = batch
        self.forward_ms = 0.0                      # crude self-profiling

    def _net_values(self, tokens: List[np.ndarray]) -> np.ndarray:
        """v = pW − pL for each position (MOVER's view), batched.

        Goes through trunk + value head directly: model.forward needs a
        legal-move mask for the policy heads, which the value-only player
        neither has nor needs.
        """
        import time
        out = np.empty(len(tokens), dtype=np.float32)
        for s in range(0, len(tokens), self.batch):
            t = torch.from_numpy(np.stack(tokens[s:s + self.batch])).to(self.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                p = torch.softmax(self.model.value(self.model.trunk(t)), dim=-1)
            self.forward_ms += (time.perf_counter() - t0) * 1e3
            out[s:s + self.batch] = (p[:, 0] - p[:, 2]).cpu().numpy()
        return out

    def choose_move(self, board: Board) -> str:
        legal = board.legal_moves
        candidates: List[str] = []
        tokens: List[np.ndarray] = []
        exact: List[Optional[float]] = []          # terminal scores, else None

        for uci in legal:
            board.push(uci)
            try:
                term = board.termination
                if term == Termination.CHECKMATE:      # we just mated them
                    exact.append(1.0)
                elif term != Termination.PLAYING:      # stalemated / drew
                    exact.append(0.0)
                else:
                    exact.append(None)
                    tokens.append(board.encode())      # opponent to move
            finally:
                board.pop()
            candidates.append(uci)

        if tokens:
            opp_v = self._net_values(tokens)           # opponent's view
            it = iter(opp_v)
            values = [e if e is not None else -float(next(it))
                      for e in exact]                  # flip sign → our view
        else:
            values = [e for e in exact if e is not None]

        # Deterministic argmax: earliest legal move wins ties.
        best_i = max(range(len(candidates)), key=lambda i: values[i])
        return candidates[best_i]
