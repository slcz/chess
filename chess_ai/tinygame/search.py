"""chess_ai/tinygame/search.py — the Gumbel self-play search, ported to
tic-tac-toe (Phase 3.5).

This is `chess_ai/selfplay/gumbel.py` — the executable spec of PLAN Phase
3's agreed search — re-instantiated on the tiny game. The port is
deliberately line-for-line: same class/method names, same control flow,
same pinned items, so the smoke test exercises the search design that
Phase 3 chess self-play actually runs. Mapping to the chess original:

    gumbel.py                         here
    ------------------------------    ------------------------------
    Board / Termination / FFI         TicTacToe (game.py); slots ARE cell
    uci_to_index(move, mover)         identity (no coordinate frame)
    Board.legal_moves (FFI order)     empty cells ascending (the pinned
                                      enumeration order — the RNG draws
                                      Gumbels in this order and every
                                      argmax tie-breaks on it)
    _terminal_wdl: CHECKMATE→loss     WIN → side to move just lost: (0,0,1);
    else draw                         DRAW → (0,1,0)
    board.encode() [64,117]           board.encode() [18]
    Xoshiro256PP / game_seed          imported VERBATIM from rng.py (the
                                      same generator, seeding, uniform and
                                      Gumbel maps the C engine mirrors)

Kept identical (these are the load-bearing pieces the smoke validates):
the RNG consumption order (Gumbels per legal action in enumeration order
at root time; one uniform per SAMPLED move, inverse-CDF in enumeration
order); the halving schedule rounds = (m−1).bit_length(), plays =
ceil(n/(rounds·|A|)), candidate-major descents; Eq. 14 below the root
with the Eq. 10 unvisited-child fallback; σ(q) = (c_visit + max_b N)·
c_scale·q; the sign conventions (stats are each node's OWN mover's view;
a parent negates); π′ = softmax over A₀ of [logp + σ(completedQ)];
resignation at outlook ≤ resign_value (labels: resigner loses; the resign
ply contributes no sample); the F4 handoff (the played child is the next
root). Differences from chess are mechanical only: no history ring in
encode, TERMINAL detection by win lines, and the record stores cells
instead of a FEN.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from chess_ai.selfplay.rng import Xoshiro256PP, game_seed
from chess_ai.tinygame.game import (
    DRAW,
    PLAYING,
    WIN,
    X,
    TicTacToe,
)

NEW = "NEW"
EVALUATED = "EVALUATED"
TERMINAL = "TERMINAL"


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass
class SelfPlayConfig:
    """Knobs — field-for-field gumbel.SelfPlayConfig (defaults included);
    see that class's docstring. m_cap 16 exceeds ttt's branching (≤9), so
    m = min(n, 16, k) collapses to min(n, k) here.

    One ADDITION over the chess config (also a chess knob-to-be, PLAN
    Phase 3): `gumbels_off` — the plan's optional eval variant "m = k,
    pure sequential halving ... removes Gumbel randomness from strength
    measurement". With it, z(a) = logp(a) (no draw, no RNG consumption),
    so the survivor is exactly argmax over π′'s score — the deterministic
    measurement strategy. The 3.5 gate uses it (per-position Gumbel draws
    were measurably flipping near-tie survivors)."""

    n: int = 8
    m_cap: int = 16
    c_visit: float = 50.0
    c_scale: float = 1.0
    c_fpu: float = 0.0
    sample_plies: Optional[int] = None   # None = sample every ply (training)
    sample_mode: str = "pi"              # "pi": ∝ π′ (chess training default,
                                         # 2026-09-27 amendment); "visits":
                                         # ∝ root visit counts — the ORIGINAL
                                         # AGZ/Gumbel-paper sampling (their
                                         # Fig. 8b) and the plan's
                                         # openingSampleK insurance. 3.5
                                         # bring-up found π′ sampling
                                         # self-consistently poisoned at the
                                         # last trap positions (π′ amplifies
                                         # the value head's errors via σ);
                                         # ∝N is flat in the value signal, so
                                         # the correct defense keeps ~1/k play
                                         # probability and clean outcomes
                                         # unwind the poison.
    resign_value: float = -0.90
    gumbels_off: bool = False


class Node:
    """One search-tree node; NO position state (gumbel.py F7) — reached by
    pushing `cell` from the parent on the game's working board. `moves` is
    the legal-slot table in enumeration order; its order IS the RNG/tie
    spec."""

    __slots__ = ("state", "cell", "wdl", "logp", "moves", "children",
                 "visits", "value_sum")

    def __init__(self, state: str, cell: Optional[int] = None):
        self.state = state
        self.cell = cell
        self.wdl: Optional[Tuple[float, float, float]] = None
        self.logp: Dict[int, float] = {}
        self.moves: Tuple[int, ...] = ()
        self.children: Dict[int, "Node"] = {}
        self.visits = 0
        self.value_sum = 0.0

    @property
    def v(self) -> float:
        assert self.wdl is not None
        return self.wdl[0] - self.wdl[2]

    @property
    def q_own(self) -> float:
        assert self.visits > 0
        return self.value_sum / self.visits


class Evaluator:
    """`evaluate(planes, legal_slots, mover) -> (logp dict, wdl triple)` —
    the 3.2a interface, verbatim (net.NumpyEvaluator implements it)."""

    def evaluate(self, planes, legal_slots, mover):
        raise NotImplementedError


class TableEvaluator(Evaluator):
    """Fixture evaluator (gumbel.py's, minus torch): raw prior scores per
    slot, renormalized over the legal slots; WDL from a mover callable."""

    def __init__(self, prior: Optional[Dict[int, float]] = None,
                 value_fn=None):
        self.prior = prior or {}
        self.value_fn = value_fn or (lambda mover: (1 / 3, 1 / 3, 1 / 3))

    def evaluate(self, planes, legal_slots, mover):
        raws = [self.prior.get(s, 0.0) for s in legal_slots]
        m = max(raws)
        lse = m + math.log(sum(math.exp(r - m) for r in raws))
        logp = {s: r - lse for s, r in zip(legal_slots, raws)}
        return logp, self.value_fn(mover)


# ---------------------------------------------------------------------------
# Decision / records (field-for-field gumbel.py; cells replace FEN/UCI)
# ---------------------------------------------------------------------------

@dataclass
class MoveDecision:
    candidate_slots: List[int]
    z: Dict[int, float]
    q_root: Dict[int, float]
    pi_probs: List[float]
    survivor_slot: int
    outlook: float
    candidate_visits: Dict[int, int] = field(default_factory=dict)


@dataclass
class MoveRecord:
    ply: int
    mover: int
    cells_before: Tuple[int, ...]
    played_cell: int                    # −1 for a resign-terminated move
    pi_slots: List[int]
    pi_probs: List[float]
    outlook: float


@dataclass
class GameRecord:
    start_cells: Tuple[int, ...]
    moves: List[MoveRecord] = field(default_factory=list)
    z_white: float = 0.0                # +1 X win / 0 draw / −1 O win
    outcome_kind: str = "natural"       # natural | resign
    resigner: Optional[int] = None
    seed: int = 0
    game_index: int = 0
    n_sims: int = 0


def _softmax(raw: Sequence[float]) -> List[float]:
    m = max(raw)
    exps = [math.exp(r - m) for r in raw]
    total = sum(exps)
    return [e / total for e in exps]


def _terminal_wdl(board: TicTacToe) -> Tuple[float, float, float]:
    """Side-to-move view: the previous mover made a line → this mover lost."""
    if board.termination == WIN:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)              # DRAW (ttt's only other terminal)


# ---------------------------------------------------------------------------
# Game driver + search
# ---------------------------------------------------------------------------

class SelfPlayGame:
    """One game bound to one RNG (gumbel.py §1.2 shape)."""

    def __init__(self, cfg: SelfPlayConfig, seed: int, game_index: int,
                 start_cells: Optional[Tuple[int, ...]] = None):
        self.cfg = cfg
        self.board = TicTacToe(start_cells)
        self.start_cells = self.board.cells
        self.seed = seed
        self.game_index = game_index
        self.rng = Xoshiro256PP(game_seed(seed, game_index))
        self.moves: List[MoveRecord] = []
        self.resigned: Optional[int] = None

    def play(self, evaluator: Evaluator) -> GameRecord:
        """Play to natural termination or resignation (ttt needs no
        max_plies adjudication: 9 plies bound every game)."""
        handoff: Optional[Node] = None
        while True:
            if self.board.termination != PLAYING:
                kind = "natural"
                break
            handoff = self._play_one_move(evaluator, handoff)
            if self.resigned is not None:
                kind = "resign"
                break
        return self._finish(kind)

    def _play_one_move(self, evaluator: Evaluator,
                       handoff: Optional[Node] = None) -> Optional[Node]:
        cfg = self.cfg
        board = self.board
        mover = board.side_to_move
        cells_before = board.cells

        decision, root = self._search_move(evaluator, handoff)

        if decision.outlook <= cfg.resign_value:
            self.resigned = mover
            self.moves.append(MoveRecord(
                ply=board.ply, mover=mover, cells_before=cells_before,
                played_cell=-1,
                pi_slots=decision.candidate_slots,
                pi_probs=decision.pi_probs, outlook=decision.outlook))
            return None

        if cfg.sample_plies is None or board.ply < cfg.sample_plies:
            played = (self._sample_visits(decision, root)
                      if cfg.sample_mode == "visits"
                      else self._sample_pi(decision, root))
        else:
            played = decision.survivor_slot

        self.moves.append(MoveRecord(
            ply=board.ply, mover=mover, cells_before=cells_before,
            played_cell=played,
            pi_slots=decision.candidate_slots, pi_probs=decision.pi_probs,
            outlook=decision.outlook))
        board.push(played)
        return root.children.get(played)      # F4 handoff

    # -- the search (gumbel.py's _search_move, verbatim math) ----------------

    def _search_move(self, evaluator: Evaluator,
                     handoff: Optional[Node] = None):
        cfg = self.cfg
        board = self.board
        root_ply = board.ply

        if handoff is not None:
            root = handoff
        else:
            root = Node(NEW)
            root.moves = board.legal_moves
            self._evaluate(root, evaluator)

        legal = list(root.moves)              # enumeration order
        rank = {s: i for i, s in enumerate(legal)}

        if cfg.gumbels_off:                   # eval variant: z = logp, no draw
            z = {a: root.logp[a] for a in legal}
        else:
            z = {a: root.logp[a] + self.rng.gumbel() for a in legal}
        m = min(cfg.n, cfg.m_cap, len(legal))
        A = sorted(legal, key=lambda s: (-z[s], rank[s]))[:m]
        A0 = list(A)
        rounds = (m - 1).bit_length()

        while len(A) > 1:
            plays = _ceil_div(cfg.n, rounds * len(A))
            for a in A:                       # candidate-major
                for _ in range(plays):
                    self._descend(root, a, root_ply, evaluator)
            A = self._halve(root, A, z, rank, cfg)

        self._rewind(root_ply)
        return self._finalize(root, A0, z, A[0], cfg), root

    def _descend(self, root: Node, forced_slot: int, root_ply: int,
                 evaluator: Evaluator) -> None:
        self._rewind(root_ply)
        path = [root]
        node = root
        slot: Optional[int] = forced_slot
        while True:
            if node.state == TERMINAL:
                self._backup(path, node.wdl)
                return
            assert node.state == EVALUATED, f"selecting from {node.state} node"
            if slot is None:
                slot = self._select_eq14(node)
            child = node.children.get(slot)
            if child is not None:
                self.board.push(child.cell)
            else:
                child = self._make_child(node, slot)
                if child.state == NEW:
                    self._evaluate(child, evaluator)
                    self._backup(path + [child], child.wdl)
                    return
            node = child
            path.append(child)
            slot = None

    def _evaluate(self, node: Node, evaluator: Evaluator) -> None:
        logp, wdl = evaluator.evaluate(self.board.encode(),
                                       node.moves, self.board.side_to_move)
        node.logp = logp
        node.wdl = wdl
        node.state = EVALUATED

    def _make_child(self, parent: Node, slot: int) -> Node:
        self.board.push(slot)
        if self.board.termination != PLAYING:
            child = Node(TERMINAL, slot)
            child.wdl = _terminal_wdl(self.board)
        else:
            child = Node(NEW, slot)
            child.moves = self.board.legal_moves
        parent.children[slot] = child
        return child

    def _select_eq14(self, node: Node) -> int:
        children = node.children
        max_n = 0
        total_n = 0
        for child in children.values():
            if child.visits > max_n:
                max_n = child.visits
            total_n += child.visits
        base = self.cfg.c_visit + max_n
        best_slot = None
        best_score = -math.inf
        for slot in node.moves:               # enumeration order; strict >
            child = children.get(slot)
            if child is not None and child.visits > 0:
                q = -child.q_own               # this node's mover's view
            else:
                q = node.v - self.cfg.c_fpu    # Eq. 10
            n_a = child.visits if child is not None else 0
            score = (node.logp[slot] + base * self.cfg.c_scale * q
                     - n_a / (1 + total_n))
            if score > best_score:
                best_slot, best_score = slot, score
        assert best_slot is not None
        return best_slot

    def _halve(self, root: Node, A: List[int], z: Dict[int, float],
               rank: Dict[int, int], cfg: SelfPlayConfig) -> List[int]:
        max_n = 0
        for a in A:
            max_n = max(max_n, root.children[a].visits)
        base = cfg.c_visit + max_n

        def score(a: int) -> float:
            child = root.children[a]
            return z[a] + base * cfg.c_scale * (-child.q_own)

        ranked = sorted(A, key=lambda a: (-score(a), rank[a]))
        return ranked[:_ceil_div(len(A), 2)]

    def _finalize(self, root: Node, A0: List[int], z: Dict[int, float],
                  survivor: int, cfg: SelfPlayConfig) -> MoveDecision:
        max_n = 0
        qs = {}
        for a in A0:
            child = root.children.get(a)
            if child is not None and child.visits > 0:
                qs[a] = -child.q_own
                max_n = max(max_n, child.visits)
            else:
                qs[a] = root.v - cfg.c_fpu    # Eq. 10 fallback
        base = cfg.c_visit + max_n
        raw = [root.logp[a] + base * cfg.c_scale * qs[a] for a in A0]
        return MoveDecision(
            candidate_slots=list(A0), z=dict(z), q_root=qs,
            pi_probs=_softmax(raw), survivor_slot=survivor,
            outlook=max(qs.values()),
            candidate_visits={a: root.children[a].visits if a in root.children
                              else 0 for a in A0})

    @staticmethod
    def _backup(path: Sequence[Node], wdl_leaf: Tuple[float, float, float]) -> None:
        v = wdl_leaf[0] - wdl_leaf[2]
        sign = 1.0
        for node in reversed(path):
            node.visits += 1
            node.value_sum += sign * v
            sign = -sign

    def _rewind(self, root_ply: int) -> None:
        while self.board.ply > root_ply:
            self.board.pop()

    def _sample_pi(self, decision: MoveDecision, root: Node) -> int:
        """One uniform(), inverse CDF in enumeration order (R6)."""
        rank = {s: i for i, s in enumerate(root.moves)}
        u = self.rng.uniform()
        pairs = sorted(zip(decision.candidate_slots, decision.pi_probs),
                       key=lambda sp: rank[sp[0]])
        cum = 0.0
        for slot, p in pairs:
            cum += p
            if u < cum:
                return slot
        return pairs[-1][0]

    def _sample_visits(self, decision: MoveDecision, root: Node) -> int:
        """∝ visit counts (AGZ/Gumbel-paper Fig. 8b mode): same RNG
        discipline as _sample_pi — one uniform(), inverse CDF in enumeration
        order. Candidates never descended keep weight 0 (they are the m<k
        tail; ttt always has m=k)."""
        rank = {s: i for i, s in enumerate(root.moves)}
        u = self.rng.uniform()
        pairs = sorted(zip(decision.candidate_slots, decision.candidate_visits),
                       key=lambda sp: rank[sp[0]])
        total = sum(n for _, n in pairs)
        if total == 0:
            # A forced position (m = 1): the halving loop never ran and no
            # candidate was descended — the survivor is the only candidate.
            return decision.survivor_slot
        cum = 0.0
        for slot, n in pairs:
            cum += n / total
            if u < cum:
                return slot
        return pairs[-1][0]

    def _finish(self, kind: str) -> GameRecord:
        board = self.board
        if self.resigned is not None:
            z_white = -1.0 if self.resigned == 0 else 1.0
        elif board.winner is not None:
            z_white = 1.0 if board.winner == X else -1.0   # winner: X=1/O=2
        else:
            z_white = 0.0
        return GameRecord(
            start_cells=self.start_cells, moves=self.moves, z_white=z_white,
            outcome_kind=kind, resigner=self.resigned,
            seed=self.seed, game_index=self.game_index, n_sims=self.cfg.n)
