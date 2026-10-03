"""
chess_ai/selfplay/gumbel.py — sequential Python Gumbel reference (Phase 3.2a).

The executable spec of the self-play search and the comparison oracle for the
C engine. Simplicity over performance (user decision 2026-09-27): every NN
evaluation is a **synchronous size-1 call at the expansion point** — no
request queues, no batching, no park/resume, no state machine. The file is
literally the high-level algorithm (PLAN Phase 3, "Agreed search design",
Danihelka et al. ICLR 2022) plus the inference engine behind one narrow
interface (`Evaluator.evaluate`).

Comparison contract with the C engine (PARALLEL_DESIGN R6):

- **L2 (exact):** the C engine's *sequential mode* (one worker, inline
  evaluator, CPU fp32) must reproduce this file's games exactly. The pinned
  items are: the RNG consumption order (rng.py); the descent order —
  candidate-major per round (`for a in A: plays × descend(a)`, A in
  z-descending order, movegen-order tie-breaks everywhere); synchronous
  expansion (each descent evaluates exactly the one fresh leaf it reaches,
  then backs up immediately); and IEEE-double backup accumulation in descent
  order.
- **L3 (statistical):** the C *parallel* engine intentionally diverges from
  this reference under batching (sibling descents of one wave touch disjoint
  subtrees, so the divergence is bounded); it is verified by invariants,
  statistical agreement over seeded runs, and deterministic-interleave
  goldens — not by exact diff.

Search math (verbatim from PLAN; R7 for the log-space convention):

    m      = min(n, m_cap, #legal)          # candidates
    g(a)   ~ Gumbel(0,1) per legal action   # movegen enumeration order (R6)
    z(a)   = logp(a) + g(a)                 # logp = net policy_logp (legal slots)
    A      = top-m by z                     # ties: movegen order
    rounds = ceil(log2 m)                   # from the ORIGINAL m; (m-1).bit_length()
    while |A| > 1:
        plays = ceil(n / (rounds·|A|))      # PLAN formula, kept as-is (~1.3–2n
                                            # descents/move; confirmed 2026-09-27)
        for a in A: plays × descend(child(s, a)); back up, sign-flip every ply
        A = top ceil(|A|/2) by z(a) + σ(q(a))    # σ(q) = (C_VISIT+max_b N)·C_SCALE·q
    survivor = last A[0]; π′ = softmax over A₀ of [logp + σ(completedQ)]

    below the root (Eq. 14): argmax over legal a of
        π′(a) − N(a)/(1+ΣN),  π′ = softmax_a[logp(a) + σ(completedQ(a))]
        completedQ(a) = q(a) if N(a)>0 else v_mix (Appendix D, Eq. 33;
                        degenerates to v̂(node) − c_fpu at ΣN = 0)
    The argmax runs on the π′ PROBABILITIES (max-shifted softmax, f64,
    summation pinned to enumeration order), not on the logits — mctx
    parity; logit-space argmax diverges once logit gaps exceed ~1 nat.
    σ's max_b N(b) is taken over the node's created children. All argmax/sort
    ties break by movegen enumeration order (ascending movegen rank).

Sign conventions (load-bearing — see SelfPlayGame._backup):

- A node's stats are from ITS OWN mover's view: value_sum accumulates the
  leaf value (pW−pL, leaf-mover view) with a sign flip per ply.
- A parent reading a child's q for selection/targets negates it:
  q_parent_view(a) = −value_sum[child]/visits[child].
- v̂(node) is the node's own net WDL value (pW−pL), stats-independent.

Board handling: the search walks the game's working Board with push/pop
(make/unmake); nodes store no position state (PARALLEL_DESIGN F7). Each
descent rewinds to the root ply first; encoding happens with the board
standing on the requested node, so the 8-ply history ring is correct by
construction. The played move's child node — always evaluated — is handed to
the next search as its root (F4 handoff), so every root after move 1 costs
zero evaluations.

RNG consumption order (rng.py, mirrored in C): no-resign coin at game
creation; per move, Gumbels for all legal actions in movegen order at root
time; one uniform() for the π′ sampling draw when the ply is sampled; nothing
for deterministic plies. There are NO other special cases: a forced position
(k == 1 legal move) flows through the same path as any move — m = 1, the
halving loop never runs, π′ = softmax over the single candidate = exactly 1.0,
and the outlook comes from the Eq. 10 fallback (v̂(root) − c_fpu, since no
candidate was visited) — so resignation works uniformly there too. The handoff
usually makes even that free: after a searched move the forced position's node
already exists and is evaluated.

Resignation (R4): at every move, outlook = max over candidates of completedQ
(root-mover view; Eq. 10 fallback for unvisited candidates). outlook ≤
resign_value → resign (label: loss for the resigner).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from chess_ai.engine import Board, Termination
from chess_ai.net.features import uci_to_index
from chess_ai.selfplay.rng import Xoshiro256PP, game_seed

NEW = "NEW"            # created, never evaluated
EVALUATED = "EVALUATED"  # logp + wdl known; may be selected from
TERMINAL = "TERMINAL"  # game-theoretic value known (mate/draw); never evaluated


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SelfPlayConfig:
    """Knobs of the reference; defaults per PARALLEL_DESIGN §4.0 / PLAN Phase 3.
    n must be ≥ 2 (m ≤ n is what guarantees every candidate a round-1 visit
    when k ≥ 2; k == 1 gives m = 1 and simply skips the halving loop).
    n is a tuning knob — to be swept on salad (2026-09-27); 8 is a placeholder
    default, not a recommendation."""

    n: int = 8                      # sims per move (self-play 2–32; eval 100–200)
    m_cap: int = 16                 # candidate cap: m = min(n, m_cap, #legal)
    c_visit: float = 50.0
    c_scale: float = 1.0
    c_fpu: float = 0.0              # Eq. 10 reduction (KataGo knob, default = paper)
    sample_plies: Optional[int] = None  # sample played move ∝ π′ on these plies;
                                        # None = every ply (training default, 2026-09-27:
                                        # the deterministic survivor is for eval — R5).
                                        # Eval matches will set 10 (sampled seed phase).
    resign_value: float = -0.90     # resign at best-candidate q ≤ this (~5% win prob)


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------

class Node:
    """One search-tree node. Holds NO position state (F7): the position is
    reached by pushing `uci` from the parent on the game's working Board.

    logp is the sparse legal-masked policy (log-softmax over legal slots, from
    the net's policy_logp — log space everywhere, R7). `slot_to_uci` is built
    once at expansion from Board.legal_moves in FFI generation order; its key
    order IS the enumeration order the RNG and tie-breaks are specified over.
    """

    __slots__ = ("state", "uci", "wdl", "logp", "slot_to_uci", "children",
                 "visits", "value_sum")

    def __init__(self, state: str, uci: Optional[str] = None):
        self.state = state
        self.uci = uci                  # move from the parent leading here (root: None)
        self.wdl: Optional[Tuple[float, float, float]] = None   # own mover's view
        self.logp: Dict[int, float] = {}                        # slot → log-prob
        self.slot_to_uci: Dict[int, str] = {}                   # movegen order (dict order)
        self.children: Dict[int, "Node"] = {}                   # slot → child
        self.visits = 0
        self.value_sum = 0.0

    @property
    def v(self) -> float:
        """The node's own net value (pW−pL), own mover's view."""
        assert self.wdl is not None
        return self.wdl[0] - self.wdl[2]

    @property
    def q_own(self) -> float:
        """Mean backed-up value, own mover's view."""
        assert self.visits > 0
        return self.value_sum / self.visits


# ---------------------------------------------------------------------------
# Evaluators (one synchronous size-1 call per position)
# ---------------------------------------------------------------------------

class Evaluator:
    """Inference interface: `evaluate(planes, legal_slots, mover)` ->
    (logp, wdl), where logp is a dict slot → log-prob over the legal slots
    (the net's masked policy_logp contract) and wdl is the value triple from
    the mover's view. `mover` (0 white / 1 black) is ignored by a real net —
    the planes are already mover-relative — but lets fixture value functions
    know the perspective."""

    def evaluate(self, planes: np.ndarray, legal_slots: Sequence[int],
                 mover: int) -> Tuple[Dict[int, float],
                                      Tuple[float, float, float]]:
        raise NotImplementedError


class TableEvaluator(Evaluator):
    """Torch-free fixture evaluator: raw prior scores per policy slot
    (default 0), renormalized over the legal slots into a log-softmax (same
    contract as the net's policy_logp); WDL from a callable of the mover."""

    def __init__(
        self,
        prior: Optional[Dict[int, float]] = None,
        value_fn: Optional[Callable[[int], Tuple[float, float, float]]] = None,
    ):
        self.prior = prior or {}
        self.value_fn = value_fn or (lambda mover: (1 / 3, 1 / 3, 1 / 3))

    def evaluate(self, planes, legal_slots, mover):
        raws = [self.prior.get(s, 0.0) for s in legal_slots]
        m = max(raws)
        lse = m + math.log(sum(math.exp(r - m) for r in raws))
        logp = {s: r - lse for s, r in zip(legal_slots, raws)}
        return logp, self.value_fn(mover)


# ---------------------------------------------------------------------------
# Decision / records
# ---------------------------------------------------------------------------

@dataclass
class MoveDecision:
    candidate_slots: List[int]                 # A₀, z-descending
    z: Dict[int, float]                        # gumbel scores z(a) = logp + g
    q_root: Dict[int, float]                   # candidate slot → completedQ, root view
    pi_probs: List[float]                      # π′, aligned with candidate_slots
    survivor_slot: int                         # sequential-halving winner
    outlook: float                             # max completedQ over A₀ (resign input)


@dataclass
class MoveRecord:
    ply: int
    mover: int                                 # 0 white / 1 black
    fen_before: str
    played_uci: str                            # "" for a resign-terminated move
    played_slot: int                           # −1 for a resign-terminated move
    pi_slots: List[int]                        # A₀ (z-descending), π′ support
    pi_probs: List[float]
    outlook: float                             # max completedQ over A₀


@dataclass
class GameRecord:
    start_fen: str
    moves: List[MoveRecord] = field(default_factory=list)
    z_white: float = 0.0                       # +1 white win / 0 draw / −1 black win
    outcome_kind: str = "natural"              # natural | resign | adjudicated
    resigner: Optional[int] = None
    seed: int = 0
    game_index: int = 0
    n_sims: int = 0


def _softmax(raw: Sequence[float]) -> List[float]:
    m = max(raw)
    exps = [math.exp(r - m) for r in raw]
    total = sum(exps)
    return [e / total for e in exps]


def _v_mix(v_hat: float, total_n: int,
           sum_pi_vis: float, sum_piq_vis: float) -> float:
    """Appendix D's mixed value estimator (Eq. 33) — the Eq. 10 fallback
    for unvisited actions:

        v_mix = (v̂_π + ΣN · q̄) / (1 + ΣN),
        q̄ = Σ_{N(a)>0} π(a)·q(a) / Σ_{N(a)>0} π(a)

    weight 1 on the value-net estimate, weight ΣN on the π-weighted mean
    of the visited q's (parent's view). Invariant to the normalization of
    π (the terms are degree-1 ratios). ΣN = 0 — or no visited π mass, an
    f64-underflow corner — degenerates to v̂_π. Figure 6 of the paper:
    this completes Q-values better than the raw value-net estimate.
    Expression order is pinned (L2): q̄ then the blend, f64 throughout.
    """
    if total_n == 0 or sum_pi_vis == 0.0:
        return v_hat
    q_bar = sum_piq_vis / sum_pi_vis
    return (v_hat + total_n * q_bar) / (1 + total_n)


def _terminal_wdl(board: Board) -> Tuple[float, float, float]:
    """Terminal value one-hot from the side TO MOVE's view: checkmate = the
    mover is mated; every other termination is a draw (the C engine's
    Termination already implements claimable-draw semantics, PLAN 1.3)."""
    if board.termination == Termination.CHECKMATE:
        return (0.0, 0.0, 1.0)
    return (0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# Game driver + search
# ---------------------------------------------------------------------------

class SelfPlayGame:
    """One game bound to one "thread" (PARALLEL_DESIGN §1.2). The driver owns
    the RNG; the handoff between moves is threaded explicitly through `play()`
    (no game-level field). The search is a set of flat methods below —
    `_search_move` is PLAN's GUMBEL_SEARCH."""

    def __init__(self, cfg: SelfPlayConfig, seed: int, game_index: int,
                 start_fen: Optional[str] = None):
        self.cfg = cfg
        self.board = Board(start_fen)
        self.start_fen = self.board.fen
        self.seed = seed
        self.game_index = game_index
        self.rng = Xoshiro256PP(game_seed(seed, game_index))
        self.moves: List[MoveRecord] = []
        self.resigned: Optional[int] = None

    # -- public -----------------------------------------------------------------

    def play(self, evaluator: Evaluator,
             max_plies: Optional[int] = None) -> GameRecord:
        """Play to natural termination, resignation, or (with max_plies)
        adjudicated draw; return the record.

        The handoff (F4: the played move's child node, already evaluated) is
        threaded explicitly through this loop as a local — each move's search
        root IS the previous move's played child, so every root after the
        first costs zero evaluations and nothing can go stale."""
        handoff: Optional[Node] = None
        while True:
            if self.board.termination != Termination.PLAYING:
                kind = "natural"
                break
            if max_plies is not None and self.board.ply >= max_plies:
                kind = "adjudicated"
                break
            handoff = self._play_one_move(evaluator, handoff)
            if self.resigned is not None:
                kind = "resign"
                break
        return self._finish(kind)

    # -- one move ----------------------------------------------------------------

    def _play_one_move(self, evaluator: Evaluator,
                       handoff: Optional[Node] = None) -> Optional[Node]:
        """One move: search → resign check → choose → record → push.

        `handoff` is the previous move's played child (or None at game start /
        after a forced move); the return value is this move's played child for
        the next search — or None, ending the chain, if the game ends here by
        resignation or if the played move's child was never created (a forced
        move)."""
        cfg = self.cfg
        board = self.board
        mover = board.side_to_move
        fen_before = board.fen

        decision, root = self._search_move(evaluator, handoff)

        # Resignation: the mover's best post-search value is hopeless.
        if decision.outlook <= cfg.resign_value:
            self.resigned = mover
            self.moves.append(MoveRecord(
                ply=board.ply, mover=mover, fen_before=fen_before,
                played_uci="", played_slot=-1,
                pi_slots=decision.candidate_slots,
                pi_probs=decision.pi_probs, outlook=decision.outlook))
            return None

        # Move choice: sample ∝ π′ (training: every ply by default), else the
        # survivor — the deterministic move is the eval-measurement strategy
        # (R5: eval configs set sample_plies = seed-phase length, e.g. 10).
        if cfg.sample_plies is None or board.ply < cfg.sample_plies:
            played_slot = self._sample_pi(decision, root)
        else:
            played_slot = decision.survivor_slot
        played_uci = root.slot_to_uci[played_slot]

        self.moves.append(MoveRecord(
            ply=board.ply, mover=mover, fen_before=fen_before,
            played_uci=played_uci, played_slot=played_slot,
            pi_slots=decision.candidate_slots, pi_probs=decision.pi_probs,
            outlook=decision.outlook))
        board.push(played_uci)
        # Handoff (F4): the played child (always evaluated, subtree included)
        # is the next search's root — zero evals there. A forced move's child
        # was never created, so the chain ends (None) naturally.
        return root.children.get(played_slot)

    # -- the search (PLAN's GUMBEL_SEARCH) ----------------------------------------

    def _search_move(self, evaluator: Evaluator,
                     handoff: Optional[Node] = None):
        """Returns (MoveDecision, root node).

        `handoff` — the previous move's played child, already evaluated with
        its subtree — is used directly as this search's root (F4)."""
        cfg = self.cfg
        board = self.board
        root_ply = board.ply

        # -- root: the handed-off node (already evaluated) or a fresh eval ----
        if handoff is not None:
            root = handoff
        else:
            root = Node(NEW)
            mover = board.side_to_move
            for move in board.legal_moves:          # FFI generation order
                root.slot_to_uci[uci_to_index(move, mover)] = move
            self._evaluate(root, evaluator)

        legal = list(root.slot_to_uci)              # movegen order
        rank = {s: i for i, s in enumerate(legal)}

        # -- candidates: Gumbel top-m over ALL legal actions -------------------
        z = {a: root.logp[a] + self.rng.gumbel() for a in legal}  # RNG: movegen order
        m = min(cfg.n, cfg.m_cap, len(legal))
        # m == 1 (forced position) is legal: rounds = 0 and the halving loop
        # below never runs — the general path handles it with no special case.
        A = sorted(legal, key=lambda s: (-z[s], rank[s]))[:m]
        A0 = list(A)
        rounds = (m - 1).bit_length()               # ceil(log2 m), from ORIGINAL m

        # -- sequential halving --------------------------------------------------
        while len(A) > 1:
            plays = _ceil_div(cfg.n, rounds * len(A))
            for a in A:                             # candidate-major (PLAN pseudocode)
                for _ in range(plays):
                    self._descend(root, a, root_ply, evaluator)
            A = self._halve(root, A, z, rank, cfg)

        self._rewind(root_ply)      # the board must stand on the root for the
                                    # driver's push (descent left it at a leaf)
        return self._finalize(root, A0, z, A[0], cfg), root

    def _descend(self, root: Node, forced_slot: int, root_ply: int,
                 evaluator: Evaluator) -> None:
        """One descent: forced first step (the SH schedule), Eq. 14 below.
        Ends at the fresh leaf it expands — evaluated synchronously RIGHT HERE,
        then backed up — or at a terminal node (inline backup).
        Invariant: the board stands on `node`; `path` ends with `node`."""
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
            uci = node.slot_to_uci[slot]
            child = node.children.get(slot)
            if child is not None:
                self.board.push(child.uci)          # board: parent -> child
            else:
                child = self._make_child(node, slot, uci)   # pushes
                if child.state == NEW:
                    # Fresh leaf: evaluate synchronously, back up, done.
                    self._evaluate(child, evaluator)
                    self._backup(path + [child], child.wdl)
                    return
                # fresh TERMINAL: board already at it; loop top backs it up
            node = child
            path.append(child)
            slot = None

    def _evaluate(self, node: Node, evaluator: Evaluator) -> None:
        """One synchronous size-1 inference call; the board stands on `node`."""
        logp, wdl = evaluator.evaluate(self.board.encode(),
                                       list(node.slot_to_uci),
                                       self.board.side_to_move)
        node.logp = logp
        node.wdl = wdl
        node.state = EVALUATED

    def _make_child(self, parent: Node, slot: int, uci: str) -> Node:
        """Create the child for `slot` (board stands on the parent). Pushes
        the move; records the child's legal-move table; detects terminals."""
        self.board.push(uci)
        if self.board.termination != Termination.PLAYING:
            child = Node(TERMINAL, uci)
            child.wdl = _terminal_wdl(self.board)
        else:
            child = Node(NEW, uci)
            mover = self.board.side_to_move
            for move in self.board.legal_moves:     # FFI generation order
                child.slot_to_uci[uci_to_index(move, mover)] = move
        parent.children[slot] = child
        return child

    def _select_eq14(self, node: Node) -> int:
        """Full Gumbel Eq. 14 at an interior node — probability space, per
        the paper (Eq. 11 + Eq. 14) and mctx's reference implementation:

            completedQ(a) = q(a)                     if N(a) > 0
                          = v_mix (Appendix D)       otherwise
            π′(a) = softmax_a[ logp(a) + σ(completedQ(a)) ]        (Eq. 11)
            select argmax_a[ π′(a) − N(a)/(1 + Σ_b N(b)) ]         (Eq. 14)

        The softmax is max-shifted; the π′ denominator sum runs in movegen
        enumeration order (pinned f64, same discipline as the root π′).
        Ties break by movegen order (first max wins under strict >).

        Note the argmax is on the π′ PROBABILITIES minus the visit fraction —
        not on the logits. The two forms genuinely diverge once logit gaps
        exceed ~1 nat (the visit fraction is capped at 1): probability space
        rebalances visits toward π′ proportions, logit space freezes on the
        prior. (2026-09-29: fixed from an earlier logit-space form that
        matched neither the paper nor mctx.)"""
        children = node.children
        max_n = 0
        total_n = 0
        sum_pi_vis = 0.0        # Σ_{N(a)>0} π(a)               (v_mix, Eq. 33)
        sum_piq_vis = 0.0       # Σ_{N(a)>0} π(a)·q(a), parent's view
        for slot in node.slot_to_uci:               # movegen order
            child = children.get(slot)
            if child is None or child.visits == 0:
                continue
            if child.visits > max_n:
                max_n = child.visits
            total_n += child.visits
            pi_a = math.exp(node.logp[slot])
            q_a = -child.q_own                      # parent's view
            sum_pi_vis += pi_a
            sum_piq_vis += pi_a * q_a
        base = self.cfg.c_visit + max_n
        fallback = _v_mix(node.v - self.cfg.c_fpu, total_n,
                          sum_pi_vis, sum_piq_vis)

        scores = {}
        s_max = -math.inf
        for slot in node.slot_to_uci:               # movegen order
            child = children.get(slot)
            if child is not None and child.visits > 0:
                q = -child.q_own                    # parent's view
            else:
                q = fallback                        # Eq. 10 via Eq. 33
            s = node.logp[slot] + base * self.cfg.c_scale * q
            scores[slot] = s
            if s > s_max:
                s_max = s
        total_exp = 0.0
        for slot in node.slot_to_uci:               # pinned summation order
            total_exp += math.exp(scores[slot] - s_max)

        best_slot = None
        best_score = -math.inf
        denom = 1 + total_n
        for slot in node.slot_to_uci:               # movegen order; strict >
            child = children.get(slot)
            n_a = child.visits if child is not None else 0
            score = (math.exp(scores[slot] - s_max) / total_exp
                     - n_a / denom)
            if score > best_score:
                best_slot, best_score = slot, score
        assert best_slot is not None
        return best_slot

    def _halve(self, root: Node, A: List[int], z: Dict[int, float],
               rank: Dict[int, int], cfg: SelfPlayConfig) -> List[int]:
        """One halving step: keep top ⌈|A|/2⌉ by z(a) + σ(q(a))."""
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
        """π′ over A₀ (the training target), the survivor, and the resignation
        outlook. completedQ follows Eq. 10 uniformly: q(a) from the child's
        stats when visited, v̂(root) − c_fpu otherwise — which is the whole
        story for a forced position (m = 1), where no candidate was visited."""
        max_n = 0
        qs = {}
        for a in A0:
            child = root.children.get(a)
            if child is not None and child.visits > 0:
                qs[a] = -child.q_own                 # root-mover view
                max_n = max(max_n, child.visits)
            else:
                qs[a] = root.v - cfg.c_fpu           # Eq. 10 fallback —
                                                     # v_mix (Eq. 33)
                                                     # degenerates to this:
                                                     # unvisited candidates
                                                     # at the root imply
                                                     # ΣN = 0 (forced
                                                     # position, m = 1)
        base = cfg.c_visit + max_n
        raw = [root.logp[a] + base * cfg.c_scale * qs[a] for a in A0]
        return MoveDecision(
            candidate_slots=list(A0), z=dict(z), q_root=qs,
            pi_probs=_softmax(raw), survivor_slot=survivor,
            outlook=max(qs.values()))

    @staticmethod
    def _backup(path: Sequence[Node], wdl_leaf: Tuple[float, float, float]) -> None:
        """Back a leaf value up along its path, sign-flipping every ply.
        Stats are each node's OWN mover's view; the leaf itself is visited
        (so its parent's completedQ is well defined). `path` includes the
        leaf; the board is untouched."""
        v = wdl_leaf[0] - wdl_leaf[2]
        sign = 1.0
        for node in reversed(path):
            node.visits += 1
            node.value_sum += sign * v
            sign = -sign

    # -- bookkeeping ----------------------------------------------------------------

    def _rewind(self, root_ply: int) -> None:
        while self.board.ply > root_ply:
            self.board.pop()

    def _sample_pi(self, decision: MoveDecision, root: Node) -> int:
        """One uniform() draw, inverse CDF over the candidate set in movegen
        enumeration order (R6)."""
        rank = {s: i for i, s in enumerate(root.slot_to_uci)}
        u = self.rng.uniform()
        pairs = sorted(zip(decision.candidate_slots, decision.pi_probs),
                       key=lambda sp: rank[sp[0]])
        cum = 0.0
        for slot, p in pairs:
            cum += p
            if u < cum:
                return slot
        return pairs[-1][0]                         # fp guard: last candidate

    def _finish(self, kind: str) -> GameRecord:
        board = self.board
        if self.resigned is not None:
            z_white = -1.0 if self.resigned == 0 else 1.0
        elif board.termination == Termination.CHECKMATE:
            z_white = 1.0 if board.winner == 0 else -1.0
        else:
            z_white = 0.0                           # draws and adjudication
        return GameRecord(
            start_fen=self.start_fen, moves=self.moves, z_white=z_white,
            outcome_kind=kind, resigner=self.resigned,
            seed=self.seed, game_index=self.game_index, n_sims=self.cfg.n)
