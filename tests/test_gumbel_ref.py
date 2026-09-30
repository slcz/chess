"""tests/test_gumbel_ref.py — Phase 3.2a: the sequential Python Gumbel reference.

The reference is deliberately simple: synchronous size-1 evals, no request
queues (user decision 2026-09-27). These tests pin the parts the C engine
must reproduce (PARALLEL_DESIGN R6/L2 contract) plus the algorithm's
properties:

- RNG spec: splitmix64 known vector, uniform/gumbel sanity, stream determinism.
- Search math: backup sign flipping, π′ properties, halving survivor,
  candidate visit coverage (m ≤ n ⇒ every candidate visited).
- Protocol: forced-move (k==1) shortcut consumes zero evals and no RNG;
  handoff makes every root after move 1 free; same seed ⇒ identical games.
- Resignation: immediate trigger at the threshold, sign correctness.
- Mate finding: a terminal-detected mate-in-1 dominates the halving.
"""
import math

import pytest

from chess_ai.net.features import uci_to_index
from chess_ai.selfplay.gumbel import (
    EVALUATED,
    GameRecord,
    Node,
    SelfPlayConfig,
    SelfPlayGame,
    TableEvaluator,
    _v_mix,
)
from chess_ai.selfplay.rng import SplitMix64, Xoshiro256PP, game_seed

FORCED_FEN = "k7/8/1K6/8/8/8/R7/8 b - - 0 1"   # black: exactly one legal move (a8b8)
MATE_FEN = "k7/8/1K6/8/8/8/8/2R5 w - - 0 1"    # white: Rc1-c8 is mate
MATE_SLOT = uci_to_index("c1c8", 0)            # 186

# White always wins from the mover's view: the side to move is winning iff
# it is white. Under this net black's outlook is −1 (it resigns), white's +1.
WHITE_WINS = lambda mover: (1.0, 0.0, 0.0) if mover == 0 else (0.0, 0.0, 1.0)


def records_equal(a: GameRecord, b: GameRecord, tol=1e-12) -> None:
    """Strict record comparison for the determinism gate."""
    assert a.start_fen == b.start_fen
    assert a.z_white == b.z_white
    assert a.outcome_kind == b.outcome_kind
    assert a.resigner == b.resigner
    assert len(a.moves) == len(b.moves)
    for ma, mb in zip(a.moves, b.moves):
        assert ma.ply == mb.ply and ma.mover == mb.mover
        assert ma.fen_before == mb.fen_before
        assert ma.played_uci == mb.played_uci
        assert ma.played_slot == mb.played_slot
        assert ma.pi_slots == mb.pi_slots
        for pa, pb in zip(ma.pi_probs, mb.pi_probs):
            assert abs(pa - pb) <= tol
        assert abs(ma.outlook - mb.outlook) <= tol


# ---------------------------------------------------------------------------
# RNG spec
# ---------------------------------------------------------------------------

def test_splitmix64_known_vector():
    sm = SplitMix64(0)
    assert sm.next() == 0xE220A8397B1DCDAF


def test_uniform_and_gumbel_sanity():
    x = Xoshiro256PP(7)
    vals = [x.uniform() for _ in range(100_000)]
    assert all(0.0 < v < 1.0 for v in vals)
    assert abs(sum(vals) / len(vals) - 0.5) < 0.01
    gs = [x.gumbel() for _ in range(100_000)]
    assert abs(sum(gs) / len(gs) - 0.5772) < 0.02   # Euler–Mascheroni


def test_game_seed_stable_and_distinct():
    assert game_seed(123, 0) == game_seed(123, 0)
    assert len({game_seed(123, i) for i in range(1000)}) == 1000


# ---------------------------------------------------------------------------
# Search math
# ---------------------------------------------------------------------------

def test_backup_sign_flipping():
    root, mid, leaf = Node(EVALUATED), Node(EVALUATED), Node(EVALUATED)
    SelfPlayGame._backup([root, mid, leaf], (0.0, 0.0, 1.0))  # leaf mover loses
    assert leaf.visits == 1 and leaf.value_sum == -1.0        # own view
    assert mid.visits == 1 and mid.value_sum == 1.0           # opponent loses
    assert root.visits == 1 and root.value_sum == -1.0
    # A parent reads a child's q negated: q(root view of mid's move) = −(+1) = −1.
    assert -mid.q_own == -1.0


def test_pi_prime_and_visit_properties():
    cfg = SelfPlayConfig(n=8, sample_plies=0)
    game = SelfPlayGame(cfg, seed=1, game_index=0)
    decision, root = game._search_move(TableEvaluator())
    k = len(game.board.legal_moves)
    assert len(decision.candidate_slots) == min(cfg.n, cfg.m_cap, k)
    assert abs(sum(decision.pi_probs) - 1.0) < 1e-9
    assert all(p >= 0.0 for p in decision.pi_probs)
    assert decision.survivor_slot in decision.candidate_slots
    for a in decision.candidate_slots:
        child = root.children[a]
        assert child.visits >= 1                 # m ≤ n ⇒ round-1 coverage
        assert child.state == EVALUATED
        assert -1.0 <= decision.q_root[a] <= 1.0
    # π′ support is exactly the candidate set (sparse over legal moves).
    assert set(decision.q_root) == set(decision.candidate_slots)


def test_halving_survivor_is_argmax_z_when_q_equal():
    # Uniform evaluator ⇒ every candidate has the same q ⇒ the halving
    # ranking reduces to z(a) = logp + gumbel ⇒ the survivor is argmax z.
    for seed in range(5):
        cfg = SelfPlayConfig(n=8, sample_plies=0)
        game = SelfPlayGame(cfg, seed=seed, game_index=0)
        decision, root = game._search_move(TableEvaluator())
        best = max(decision.candidate_slots, key=lambda a: decision.z[a])
        assert decision.z[decision.survivor_slot] >= decision.z[best] - 1e-12


# ---------------------------------------------------------------------------
# Protocol: forced moves, handoff, determinism
# ---------------------------------------------------------------------------

class CountingEvaluator(TableEvaluator):
    """Counts evaluator calls; optionally records evaluated plane bytes."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0
        self.seen: list[bytes] = []

    def evaluate(self, planes, legal_slots, mover):
        self.calls += 1
        self.seen.append(planes.tobytes())
        return super().evaluate(planes, legal_slots, mover)


def test_forced_move_flows_through_search():
    """No k==1 special case: a forced position runs the general path with
    m = 1 — one root eval, no descents, π′ exactly one-hot, outlook = v̂
    (uniform net ⇒ 0) via the Eq. 10 fallback, no handoff (the forced move's
    child is never created)."""
    cfg = SelfPlayConfig(n=8, sample_plies=30)
    game = SelfPlayGame(cfg, seed=3, game_index=0, start_fen=FORCED_FEN)
    ev = CountingEvaluator()
    handoff = game._play_one_move(ev)
    first = game.moves[0]
    assert first.played_uci == "a8b8"
    assert first.pi_probs == [1.0] and first.pi_slots == [first.played_slot]
    assert ev.calls == 1                         # the root eval; no descents at m=1
    assert first.outlook == pytest.approx(0.0)
    assert handoff is None
    rec = game.play(ev, max_plies=3)             # the game progresses normally
    assert len(rec.moves) == 3


def test_mate_in_1_found_and_played():
    cfg = SelfPlayConfig(n=32, sample_plies=0)
    game = SelfPlayGame(cfg, seed=5, game_index=0, start_fen=MATE_FEN)
    ev = CountingEvaluator(prior={MATE_SLOT: 10.0})
    rec = game.play(ev)
    assert rec.moves[0].played_uci == "c1c8"
    assert rec.z_white == 1.0 and rec.outcome_kind == "natural"
    assert ev.calls <= 1 + 2 * cfg.n             # the documented overshoot bound


def test_handoff_makes_every_root_after_move_1_free():
    """The handoff property, verified directly at the evaluator boundary:
    move 1 evaluates its root; from move 2 on, the root position — the played
    child of the previous search — is never sent to the evaluator again."""
    cfg = SelfPlayConfig(n=8, sample_plies=0)
    game = SelfPlayGame(cfg, seed=6, game_index=0)
    ev = CountingEvaluator()

    root_planes_1 = game.board.encode().tobytes()
    handoff = game._play_one_move(ev)            # move 1: fresh root
    assert handoff is not None
    assert root_planes_1 in ev.seen              # move 1 evaluated its root

    root_planes_2 = game.board.encode().tobytes()  # move 2's root (after the push)
    assert root_planes_2 != root_planes_1
    ev.seen.clear()
    game._play_one_move(ev, handoff)             # move 2: handoff root
    assert root_planes_2 not in ev.seen          # the root was never re-evaluated


def test_sequential_determinism_same_seed():
    """The L2 precondition: the reference is a pure function of its seed, so
    the C engine's sequential mode can be diffed against it exactly."""
    cfg = SelfPlayConfig(n=16, sample_plies=30)
    recs = [SelfPlayGame(cfg, seed=11, game_index=4)
            .play(TableEvaluator(), max_plies=10) for _ in range(2)]
    records_equal(recs[0], recs[1])


def test_different_seeds_diverge():
    cfg = SelfPlayConfig(n=16, sample_plies=30)
    recs = [SelfPlayGame(cfg, seed=s, game_index=0)
            .play(TableEvaluator(), max_plies=10) for s in (1, 2)]
    moves_a = [m.played_uci for m in recs[0].moves]
    moves_b = [m.played_uci for m in recs[1].moves]
    assert moves_a != moves_b


# ---------------------------------------------------------------------------
# Resignation (R4)
# ---------------------------------------------------------------------------

def test_resign_immediate_and_labels():
    cfg = SelfPlayConfig(n=8, sample_plies=0, resign_value=-0.90)
    game = SelfPlayGame(cfg, seed=9, game_index=0)
    rec = game.play(TableEvaluator(value_fn=WHITE_WINS))
    # Black's outlook is −1 on its first turn → immediate resignation.
    assert rec.outcome_kind == "resign"
    assert rec.resigner == 1
    assert rec.z_white == 1.0                      # resigner loses
    assert len(rec.moves) == 2                     # white's move + the resign record
    last = rec.moves[-1]
    assert last.played_uci == "" and last.mover == 1
    assert last.outlook == pytest.approx(-1.0)


def test_winning_side_never_resigns():
    cfg = SelfPlayConfig(n=8, sample_plies=0, resign_value=-0.90)
    game = SelfPlayGame(cfg, seed=9, game_index=0)
    # White's outlook is +1 at its turn — above the threshold, never resigns.
    rec = game.play(TableEvaluator(value_fn=WHITE_WINS), max_plies=1)
    white_moves = [m for m in rec.moves if m.mover == 0]
    assert white_moves and all(m.outlook == pytest.approx(1.0)
                               for m in white_moves)


# ---------------------------------------------------------------------------
# Eq. 14 fidelity (probability space + v_mix fallback) — paper/mctx parity
# ---------------------------------------------------------------------------

def _hand_node(logp, children_stats, wdl=(1 / 3, 1 / 3, 1 / 3)):
    """Hand-built EVALUATED node for Eq. 14 unit tests: logp {slot: value}
    (dict order = enumeration order), children_stats {slot: (visits,
    value_sum)}; a missing slot means no child exists yet. q(a) from the
    parent's view is −value_sum/visits."""
    node = Node(EVALUATED)
    node.slot_to_uci = {s: f"a1a{s + 2}" for s in logp}
    node.logp = dict(logp)
    node.wdl = wdl
    for s, (visits, vsum) in children_stats.items():
        c = Node(EVALUATED)
        c.wdl = (1 / 3, 1 / 3, 1 / 3)
        c.visits = visits
        c.value_sum = vsum
        node.children[s] = c
    return node


def test_eq14_selects_in_probability_space():
    """Pin the paper's Eq. 14: the argmax runs on π′ PROBABILITIES minus
    the visit fraction (mctx-faithful), not on the logits. With σ(q) flat
    (q = 0 everywhere), s = logp: a 1.5-nat logp gap under a 9:1 visit
    imbalance flips to the follower — π′(1) − 1/11 = 0.0916 >
    π′(0) − 9/11 = −0.0006 — while the old logit-space form kept the
    leader (1.5 − 9/11 = 0.68 > −1/11)."""
    cfg = SelfPlayConfig(n=8)
    game = SelfPlayGame(cfg, seed=0, game_index=0)
    node = _hand_node({0: 1.5, 1: 0.0}, {0: (9, 0.0), 1: (1, 0.0)})
    assert game._select_eq14(node) == 1


def test_eq14_v_mix_fallback():
    """The Eq. 10 fallback is Appendix D's v_mix (Eq. 33), not the raw
    value-net estimate: (v̂ + ΣN·q̄)/(1 + ΣN), invariant to the π
    normalization, degenerating to v̂ when nothing is visited."""
    assert _v_mix(0.9, 3, 1.0, 0.5) == pytest.approx((0.9 + 3 * 0.5) / 4)
    assert _v_mix(0.9, 0, 0.0, 0.0) == 0.9              # ΣN = 0 → v̂
    assert _v_mix(0.9, 3, 2.0, 1.0) == pytest.approx(0.6)   # π-scale invariant
