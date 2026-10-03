"""
tests/test_az_c.py — cross-boundary verification of the C AZ-PUCT port
(engine/src/az.c via chess_ai.selfplay.azsearch) against the Python
reference (scripts/az_eval.py, the clean-room original — untouched).

Both engines run in ONE process under the SAME deterministic fixture
evaluator, so the neural-net indirection that normally makes engines
differ (fp32 batch-shape rounding, the L3d effect) is gone: what remains
is the walk itself, which is required to match:

  1. root statistics per position, per sim count: identical move order
     (movegen enumeration), identical visit vectors, priors/q equal to
     the last bit the doubles allow, same root value;
  2. the Q-sign contract (chess_az.h): verbatim sign avoids a mate-in-1
     under a value-free uniform net, paper sign piles onto it (the same
     pin as the C unit test, through the binding);
  3. full-game parity: same per-game rng stream ⇒ move-for-move
     identical games (search → choose_move → push loops side by side);
  4. choose_move_visits ≡ az_eval.choose_move on synthetic visit vectors
     (the rng-consumption contract);
  5. many handles concurrently on Python threads (the eval-fleet shape).

The mate-in-1 sign test is the empirical pin of the branch finding: the
Python reference's selection maximizes the child's own-mover q — the
OPPONENT's view — so under a value-free net the mating move (terminal q
−1) sinks to the bottom of the visit race exactly once it has been
touched; with the AlphaGo paper's Q(s,a) (parent view) it dominates.
"""
from __future__ import annotations

import random
import threading

from chess_ai.engine import Board, Termination
from chess_ai.selfplay.azsearch import (
    CAzGame,
    choose_move_visits,
    play_az_game,
)
from scripts.az_eval import Node, choose_move, search as py_search

# Same board as the C unit test: white Kb6 + Rc1, black Ka8 — c1c8 is the
# unique mate in one.
MATE_FEN = "k7/8/1K6/8/8/8/8/2R5 w - - 0 1"

# A spread of positions: the start, an open Italian middlegame, a
# promotion race (underpromotion slots exercised), and the mate-in-1.
PARITY_FENS = [
    None,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/P6k/8/8/8/8/7K/8 w - - 0 1",
    MATE_FEN,
]


class FixtureNet:
    """Deterministic hash evaluator — same (logp, wdl) for a given
    position in both engines, so any walk divergence shows up directly.
    The empty-slot call is the Python engine's wasteful terminal-leaf
    evaluation (its value is discarded; the C port skips the call)."""

    def evaluate(self, planes, legal_slots, mover):
        if not legal_slots:
            return {}, (0.3, 0.4, 0.3)
        logp = {s: -3.0 * ((s * 2654435761 + mover * 91) % 997) / 996.0
                for s in legal_slots}
        # planes arrives [64,117] from the Python engine's Board.encode()
        # and flat 7488 from the C request — same bytes either way.
        flat = planes.reshape(-1)
        frac = float(flat[0]) * 0.13 + float(flat[116]) * 0.71
        frac -= int(frac)
        return logp, (0.15 + 0.5 * frac, 0.2, 0.65 - 0.45 * frac)


class UniformNet:
    """Value-free uniform net (the sign test's isolation chamber)."""

    def evaluate(self, planes, legal_slots, mover):
        return {s: 0.0 for s in legal_slots}, (1 / 3, 1 / 3, 1 / 3)


def _py_children(root) -> list[tuple[str, int, float, float]]:
    """az_eval.py's root stats in insertion (= movegen) order."""
    return [(u, c.visits, c.prior, c.q)
            for u, c in root.children.items()]


def test_root_stats_parity():
    fix = FixtureNet()
    for fen in PARITY_FENS:
        for sims in (24, 96):
            board = Board(fen)
            py_root, py_v = py_search(fix, board, sims, 1.25,
                                      random.Random(0))
            py_ch = _py_children(py_root)

            game = CAzGame(fen)
            try:
                game.think(fix, sims, 1.25)
                c_ch = game.root_children()
                c_v = game.root_value()
            finally:
                game.close()

            assert [c[0] for c in c_ch] == [c[0] for c in py_ch]
            assert [c[1] for c in c_ch] == [c[1] for c in py_ch]
            for (u, n_c, p_c, q_c), (_, n_p, p_p, q_p) in zip(c_ch, py_ch):
                assert p_c == p_p or abs(p_c - p_p) < 1e-15, (u, p_c, p_p)
                assert q_c == q_p or abs(q_c - q_p) < 1e-15, (u, q_c, q_p)
            assert c_v == py_v or abs(c_v - py_v) < 1e-15


def test_mate_in_1_q_sign():
    """The sign pin through the binding (mirrors engine/tests/test_az.c):
    unvisited children carry no q under either sign, so the first ~20
    sims rotate through all children; from the mate's first touch on,
    verbatim keeps it at the bottom, paper piles everything onto it."""
    for q_parent, expect_dominates in ((False, False), (True, True)):
        game = CAzGame(MATE_FEN)
        try:
            game.think(UniformNet(), sims=60, c_puct=1.25,
                       q_parent_view=q_parent)
            ch = game.root_children()
        finally:
            game.close()
        mate = next(c for c in ch if c[0] == "c1c8")
        best = max(c[1] for c in ch)
        if expect_dominates:
            assert mate[1] == best and mate[1] >= 40
        else:
            assert mate[1] == 1 and best > 1
        # the mated child's q is its own (black's) view: -1
        assert mate[3] == -1.0


def test_full_game_parity():
    """Same rng stream ⇒ move-for-move identical games, both seats the
    same fixture net (the az_eval.py match loop, side by side)."""
    fix = FixtureNet()
    for seed in (3, 17):
        rng = random.Random((seed << 1) | 1)          # a_white = True
        board = Board()
        py_moves = []
        ply = 0
        while board.termination == Termination.PLAYING and ply < 80:
            root, _ = py_search(fix, board, 20, 1.25, rng)
            u, _n = choose_move(root, rng, ply < 8)
            board.push(u)
            py_moves.append(u)
            ply += 1

        res = play_az_game(fix, fix, sims=20, c_puct=1.25,
                           rng=random.Random((seed << 1) | 1),
                           sample_plies=8, max_plies=80)
        assert res["moves"] == py_moves
        assert res["plies"] == len(py_moves)
        assert res["winner"] == board.winner
        assert res["kind"] == ("natural"
                               if board.termination != Termination.PLAYING
                               else "adjudicated")


def test_choose_move_visits_matches_reference():
    """choose_move_visits ≡ az_eval.choose_move: same stable sort, same
    inverse-CDF, same rng draw pattern."""
    for trial in range(200):
        rng = random.Random(trial)
        k = rng.randint(1, 8)
        rows = [(f"m{i}", rng.randint(0, 30), 1.0 / k, 0.0)
                for i in range(k)]
        root = Node()
        for u, visits, prior, _q in rows:
            child = Node(parent=root, move=u, prior=prior)
            child.visits = visits
            child.value_sum = 0.0
            root.children[u] = child
        for sample in (False, True):
            rng_a, rng_b = random.Random(99), random.Random(99)
            u_py, _ = choose_move(root, rng_a, sample)
            u_c = choose_move_visits(rows, rng_b, sample)
            assert u_c == u_py


def test_concurrent_handles():
    """Many games on Python threads sharing one evaluator — the eval
    fleet's shape (each thread's search is single-threaded C; cffi
    releases the GIL around the calls)."""
    class CountingNet(FixtureNet):
        def __init__(self):
            self.lock = threading.Lock()
            self.calls = 0

        def evaluate(self, planes, legal_slots, mover):
            with self.lock:
                self.calls += 1
            return super().evaluate(planes, legal_slots, mover)

    net = CountingNet()
    results = [None] * 6

    def run(i):
        results[i] = play_az_game(net, net, sims=16, c_puct=1.25,
                                  rng=random.Random(1000 + i),
                                  sample_plies=4, max_plies=40)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r and r["plies"] > 0 for r in results)
    assert net.calls > 0
    # same seeds replay identical games even interleaved (walk
    # determinism does not depend on evaluation order between requests).
    assert results[0]["moves"] == play_az_game(
        net, net, sims=16, c_puct=1.25, rng=random.Random(1000),
        sample_plies=4, max_plies=40)["moves"]
