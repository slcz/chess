"""tests/test_tinygame.py — Phase 3.5: the tiny-game smoke loop.

Three layers, mirroring the house style of test_gumbel_ref.py /
test_selfplay_data.py:

1. ORACLE + GAME: the exhaustive minimax is right (canonical counts,
   known tic-tac-toe theory spots), and the Board-shaped game interface
   round-trips (push/pop, mover-relative encode).
2. SEARCH PORT: the gumbel.py contract transplanted — RNG determinism,
   π′ properties, round-1 candidate coverage, forced-move generality,
   mate-in-1 domination, handoff, resignation labels, and the
   Gumbels-off eval variant. Plus THE SEARCH-CORRECTNESS ANCHOR: an
   oracle evaluator (exact WDL, optimal-move priors) through the same
   search + gate metric must be 100% value-preserving — proving the
   search and the gate are correct independent of any learned net.
3. DATA/TRAINER/LOOP: dense-π′ padding, multiset buffer sampling, aux
   and WDL sign conventions, numpy≈torch evaluator parity, the one-hot
   π′ == sparse CE identity (the chess-pinned equivalence), and the
   end-to-end learn-to-perfect-play run (the actual 3.5 gate, reduced to
   a CI-sized budget; the full-budget run is scripts via the CLI — see
   chess_ai/tinygame/README.md).
"""
import math
import random

import numpy as np
import pytest
import torch

from chess_ai.tinygame import game as ttt
from chess_ai.tinygame.data import (
    PI_SUPPORT_MAX,
    TinyReplayBuffer,
    _dense_pi,
    _wdl_onehot,
    samples_at,
    selfplay_samples,
)
from chess_ai.tinygame.game import DRAW, EMPTY, O, PLAYING, WIN, X, TicTacToe
from chess_ai.tinygame.loop import (
    SmokeConfig,
    all_positions,
    critical_positions,
    eval_positions,
    random_start,
    run_smoke,
)
from chess_ai.tinygame.net import NumpyEvaluator, TinyNet, masked_log_softmax
from chess_ai.tinygame.search import (
    EVALUATED,
    SelfPlayConfig,
    SelfPlayGame,
    TableEvaluator,
)


def cells_of(*moves):
    """Build a position by pushing moves — the low-typo way."""
    b = TicTacToe()
    for m in moves:
        b.push(m)
    return b.cells


# X to move with EXACTLY ONE IMMEDIATE WIN: 2 completes the top row —
# the ttt analogue of the chess suite's mate-in-1 FEN.
MATE_CELLS = cells_of(0, 3, 1, 4)
# X to move with a UNIQUE optimal move (8) — but the win is deep, not a
# mate-in-1 (the oracle was searched for a unique-optimal position;
# hand-crafting these is error-prone: standing double threats usually make
# several moves win).
UNIQUE_WIN_CELLS = cells_of(5, 6)
# O to move, already dead: X triple-threatens 2, 7, 8 — every reply loses,
# so no choice matters (a LOST position is never critical).
LOST_CELLS = cells_of(0, 3, 1, 2, 4)
# O to move after X's center opening: corners draw, edges lose.
CENTER_REPLY = cells_of(4)


# ---------------------------------------------------------------------------
# 1. Oracle + game
# ---------------------------------------------------------------------------

def test_reachable_counts_canonical():
    assert len(ttt.reachable_positions()) == 5478        # the classic count
    assert len(all_positions()) == 4520                 # minus terminals


def test_oracle_known_theory():
    empty = (EMPTY,) * 9
    assert ttt.value(empty) == 0
    assert ttt.optimal_moves(empty) == tuple(range(9))  # every opening draws
    assert not ttt.is_critical(empty)

    assert TicTacToe(MATE_CELLS).side_to_move == 0
    assert ttt.value(MATE_CELLS) == 1
    assert ttt.optimal_moves(MATE_CELLS) == (2,)
    assert ttt.is_critical(MATE_CELLS)

    assert ttt.value(UNIQUE_WIN_CELLS) == 1
    assert ttt.optimal_moves(UNIQUE_WIN_CELLS) == (8,)
    assert TicTacToe(UNIQUE_WIN_CELLS).side_to_move == 0

    # classic theory, second anchor: vs a corner opening ONLY the center draws
    assert ttt.optimal_moves(cells_of(8)) == (4,)

    assert TicTacToe(LOST_CELLS).side_to_move == 1
    assert ttt.value(LOST_CELLS) == -1
    assert len(ttt.optimal_moves(LOST_CELLS)) == len(TicTacToe(LOST_CELLS).legal_moves)
    assert not ttt.is_critical(LOST_CELLS)

    assert ttt.value(CENTER_REPLY) == 0
    assert set(ttt.optimal_moves(CENTER_REPLY)) == {0, 2, 6, 8}   # corners only
    assert ttt.is_critical(CENTER_REPLY)


def test_oracle_exhaustive_consistency():
    for cells in all_positions():
        v = ttt.value(cells)
        assert v in (-1, 0, 1)
        opt = ttt.optimal_moves(cells)
        assert 0 < len(opt) <= len(TicTacToe(cells).legal_moves)
        for m in opt:                       # optimal ⇒ value-preserving
            board = TicTacToe(cells)
            board.push(m)
            child = (1 if board.termination == WIN else
                     0 if board.termination == DRAW else -ttt.value(board.cells))
            assert child == v


def test_game_push_pop_encode_roundtrip():
    b = TicTacToe()
    b.push(4)                                # X
    b.push(0)                                # O
    b.push(2)                                # X
    assert b.side_to_move == 1 and b.ply == 3
    enc = b.encode()                         # mover-relative (O's view)
    assert enc.shape == (18,) and enc.dtype == np.float32
    assert enc[0] == 1.0                             # my (O's) piece
    assert enc[9 + 4] == 1.0 and enc[9 + 2] == 1.0   # opponent's (X)
    assert enc.sum() == 3.0
    b.pop()
    assert b.cells == cells_of(4, 0) and b.side_to_move == 0
    enc2 = b.encode()                        # mover-relative (X's view)
    assert enc2[4] == 1.0                    # X's piece (mine, X's view)
    assert enc2[9 + 0] == 1.0                # O's piece (opponent's)


def test_game_termination_detection():
    b = TicTacToe()
    for m in (0, 3, 1, 5, 2):              # X takes the top row
        b.push(m)
    assert b.termination == WIN and b.winner == X
    full = cells_of(0, 1, 2, 3, 4, 6, 5, 8, 7)      # full board, no line
    assert TicTacToe(full).termination == DRAW
    assert TicTacToe(full).winner is None


# ---------------------------------------------------------------------------
# 2. Search port (the gumbel.py contract)
# ---------------------------------------------------------------------------

class CountingEvaluator(TableEvaluator):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0
        self.seen = []

    def evaluate(self, planes, legal_slots, mover):
        self.calls += 1
        self.seen.append(tuple(planes))
        return super().evaluate(planes, legal_slots, mover)


def records_equal(a, b, tol=1e-12):
    assert a.z_white == b.z_white and a.outcome_kind == b.outcome_kind
    assert a.resigner == b.resigner and len(a.moves) == len(b.moves)
    for ma, mb in zip(a.moves, b.moves):
        assert ma.ply == mb.ply and ma.mover == mb.mover
        assert ma.cells_before == mb.cells_before
        assert ma.played_cell == mb.played_cell
        assert ma.pi_slots == mb.pi_slots
        for pa, pb in zip(ma.pi_probs, mb.pi_probs):
            assert abs(pa - pb) <= tol


def test_same_seed_identical_game():
    cfg = SelfPlayConfig(n=8)
    r1 = SelfPlayGame(cfg, seed=1, game_index=0).play(TableEvaluator())
    r2 = SelfPlayGame(cfg, seed=1, game_index=0).play(TableEvaluator())
    records_equal(r1, r2)
    r3 = SelfPlayGame(cfg, seed=2, game_index=0).play(TableEvaluator())
    assert [m.played_cell for m in r1.moves] != [m.played_cell for m in r3.moves]


def test_pi_prime_and_round1_coverage():
    cfg = SelfPlayConfig(n=8, sample_plies=0)
    game = SelfPlayGame(cfg, seed=1, game_index=0)
    decision, root = game._search_move(TableEvaluator())
    k = len(game.board.legal_moves)
    assert len(decision.candidate_slots) == min(cfg.n, cfg.m_cap, k)
    assert abs(sum(decision.pi_probs) - 1.0) < 1e-9
    assert all(p >= 0.0 for p in decision.pi_probs)
    for a in decision.candidate_slots:
        assert root.children[a].visits >= 1            # m ≤ n ⇒ coverage
        assert root.children[a].state == EVALUATED
    # halving survivor is argmax z when q is uniform
    best = max(decision.candidate_slots, key=lambda a: decision.z[a])
    assert decision.z[decision.survivor_slot] >= decision.z[best] - 1e-12


def test_forced_move_general_path():
    """8 pieces, one empty cell: m = 1, no descents, π′ exactly one-hot,
    outlook from the Eq. 10 fallback (uniform net ⇒ 0), no handoff."""
    cells = cells_of(0, 1, 4, 2, 3, 5, 7, 6)           # X:0,2,3,7 O:1,4,5,6, empty 8
    game = SelfPlayGame(SelfPlayConfig(n=8, sample_plies=30), seed=3,
                        game_index=0, start_cells=cells)
    ev = CountingEvaluator()
    handoff = game._play_one_move(ev)
    first = game.moves[0]
    assert first.played_cell == 8
    assert first.pi_probs == [1.0]
    assert ev.calls == 1
    assert first.outlook == pytest.approx(0.0)
    assert handoff is None


def test_mate_in_1_found_and_played():
    """The ttt analogue of the chess mate-in-1 test: the terminal-detected
    win dominates the halving and is played, under a value-blind net."""
    cfg = SelfPlayConfig(n=32, sample_plies=0)
    game = SelfPlayGame(cfg, seed=5, game_index=0, start_cells=MATE_CELLS)
    ev = CountingEvaluator(prior={2: 10.0})
    rec = game.play(ev)
    assert rec.moves[0].played_cell == 2
    assert rec.z_white == 1.0 and rec.outcome_kind == "natural"
    assert ev.calls <= 1 + 2 * cfg.n                   # documented overshoot bound


def test_handoff_makes_every_root_after_move_1_free():
    cfg = SelfPlayConfig(n=8, sample_plies=0)
    game = SelfPlayGame(cfg, seed=6, game_index=0)
    ev = CountingEvaluator()
    root_planes_1 = tuple(game.board.encode())
    handoff = game._play_one_move(ev)
    assert handoff is not None and root_planes_1 in ev.seen
    root_planes_2 = tuple(game.board.encode())
    assert root_planes_2 != root_planes_1
    ev.seen.clear()
    game._play_one_move(ev, handoff)
    assert root_planes_2 not in ev.seen


def test_resign_immediate_and_labels():
    # X's value is always a win, O's always a loss: O resigns on sight.
    first_wins = lambda mover: (1.0, 0.0, 0.0) if mover == 0 else (0.0, 0.0, 1.0)
    cfg = SelfPlayConfig(n=8, sample_plies=0)
    rec = SelfPlayGame(cfg, seed=9, game_index=0).play(
        TableEvaluator(value_fn=first_wins))
    assert rec.outcome_kind == "resign" and rec.resigner == 1
    assert rec.z_white == 1.0 and len(rec.moves) == 2
    last = rec.moves[-1]
    assert last.played_cell == -1 and last.mover == 1
    assert last.outlook == pytest.approx(-1.0)


def test_gumbels_off_eval_variant():
    """z = logp (no draw): the survivor is exactly argmax of π′'s score,
    and the search consumes no RNG — the plan's deterministic eval."""
    game = SelfPlayGame(SelfPlayConfig(n=64, m_cap=16, sample_plies=0,
                                       gumbels_off=True), seed=42, game_index=0)
    ev = CountingEvaluator()
    d1, _ = game._search_move(ev)
    top = max(range(len(d1.pi_probs)), key=lambda i: d1.pi_probs[i])
    assert d1.survivor_slot == d1.candidate_slots[top]
    marker = game.rng.next_u64()               # stream position after search 1
    d2, _ = game._search_move(ev)
    assert game.rng.next_u64() != marker       # search 2 also drew nothing...
    assert d2.pi_probs == d1.pi_probs          # ...so the decision repeats


class OracleEvaluator(TableEvaluator):
    """Exact values + optimal-move priors, decoded from the mover-relative
    planes — the search-correctness anchor: if the net were perfect, the
    deployed player must be perfect."""

    def _cells(self, planes, mover):
        me = X if mover == 0 else O
        opp = O if mover == 0 else X
        cells = []
        for i in range(9):
            if planes[i] > 0:
                cells.append(me)
            elif planes[9 + i] > 0:
                cells.append(opp)
            else:
                cells.append(EMPTY)
        return tuple(cells)

    def evaluate(self, planes, legal_slots, mover):
        cells = self._cells(planes, mover)
        wdl = ttt.wdl_of_value(ttt.value(cells))
        prior = {m: 5.0 for m in ttt.optimal_moves(cells)}
        ev = TableEvaluator(prior=prior)
        return ev.evaluate(planes, legal_slots, mover)

    def _forward(self, x):
        mover = 0 if x[:9].sum() >= x[9:].sum() else 1   # parity from planes
        cells = self._cells(x, mover)
        pol = np.array([5.0 if m in ttt.optimal_moves(cells) else 0.0
                        for m in range(9)], dtype=np.float32)
        wdl = np.array([10.0 if ttt.value(cells) == s else 0.0
                        for s in (1, 0, -1)], dtype=np.float32)
        return pol, wdl, pol


def test_oracle_evaluator_search_is_exactly_perfect():
    """Search + gate machinery correctness, net-free: over every reachable
    position, search driven by exact values plays value-preservingly."""
    ev = OracleEvaluator()
    stats = eval_positions(ev, all_positions(), eval_n=64, resign_value=-0.90)
    assert stats["perfect_pct"] == 100.0
    assert stats["resign_blunders"] == 0.0
    assert stats["critical_perfect_pct"] == 100.0


# ---------------------------------------------------------------------------
# 3. Data / trainer / loop
# ---------------------------------------------------------------------------

def test_dense_pi_padding_and_renorm():
    slots, probs = [3, 1, 7], [0.5, 0.5, 0.0]
    pi_slots, pi_probs = _dense_pi(slots, probs)
    assert pi_slots.shape == (PI_SUPPORT_MAX,) and pi_probs.shape == (PI_SUPPORT_MAX,)
    assert list(pi_slots[:3]) == slots
    assert (pi_slots[3:] == 3).all()                  # pad repeats slot 0
    assert pi_probs[3:].sum() == 0.0
    assert abs(pi_probs.sum() - 1.0) < 1e-6           # renormalized
    with pytest.raises(AssertionError):
        _dense_pi([1], [0.6])                         # not a distribution


def _record_with_outcome(prior, n=8, seed=7):
    cfg = SelfPlayConfig(n=n)
    return SelfPlayGame(cfg, seed=seed, game_index=0).play(
        TableEvaluator(prior=prior))


def test_samples_wdl_signs_and_aux_conventions():
    rec = _record_with_outcome({})               # a full natural game
    samples = selfplay_samples(rec)
    assert len(samples) == len(rec.moves)
    # WDL is z from each MOVER's view: X (mover 0) at even plies, O at odd
    for i, smp in enumerate(samples):
        expected = _wdl_onehot(rec.z_white, rec.moves[i].mover)
        assert smp["wdl"].tolist() == expected.tolist()
    # aux: each sample's reply target is the next ply's played cell...
    for i, smp in enumerate(samples[:-1]):
        assert smp["aux_idx"] == rec.moves[i + 1].played_cell
        assert smp["aux_w"] == 1.0
    # ...and the last ply has no reply.
    assert samples[-1]["aux_w"] == 0.0
    assert samples[-1]["aux_mask"].tolist() == samples[-1]["mask"].tolist()


def test_wdl_view_flip_specific():
    """The sign convention on a known decisive outcome: from a dead-lost
    position, perfect values convert the game for X — natural mate, and
    each ply's WDL label is z from that ply's MOVER's view."""
    rec = SelfPlayGame(SelfPlayConfig(n=32, resign_value=-1.5), seed=7,
                       game_index=0, start_cells=LOST_CELLS).play(OracleEvaluator())
    assert rec.outcome_kind == "natural" and rec.z_white == 1.0
    samples = selfplay_samples(rec)
    for smp, mv in zip(samples, rec.moves):     # z from each ply's MOVER
        assert smp["wdl"].tolist() == _wdl_onehot(rec.z_white, mv.mover).tolist()
    movers = {mv.mover for mv in rec.moves}
    assert movers == {0, 1}                     # both sides' labels seen


def test_resign_ply_contributes_no_sample_and_kills_aux():
    first_wins = lambda mover: (1.0, 0.0, 0.0) if mover == 0 else (0.0, 0.0, 1.0)
    rec = SelfPlayGame(SelfPlayConfig(n=8), seed=9, game_index=0).play(
        TableEvaluator(value_fn=first_wins))
    assert rec.outcome_kind == "resign"
    samples = selfplay_samples(rec)
    assert len(samples) == 1                             # white's move only
    assert samples[0]["aux_w"] == 0.0                    # reply was a resignation


def test_samples_at_multiset_semantics():
    rec = _record_with_outcome({})                       # sampled-plies game
    n_playable = sum(1 for m in rec.moves if m.played_cell >= 0)
    if n_playable < 2:
        pytest.skip("fixture game too short")
    rows = samples_at(rec, [0, 0, 1])
    assert len(rows) == 3
    assert all(r == rows[0] for r in rows[:2])           # same ply twice ⇒ 2 rows


def test_buffer_window_and_sampling():
    buffer = TinyReplayBuffer(max_games=2)
    recs = [_record_with_outcome({}, seed=s) for s in (1, 2, 3)]
    buffer.add_all(recs)
    assert len(buffer) == 2                              # deque maxlen
    total = buffer.total_positions
    batch = buffer.sample_tensors(random.Random(0), 512)
    assert batch["tokens"].shape == (512, 18)
    assert batch["mask"].shape == (512, 9)
    assert batch["pi_slots"].shape == (512, PI_SUPPORT_MAX)
    assert set(batch.keys()) == {"tokens", "mask", "pol_idx", "pi_slots",
                                 "pi_probs", "wdl", "aux_idx", "aux_mask", "aux_w"}


def test_masked_log_softmax_contract():
    torch.manual_seed(0)
    logits = torch.randn(4, 9)
    mask = torch.zeros(4, 9, dtype=torch.bool)
    for r in range(4):
        mask[r, torch.randperm(9)[:5]] = True
    logp = masked_log_softmax(logits, mask)
    assert torch.isinf(logp[~mask]).all() and (logp[~mask] < 0).all()
    probs = logp.exp()                                   # exp(-inf) = 0
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-6)
    assert (probs[~mask] == 0.0).all()                   # exactly zero, not NaN


def test_numpy_evaluator_matches_torch():
    torch.manual_seed(1)
    net = TinyNet(hidden=64)
    ev = NumpyEvaluator(net)
    b = TicTacToe()
    b.push(4)
    b.push(0)
    mask = torch.zeros(1, 9, dtype=torch.bool)
    mask[0, list(b.legal_moves)] = True
    with torch.no_grad():
        out = net(torch.from_numpy(b.encode())[None], mask, mask)
    logp, wdl = ev.evaluate(b.encode(), b.legal_moves, b.side_to_move)
    for s in b.legal_moves:
        assert abs(logp[s] - float(out["policy_logp"][0, s])) < 1e-5
    ref = torch.softmax(out["wdl"][0], -1)
    assert max(abs(wdl[i] - float(ref[i])) for i in range(3)) < 1e-5


def test_onehot_pi_loss_equals_sparse_ce():
    """The chess-pinned identity, transplanted: with π′ one-hot on the
    played move, the dense target-weighted CE IS the sparse CE."""
    torch.manual_seed(2)
    net = TinyNet(hidden=64)
    rec = _record_with_outcome({})
    samples = selfplay_samples(rec)
    batch = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
    # force π′ one-hot on pol_idx
    batch["pi_slots"][:] = np.array(batch["pol_idx"])[:, None]
    batch["pi_probs"][:] = 0.0
    batch["pi_probs"][:, 0] = 1.0
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in batch.items()}
    with torch.no_grad():
        out = net(t["tokens"].float(), t["mask"], t["aux_mask"])
        gathered = out["policy_logp"].gather(1, t["pi_slots"].long())
        dense = -(gathered * t["pi_probs"].float()).sum(-1).mean()
        sparse = -out["policy_logp"].gather(
            1, t["pol_idx"].long()[:, None]).squeeze(1).mean()
    assert torch.allclose(dense, sparse, atol=1e-6)


def test_random_start_valid_and_reproducible():
    r1, r2 = random.Random(5), random.Random(5)
    for _ in range(50):
        cells = random_start(6, r1)
        assert TicTacToe(cells).termination == PLAYING
        assert cells == random_start(6, r2)


def test_run_smoke_reaches_perfect_play():
    """The 3.5 gate at CI budget: same loop and recipe, an iteration cap
    with 2.5x headroom over the official run's clean point (it 24) — must
    reach 100% value-preserving play over every reachable position
    (early-stop on a clean full eval)."""
    cfg = SmokeConfig(max_iters=60, eval_every=2)
    result = run_smoke(cfg, log=lambda *_: None)
    assert result.gate
    assert result.final["perfect_pct"] == 100.0
    assert result.final["resign_blunders"] == 0.0
    assert result.wall_seconds < 600                    # "in minutes"
