"""tests/test_search_l2.py — Phase 3.1: the L2 exact-diff gate
(PARALLEL_DESIGN.md R6).

The C search's *sequential mode* (engine/src/search.c driven through
chess_ai/selfplay/csearch.py) must reproduce the Python reference
(chess_ai/selfplay/gumbel.py) **exactly**: same work-unit seed and game
index ⇒ identical games end-to-end — same moves, same candidate sets, same
π′ (within 1e-6, the pinned gate; the intent is bit-exactness), same
outcomes. Both engines run in this process, fed by the *same* evaluator
object, so every input into the search math is shared; what is being
verified is the port itself — the RNG stream (consumption order included),
the descent order (candidate-major, movegen-order tie-breaks), the backup
signs, the handoff, resignation, records, and the policy-slot mapping
through real positions (including underpromotions).

The parallel engine (3.3b) is NOT verified here — by design it diverges
from the sequential order under batching; that is the L3 ladder's business
(invariants + statistics + deterministic-interleave goldens).
"""
import numpy as np
import pytest

from chess_ai.engine import ffi, lib
from chess_ai.net.features import POLICY_SIZE
from chess_ai.selfplay.csearch import CSpGame
from chess_ai.selfplay.gumbel import (
    SelfPlayConfig,
    SelfPlayGame,
    TableEvaluator,
)
from chess_ai.selfplay.rng import Xoshiro256PP, game_seed
from test_gumbel_ref import FORCED_FEN, WHITE_WINS, records_equal

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
PROMO_FEN = "1n1n4/1P6/8/8/8/8/7k/K7 w - - 0 1"  # push + capture underpromos


# ---------------------------------------------------------------------------
# RNG parity (the L2 precondition, verified over the process boundary)
# ---------------------------------------------------------------------------

def test_game_seed_parity():
    for seed in (0, 1, 123, 0xDEADBEEF):
        for gi in (0, 1, 7, 999):
            assert lib.rng_game_seed(seed, gi) == game_seed(seed, gi)


def test_rng_stream_parity():
    """The generators produce bit-identical streams: raw u64 words, then
    uniform doubles, then Gumbel doubles (the exact values feed z(a) and
    the π′ sampling, so any drift breaks L2 immediately)."""
    for seed in (0, 7, 123):
        py = Xoshiro256PP(seed)
        c = ffi.new("Rng256 *")
        lib.rng_init(c, seed)
        for _ in range(10000):
            assert py.next_u64() == lib.rng_next_u64(c)
        for _ in range(10000):
            assert py.uniform() == lib.rng_uniform(c)
        for _ in range(10000):
            assert py.gumbel() == lib.rng_gumbel(c)


# ---------------------------------------------------------------------------
# L2: C sequential vs Python reference — identical games
# ---------------------------------------------------------------------------

def run_pair(cfg, seed, game_index, start_fen, evaluator, max_plies):
    """Play the same game in both engines; return (python_record, c_record).
    max_plies follows the C convention (None or a nonnegative bound; the
    Python play() takes None for unbounded)."""
    py_max = max_plies if (max_plies is not None and max_plies >= 0) else None
    py_rec = SelfPlayGame(cfg, seed=seed, game_index=game_index,
                          start_fen=start_fen).play(evaluator, py_max)
    c_game = CSpGame(cfg, seed=seed, game_index=game_index, start_fen=start_fen)
    try:
        c_rec = c_game.play(evaluator, max_plies)
    finally:
        c_game.close()
    return py_rec, c_rec


@pytest.mark.parametrize("start_fen", [None, FORCED_FEN, PROMO_FEN, KIWIPETE])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_l2_fixture_identical_games(start_fen, seed):
    """Training-default config (sample ∝ π′ every ply) over positions that
    exercise the sharp corners: a forced move (k=1, no halving, Eq. 10
    outlook), underpromotion slots, castling/ep-heavy Kiwipete."""
    cfg = SelfPlayConfig(n=8, sample_plies=None)
    py_rec, c_rec = run_pair(cfg, seed, 3, start_fen, TableEvaluator(), 14)
    records_equal(py_rec, c_rec, tol=1e-6)


def test_l2_eval_seed_mode_and_resign():
    """Eval-mode knobs: sampled 10-ply seed phase then the deterministic
    survivor (R5); and the white-wins fixture so black resigns immediately
    (the resign record — empty UCI, played_slot −1 — must match too)."""
    cfg = SelfPlayConfig(n=16, sample_plies=10)
    py_rec, c_rec = run_pair(cfg, 5, 0, KIWIPETE, TableEvaluator(), 16)
    records_equal(py_rec, c_rec, tol=1e-6)
    assert py_rec.outcome_kind == "adjudicated"

    cfg_r = SelfPlayConfig(n=8, sample_plies=0,
                           resign_value=-0.90)
    py_rec, c_rec = run_pair(cfg_r, 9, 0, None,
                             TableEvaluator(value_fn=WHITE_WINS), -1)
    records_equal(py_rec, c_rec, tol=1e-6)
    assert py_rec.outcome_kind == "resign" and py_rec.resigner == 1


def test_l2_c_is_deterministic():
    cfg = SelfPlayConfig(n=16, sample_plies=30)
    recs = []
    for _ in range(2):
        g = CSpGame(cfg, seed=11, game_index=4)
        try:
            recs.append(g.play(TableEvaluator(), 10))
        finally:
            g.close()
    records_equal(recs[0], recs[1])


def test_l2_eval_accounting():
    """The handoff property through the shared boundary: move 1 evaluates
    its root; no root evaluation ever happens again in a game with no
    forced moves (sp_request_is_root is the C side's own admission)."""
    cfg = SelfPlayConfig(n=8, sample_plies=None)
    g = CSpGame(cfg, seed=6, game_index=0)
    try:
        root_eval_moves = set()
        max_plies = 8
        lib.sp_set_max_plies(g.g, max_plies)
        while True:
            ev = lib.sp_game_step(g.g)
            if ev == lib.SP_EVENT_DONE:
                break
            if ev == lib.SP_EVENT_NEED_EVAL:
                if g.request_is_root():
                    root_eval_moves.add(lib.sp_record_count(g.g))
                g.fulfil(TableEvaluator())
        assert root_eval_moves == {0}
    finally:
        g.close()


# ---------------------------------------------------------------------------
# L2 with the real net (CPU fp32) — encoder parity through the eval path
# ---------------------------------------------------------------------------

class TorchEvaluator:
    """ChessTransformer as the evaluator (R7 contract): policy_logp at the
    legal slots + WDL softmax. The same model instance serves both engines
    (identical weights), so the comparison isolates the port."""

    def __init__(self, model):
        import torch

        self._torch = torch
        self.model = model

    def evaluate(self, planes, legal_slots, mover):
        torch = self._torch
        x = torch.from_numpy(
            np.array(planes, dtype=np.float32).reshape(1, 64, 117))
        mask = torch.zeros(1, POLICY_SIZE, dtype=torch.bool)
        mask[0, legal_slots] = True
        with torch.no_grad():
            out = self.model(x, mask)
        logp = out["policy_logp"][0]
        lp = {s: float(logp[s]) for s in legal_slots}
        w = torch.softmax(out["wdl"][0], dim=-1)
        wdl = (float(w[0]), float(w[1]), float(w[2]))
        return lp, wdl


@pytest.mark.slow
def test_l2_torch_net_identical_games():
    pytest.importorskip("torch")
    import torch
    from chess_ai.net.model import ChessTransformer

    # Seed: the net's weights must not depend on the global torch RNG
    # state left by earlier tests (unseeded, this test was order-flaky —
    # some random draws exposed a py-vs-C divergence under full-suite
    # ordering, none under single-file runs; 2026-09-29).
    torch.manual_seed(20260929)
    model = ChessTransformer().eval()
    ev = TorchEvaluator(model)
    cfg = SelfPlayConfig(n=8, sample_plies=10)
    for seed, fen in ((0, None), (1, PROMO_FEN)):
        py_rec, c_rec = run_pair(cfg, seed, 0, fen, ev, 8)
        records_equal(py_rec, c_rec, tol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
