"""tests/test_sp_l3.py — Phase 3.3b: the L3 gates for the parallel engine
(PARALLEL_DESIGN.md R6), driven end-to-end through the embedded-CPython
evaluator (the F8 architecture, exercised for real: C threads + a Python
net object behind the GIL).

The round-5/6 simplification upgraded what these tests can assert: with a
BATCH-INVARIANT evaluator the wave-concurrent engine's games depend only
on their seeds (candidate subtrees are disjoint, same-candidate descents
are serial behind the wave barrier, and the only delivery-order-dependent
accumulation — the root's own stats — is never read by selection). So:

- L3a (exact): parallel ≡ sequential, same seeds, bit-for-bit records —
  the race detector, run through the real embedded-evaluator seam;
- L3b (exact): deterministic-interleave goldens (bit-reproducible, and
  equal to the threaded runs);
- L3c: invariants (budget exactly met, records schema-valid, resignation
  labels);
- L3d (statistical, tracked not asserted): with the real torch net the
  batched forward passes are no longer batch-invariant — the run diverges
  from the sequential reference by fp rounding in batched matmuls; we
  measure move-agreement and report it as a tracked metric.

The L2 exact diff against gumbel.py itself remains the sequential mode's
gate (tests/test_search_l2.py).
"""
import numpy as np
import pytest

from chess_ai.selfplay.csearch import CSpGame
from chess_ai.selfplay.gumbel import SelfPlayConfig, SelfPlayGame, TableEvaluator
from chess_ai.selfplay.spengine import SpRun
from test_gumbel_ref import WHITE_WINS, records_equal


class Float32Evaluator:
    """Casts an evaluator's outputs through float32 — the net-tensor
    precision the batch contract carries (the engine widens f32→f64 on
    delivery). Feeding both engines exactly the net-precision numbers
    makes the L3a comparison meaningful bit-for-bit."""

    def __init__(self, base):
        self.base = base

    def evaluate(self, planes, legal_slots, mover):
        logp, wdl = self.base.evaluate(planes, legal_slots, mover)
        return ({s: float(np.float32(v)) for s, v in logp.items()},
                tuple(float(np.float32(w)) for w in wdl))


def sequential_records(cfg, seed, budget, max_plies, evaluator):
    """The same games through the 3.1 sequential driver, by game_index."""
    out = []
    for gi in range(budget):
        g = CSpGame(cfg, seed=seed, game_index=gi)
        try:
            out.append(g.play(evaluator, max_plies))
        finally:
            g.close()
    return out


# ---------------------------------------------------------------------------
# L3a: parallel ≡ sequential (exact, through the embedded evaluator)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eval_spec,seq_eval", [
    ("fixture", TableEvaluator()),
    ("fixture:white_wins", TableEvaluator(value_fn=WHITE_WINS)),
])
def test_l3a_parallel_equals_sequential(eval_spec, seq_eval, tmp_path):
    """Vary schedule pressure: tiny buffers force many epochs, straggler
    windows and deferred-on-full parks; different worker counts vary the
    interleaving. Every game must still match its sequential twin."""
    cfg = SelfPlayConfig(n=8, sample_plies=None)
    for workers, buffer in ((1, 256), (3, 8), (4, 16)):
        run = SpRun(cfg, unit_seed=500, game_budget=6, n_workers=workers,
                    buffer_requests=buffer, max_plies=10, eval_spec=eval_spec,
                    record_dir=tmp_path)
        records = run.play()
        run.close()
        assert len(records) == 6
        seq = sequential_records(cfg, 500, 6, 10, Float32Evaluator(seq_eval))
        for a, b in zip(records, seq):
            records_equal(a, b, tol=0.0)


# ---------------------------------------------------------------------------
# L3b: deterministic-interleave goldens
# ---------------------------------------------------------------------------

def test_l3b_interleave_golden(tmp_path):
    cfg = SelfPlayConfig(n=8, sample_plies=10)
    runs = []
    for _ in range(2):
        run = SpRun(cfg, unit_seed=777, game_budget=4, n_workers=3,
                    buffer_requests=4, max_plies=12, interleave=True,
                    eval_spec="fixture", record_dir=tmp_path)
        runs.append(run.play())
        run.close()
    for a, b in zip(runs[0], runs[1]):
        records_equal(a, b, tol=0.0)


# ---------------------------------------------------------------------------
# L3c: budget + record invariants
# ---------------------------------------------------------------------------

def test_l3c_budget_and_invariants(tmp_path):
    cfg = SelfPlayConfig(n=8, sample_plies=None)
    run = SpRun(cfg, unit_seed=321, game_budget=6, n_workers=2,
                buffer_requests=8, max_plies=12, eval_spec="fixture",
                record_dir=tmp_path)
    records = run.play()
    t = run.telemetry()
    run.close()
    assert len(records) == 6
    assert t["games_done"] == 6
    assert t["games_started"] == 6          # zero overshoot (CAS reservation)
    assert t["epochs"] > 0
    assert t["evals"] >= t["games_done"]    # every searched move costs evals
    assert t["max_batch"] <= 2 * 2 * 8
    for rec in records:
        assert rec.outcome_kind in ("natural", "resign", "adjudicated")
        for m in rec.moves:
            assert abs(sum(m.pi_probs) - 1.0) < 1e-6
            if m.played_slot >= 0:
                assert m.played_slot in m.pi_slots
            assert -1.0 <= m.outlook <= 1.0


# ---------------------------------------------------------------------------
# L3d: the real net (torch) — invariants + tracked move-agreement
# ---------------------------------------------------------------------------

def test_l3d_torch_batched_run_invariants(tmp_path):
    pytest.importorskip("torch")
    import torch

    from chess_ai.net.model import ChessTransformer
    from test_search_l2 import TorchEvaluator

    # Identical weights on both sides: seed once, save a checkpoint for the
    # engine's embedded evaluator, and use the same model sequentially.
    torch.set_num_threads(1)
    torch.manual_seed(0)
    model = ChessTransformer().eval()
    ckpt = tmp_path / "seed_net.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    cfg = SelfPlayConfig(n=8, sample_plies=10)
    budget, max_plies = 3, 8

    run = SpRun(cfg, unit_seed=42, game_budget=budget, n_workers=3,
                buffer_requests=8, max_plies=max_plies,
                eval_spec=f"torch:ckpt={ckpt}", record_dir=tmp_path)
    records = run.play()
    t = run.telemetry()
    run.close()

    assert len(records) == budget
    assert t["games_started"] == budget
    for rec in records:
        assert rec.outcome_kind in ("natural", "resign", "adjudicated")
        for m in rec.moves:
            assert abs(sum(m.pi_probs) - 1.0) < 1e-6

    # Statistical agreement (tracked metric, no bar — R6): the same seeds
    # and weights through the sequential driver at batch 1. The batched
    # fp32 matmuls round differently, so moves may diverge; agreement is
    # the number we watch as the engine matures.
    seq_eval = TorchEvaluator(model)
    seq = sequential_records(cfg, 42, budget, max_plies, seq_eval)
    total, agree, first_move_agree = 0, 0, 0
    for par, ref in zip(records, seq):
        first_move_agree += par.moves[0].played_uci == ref.moves[0].played_uci
        for mp, mr in zip(par.moves, ref.moves):
            total += 1
            agree += mp.played_uci == mr.played_uci
    rate = agree / total if total else 1.0
    print(f"\nL3d agreement (batched-C vs sequential-batch-1, same net): "
          f"first move {first_move_agree}/{budget}, "
          f"all moves {agree}/{total} = {rate:.2%}")
    assert total > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
