"""tests/test_ladder.py — the R5 checkpoint-ladder runner.

Gates:

- SEAT DISPATCH: SeatEvaluator routes each evaluation to the side-to-move's
  net (the mechanism that lets two checkpoints meet in one game).
- PAIRED SYMMETRY: identical fixtures on both seats → every seed's two
  games (colors reversed, same RNG pair) are the SAME game, so the pair
  sums to 1 and the match score is exactly 0.5 — the color-cancellation
  property the protocol stands on.
- STRUCTURE: each seed appears exactly twice, once per seat assignment.
- DETERMINISM: same match_seed + deterministic evaluators ⇒ identical
  per-game results (thread scheduling must not leak into outcomes).
- STATS: split halves + seed pairs + the Elo clamp.
- TORCH SMOKE: NetEvaluator loads a tiny checkpoint (net_config-driven
  arch) and one match point plays through on CPU.
"""
import math

import pytest

from chess_ai.selfplay.gumbel import TableEvaluator
from chess_ai.selfplay.ladder import (LadderConfig, MatchResult, NetEvaluator,
                                      SeatEvaluator, play_match)


def _cfg(**kw) -> LadderConfig:
    base = dict(seeds=3, n=4, workers=3, max_plies=10, sample_plies=2)
    base.update(kw)
    return LadderConfig(**base)


def test_seat_dispatch_routes_on_mover():
    calls = []

    class Spy(TableEvaluator):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def evaluate(self, planes, legal_slots, mover):
            calls.append(self.tag)
            return super().evaluate(planes, legal_slots, mover)

    seat = SeatEvaluator(white=Spy("w"), black=Spy("b"))
    seat.evaluate(None, [0, 1], 0)
    seat.evaluate(None, [0, 1], 1)
    seat.evaluate(None, [0, 1], 0)
    assert calls == ["w", "b", "w"]


def test_paired_symmetry_scores_exactly_half():
    """Identical fixtures both seats: the color-reversed partners of a seed
    are the same game (same RNG pair, same policies), so each pair sums to
    1 and A's score is exactly 0.5."""
    res = play_match(TableEvaluator(), TableEvaluator(), _cfg(),
                     a_name="f", b_name="f", log=lambda *_: None)
    assert res.n_games == 6
    assert res.score_a == pytest.approx(0.5, abs=1e-9)
    for seed, (as_white, as_black) in res.seed_pairs().items():
        assert as_white + as_black == pytest.approx(1.0, abs=1e-9)
    # the two partner games are literally the same game
    games = {(g["seed"], g["a_white"]): g for g in res.games}
    for seed in range(3):
        a, b = games[(seed, True)], games[(seed, False)]
        assert a["plies"] == b["plies"]
        assert a["z_white"] == -b["z_white"]   # mirrored seats, same game


def test_seed_structure_and_determinism():
    cfg = _cfg(match_seed=99)
    r1 = play_match(TableEvaluator(), TableEvaluator(), cfg,
                    log=lambda *_: None)
    r2 = play_match(TableEvaluator(), TableEvaluator(), cfg,
                    log=lambda *_: None)
    assert r1.games == r2.games          # thread scheduling did not leak
    by_seed = {}
    for g in r1.games:
        by_seed.setdefault(g["seed"], []).append(g["a_white"])
    assert sorted(by_seed) == [0, 1, 2]
    for seed, seats in by_seed.items():
        assert sorted(seats) == [False, True]   # once per color


def test_split_halves_and_elo_clamp():
    res = MatchResult(a_name="a", b_name="b", cfg=_cfg(seeds=4))
    res.games = [
        {"slot": 0, "seed": 0, "a_white": True, "z_white": 1,
         "outcome_kind": "natural", "plies": 10, "a_score": 1.0},
        {"slot": 1, "seed": 1, "a_white": True, "z_white": 0,
         "outcome_kind": "natural", "plies": 10, "a_score": 0.5},
        {"slot": 2, "seed": 2, "a_white": True, "z_white": -1,
         "outcome_kind": "natural", "plies": 10, "a_score": 0.0},
        {"slot": 3, "seed": 3, "a_white": True, "z_white": 1,
         "outcome_kind": "natural", "plies": 10, "a_score": 1.0},
        {"slot": 4, "seed": 0, "a_white": False, "z_white": -1,
         "outcome_kind": "natural", "plies": 10, "a_score": 1.0},
        {"slot": 5, "seed": 1, "a_white": False, "z_white": 1,
         "outcome_kind": "natural", "plies": 10, "a_score": 0.0},
        {"slot": 6, "seed": 2, "a_white": False, "z_white": 1,
         "outcome_kind": "natural", "plies": 10, "a_score": 0.0},
        {"slot": 7, "seed": 3, "a_white": False, "z_white": 0,
         "outcome_kind": "natural", "plies": 10, "a_score": 0.5},
    ]
    assert res.score_a == pytest.approx(0.5)
    first, second = res.split_halves()   # seeds 0–1 vs 2–3, both colors
    assert (first, second) == (pytest.approx(0.625), pytest.approx(0.375))
    pairs = res.seed_pairs()
    assert pairs[0] == (1.0, 1.0)          # won both
    assert pairs[1] == (0.5, 0.0)          # split
    assert pairs[2] == (0.0, 0.0)          # lost both
    assert res.elo_a_minus_b == 0.0        # exactly 0.5 → 0 Elo
    # clamp: an all-win match stays finite
    for g in res.games:
        g["a_score"] = 1.0
    assert math.isfinite(res.elo_a_minus_b) and res.elo_a_minus_b > 0


def test_torch_net_smoke(tmp_path):
    pytest.importorskip("torch")
    import torch

    from chess_ai.net.model import ChessTransformer, NetConfig
    from chess_ai.train.trainer import TrainConfig, Trainer

    torch.manual_seed(0)
    net = ChessTransformer(NetConfig(d_model=64, n_layers=1, n_heads=4,
                                     d_ff=128))
    Trainer(net, TrainConfig(device="cpu")).save(tmp_path / "tiny.pt")
    ev = NetEvaluator(str(tmp_path / "tiny.pt"), device="cpu")
    assert ev.model.cfg.d_model == 64     # arch came from net_config

    res = play_match(ev, ev, _cfg(seeds=1, n=2, max_plies=6, sample_plies=1),
                     a_name="tiny", b_name="tiny", log=lambda *_: None)
    assert res.n_games == 2
    assert res.score_a == pytest.approx(0.5, abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
