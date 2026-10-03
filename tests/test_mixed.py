"""tests/test_mixed.py — the mixed-search head-to-head driver (same weights,
one seat PUCT, one seat Gumbel). Fixture nets only: these tests pin the
MACHINERY (termination, adjudication, determinism, pairing), never strength
— uniform-prior fixtures make every position look the same to both seats."""
import pytest

from chess_ai.selfplay.gumbel import TableEvaluator
from chess_ai.selfplay.mixed import MixedConfig, _play_one_mixed, play_mixed_match


@pytest.fixture(scope="module")
def fixture_evals():
    ev = TableEvaluator()          # uniform priors + WDL: machinery, no strength
    return {"az": ev, "gumbel": ev}


def _cfg(**kw):
    defaults = dict(seeds=2, n=16, max_plies=16, sample_plies=4,
                    workers=4, match_seed=7)
    defaults.update(kw)
    return MixedConfig(**defaults)


def test_mixed_game_completes_and_respects_ply_cap(fixture_evals):
    for slot in range(4):
        g = _play_one_mixed(slot, _cfg(), fixture_evals, "az", "gumbel")
        assert g["plies"] <= 16
        assert g["outcome_kind"] in ("natural", "adjudicated")
        assert g["z_white"] in (-1, 0, 1)
        assert g["a_score"] in (0.0, 0.5, 1.0)
        # color pairing: slots k and k+seeds share the seed
        assert g["seed"] == slot % 2


def test_mixed_match_is_deterministic_in_seed(fixture_evals):
    s1 = play_mixed_match(fixture_evals, _cfg(), "A", "B", log=lambda *_: None)
    s2 = play_mixed_match(fixture_evals, _cfg(), "A", "B", log=lambda *_: None)
    for key in ("score_a", "draws", "games", "draws_by_search"):
        assert s1[key] == s2[key], key


def test_mixed_match_summary_shape(fixture_evals):
    s = play_mixed_match(fixture_evals, _cfg(), "A", "B",
                         a_search="az", b_search="gumbel",
                         log=lambda *_: None)
    assert s["games"] == 4 and s["seeds"] == 2
    assert s["search_a"] == "az" and s["search_b"] == "gumbel"
    # draws keyed by color orientation: az-as-white vs az-as-black
    assert sum(s["draws_by_search"].values()) == s["draws"]
    assert set(s["draws_by_search"]) <= {"az-vs-gumbel", "gumbel-vs-az"}
    assert 0.0 <= s["score_a"] <= 1.0
    assert s["wall_s"] >= 0
