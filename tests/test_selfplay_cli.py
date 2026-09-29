"""tests/test_selfplay_cli.py — Phase 3.4 (the salad-node binary):
engine/build/chess_selfplay, the standalone work-unit runner.

Gates (PLAN §0 next-subtask, 3.4):

- PARITY: the binary — its own Py_Initialize, its own mmap'd static block,
  the same embedded evaluator — reproduces the Python seam's (SpRun) games
  bit-for-bit from the same unit seed. With the batch-invariant fixture
  evaluator the engine is deterministic, so any divergence points at the
  binary's plumbing (interpreter setup, config marshalling, shard dump).
- IDEMPOTENT RE-ISSUE: DONE.json present ⇒ exit 0, shards untouched (the
  cloud rule: re-issuing a completed unit is a no-op).
- CRASH RECOVERY: a killed unit (simulated: partial shards, no DONE.json)
  replays deterministically into the O_TRUNC'd shards and completes.
- MANIFEST: DONE.json carries what the trainer's read_unit needs
  (unit_seed, n_workers, games_done cross-checked against the shards).
"""
import json
import subprocess
from pathlib import Path

import pytest

from chess_ai.selfplay.gumbel import SelfPlayConfig
from chess_ai.selfplay.spengine import SpRun
from chess_ai.train.selfplay_data import read_unit
from test_gumbel_ref import records_equal

REPO = Path(__file__).resolve().parents[1]
BIN = REPO / "engine" / "build" / "chess_selfplay"


def _pythonpath():
    """sys.path prefixes the embedded interpreter needs: the repo root
    (chess_ai) + site-packages. Match the Makefile's PYTHON — the venv,
    when there is one — so numpy imports into the same minor version the
    binary embeds; else fall back to the running interpreter's."""
    paths = [str(REPO)]
    venv_lib = REPO / ".venv" / "lib"
    sites = sorted(venv_lib.glob("python*/site-packages")) \
        if venv_lib.exists() else []
    if sites:
        paths.append(str(sites[-1]))
    else:
        import sysconfig
        paths.append(sysconfig.get_paths()["purelib"])
    return ":".join(paths)


PYTHONPATH = _pythonpath()


def build_binary():
    """Fresh build (idempotent); a compiler/python-headers failure fails
    the test loudly — the environment already builds libchess_sp."""
    subprocess.run(["make", "-C", str(REPO / "engine"),
                    "build/chess_selfplay"],
                   check=True, capture_output=True, text=True)


def run_unit(out_dir, *, seed, games, extra=()):
    return subprocess.run(
        [str(BIN), "--seed", str(seed), "--games", str(games),
         "--out", str(out_dir), "--workers", "2", "--buffer", "8",
         "--max-plies", "10", "--pythonpath", PYTHONPATH, *extra],
        capture_output=True, text=True, timeout=120)


@pytest.fixture(scope="module")
def built_binary():
    if not BIN.exists():
        build_binary()
    return BIN


@pytest.fixture(scope="module")
def unit_dir(built_binary, tmp_path_factory):
    """One completed unit (6 fixture games), used by several tests."""
    out = tmp_path_factory.mktemp("unit") / "u"
    r = run_unit(out, seed=500, games=6)
    assert r.returncode == 0, r.stderr
    return out


# ---------------------------------------------------------------------------
# Parity: binary ≡ the Python seam (SpRun), same unit seed, bit-for-bit
# ---------------------------------------------------------------------------

def test_binary_matches_sprun_bit_for_bit(unit_dir, tmp_path):
    cfg = SelfPlayConfig()
    run = SpRun(cfg, unit_seed=500, game_budget=6, n_workers=2,
                buffer_requests=8, max_plies=10, eval_spec="fixture",
                record_dir=tmp_path)
    try:
        sprun_records = run.play()
    finally:
        run.close()
    bin_records = read_unit(unit_dir)
    assert len(sprun_records) == len(bin_records) == 6
    for a, b in zip(sprun_records, bin_records):
        records_equal(a, b, tol=0.0)


def test_manifest_matches_shards(unit_dir):
    manifest = json.loads((unit_dir / "DONE.json").read_text())
    assert manifest["format"] == "SPSHARD1"
    assert manifest["unit_seed"] == 500
    assert manifest["games_done"] == 6 == len(read_unit(unit_dir))
    assert manifest["n"] == SelfPlayConfig().n
    assert manifest["game_budget"] == 6


# ---------------------------------------------------------------------------
# Cloud-rules behavior: idempotent re-issue, crash recovery
# ---------------------------------------------------------------------------

def test_reissue_of_complete_unit_is_noop(unit_dir):
    before = {p.name: p.read_bytes()
              for p in sorted(unit_dir.iterdir())}
    r = run_unit(unit_dir, seed=500, games=6)
    assert r.returncode == 0, r.stderr
    assert "nothing to do" in r.stdout
    after = {p.name: p.read_bytes()
             for p in sorted(unit_dir.iterdir())}
    assert before == after          # shards and manifest untouched


def test_killed_unit_replays_and_completes(built_binary, tmp_path):
    """Simulate the churn case: partial shards (a truncated tail), no
    DONE.json. Re-running the SAME command must complete the unit with the
    same deterministic games (O_TRUNC resets the shards)."""
    out = tmp_path / "partial"
    out.mkdir()
    r = run_unit(out, seed=777, games=4)
    assert r.returncode == 0, r.stderr
    shard = out / "shard-w0.spbin"
    data = shard.read_bytes()
    shard.write_bytes(data[: len(data) - 30])     # the kill's torn tail
    (out / "DONE.json").unlink()                  # ...and no completion

    r2 = run_unit(out, seed=777, games=4)
    assert r2.returncode == 0, r2.stderr
    records = read_unit(out)                      # now complete + valid
    assert len(records) == 4

    # determinism: the replayed unit equals a fresh twin run
    twin = tmp_path / "twin"
    r3 = run_unit(twin, seed=777, games=4)
    assert r3.returncode == 0, r3.stderr
    for a, b in zip(records, read_unit(twin)):
        records_equal(a, b, tol=0.0)


def test_bad_args_fail_with_usage(built_binary):
    r = subprocess.run([str(BIN)], capture_output=True, text=True)
    assert r.returncode == 1
    assert "usage" in r.stderr
    r = subprocess.run([str(BIN), "--seed", "1", "--games", "-2",
                        "--out", "/tmp/x"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "usage" in r.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
