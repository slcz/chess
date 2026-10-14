"""
tests/test_regression_160.py — 4-seed single-game regression for the C
PUCT engine's deterministic search behavior.

For seeds 11/22/33/44 the engine plays 160-vs-160 against itself — one
game per seed, 30 sampled opening plies, 250-ply cap, single process —
and the (result, plies) pair is compared against recorded goldens. The
point is NOT that these moves are best; it is that the search is
deterministic and unchanged. The 2026-10-08 transport bug (three
swallowed exceptions in the banked nn_worker plus a dropped WDL softmax)
silently turned every search into uniform-prior noise while all runs
looked healthy — this test makes that class of regression loud.

Compares only the RESULT line (win/loss + ply count), not individual
moves: per-move equality is not needed to catch a broken transport or a
search-semantics change, and stays robust across trivial engine changes
(e.g. telemetry).

Goldens live in tests/fixtures/regression_160.json, keyed by device,
recorded with NN_BF16=0 (the deterministic fp32 path — bf16 rounding
shifts late-game argmax flips). After an INTENTIONAL search-semantics
change, regenerate and re-review:

    PUCT_REGRESSION_UPDATE=1 python3 tests/test_regression_160.py
    PUCT_REGRESSION_UPDATE=1 pytest -q tests/test_regression_160.py

Device: PUCT_REGRESSION_DEVICE overrides; default cuda if available,
else mps. Runtime: ~1-2 min per game on an RTX 3090, several minutes on
MPS.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

try:
    import pytest
except ImportError:          # standalone use: python3 tests/test_regression_160.py
    pytest = None

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / "engine/build/match_puct"
CKPT = REPO / "runs/gen1.pt"
GOLDENS = Path(__file__).resolve().parent / "fixtures/regression_160.json"
SEEDS = [11, 22, 33, 44]


def _device() -> str:
    env = os.environ.get("PUCT_REGRESSION_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _play(seed: int, device: str) -> tuple[str, int]:
    """One 160-vs-160 game; returns (result, plies) from the RESULT line."""
    # deterministic fp32 reference: pin BOTH perf knobs off
    env = dict(os.environ, NN_BF16="0", NN_COMPILE="0")
    r = subprocess.run(
        [str(BIN), str(CKPT), "160", "160", "30", str(seed), "250", device],
        capture_output=True, text=True, timeout=3600, env=env, check=False,
    )
    assert r.returncode == 0, f"engine failed:\n{r.stderr[-800:]}"
    line = next(l for l in r.stdout.splitlines() if l.startswith("RESULT"))
    _, outcome, _reason, plies = line.split()
    return outcome, int(plies.removeprefix("plies="))


def _update_mode(goldens: dict, device: str) -> bool:
    if os.environ.get("PUCT_REGRESSION_UPDATE") != "1":
        return False
    goldens.setdefault(device, {})
    for seed in SEEDS:
        goldens[device][str(seed)] = list(_play(seed, device))
        print(f"seed {seed}: {goldens[device][str(seed)]}", flush=True)
    GOLDENS.write_text(json.dumps(goldens, indent=2, sort_keys=True) + "\n")
    return True


def _load_goldens() -> dict:
    if GOLDENS.exists():
        return json.loads(GOLDENS.read_text())
    return {}                                # first run: record from scratch


if pytest is not None:

    @pytest.mark.parametrize("seed", SEEDS)
    def test_regression_160(seed: int):
        assert BIN.exists(), f"build the engine first: {BIN} missing"
        assert CKPT.exists(), f"checkpoint missing: {CKPT}"
        device = _device()
        goldens = _load_goldens()
        if _update_mode(goldens, device):
            pytest.skip("goldens regenerated")
        expected = goldens[device][str(seed)]    # KeyError = record first
        outcome, plies = _play(seed, device)
        assert [outcome, plies] == expected, (
            f"seed {seed} on {device}: got ({outcome}, {plies}), "
            f"expected {expected} — the search behavior changed")


if __name__ == "__main__":
    device = _device()
    goldens = _load_goldens()
    if os.environ.get("PUCT_REGRESSION_UPDATE") == "1":
        _update_mode(goldens, device)
        print(f"goldens updated for {device}")
    else:
        failures = []
        for seed in SEEDS:
            expected = goldens[device][str(seed)]
            got = list(_play(seed, device))
            status = "ok" if got == expected else "MISMATCH"
            print(f"seed {seed}: {got} expected {expected} -> {status}")
            if got != expected:
                failures.append(seed)
        raise SystemExit(1 if failures else 0)
