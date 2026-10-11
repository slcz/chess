"""
tests/test_stampede.py — pins the √N=0 first-batch collapse fix
(select_child's sqrt(1+N), puct.c).

The bug (found 2026-10-11 in a live GUI game, fixture
kg1_stampede_game.json): on a fresh tree, the root's children first
become selectable the moment the root's eval row delivers — but no backup
has reached the root yet, so parent visits = 0 and the PUCT score's U
term (C·P·√N) was identically zero. Every child scored Q=0, the strict >
tie-break kept the FIRST child in movegen order, and the entire first
walker batch (8 walkers) stampeded onto it — 15 of the first 16 sims on
an arbitrary 0.010-prior king shuffle (h1g1), which then won argmax-visits
at 200 sims while a mate-in-1 (Rf8#, prior 0.058, exact Q=1.000) starved.
The game shuffled into a threefold repetition in a +15 position.

The assertion: on this exact position at 16 sims, visits must follow the
prior ranking at the top (the mate's exact Q then dominates quickly) and
the gen-order-first move must NOT hold a supermajority. Deterministic
under the fixed selector (MPS forwards are deterministic here).

Skipped without the probe binary + candidate checkpoint (fresh clones).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "engine" / "build" / "puct_probe"
CKPT = ROOT / "runs" / "pcr-iter1-e1.pt"

pytestmark = pytest.mark.skipif(
    not PROBE.exists() or not CKPT.exists(),
    reason="puct_probe binary or fixture checkpoint not built/present",
)

# from tests/fixtures/kg1_stampede_game.json, white's 28th-move think:
# Rf1-f8# is legal (python-chess-verified); h1g1 is gen_legal's FIRST move.
FEN = "1rk5/4Q3/1p1p4/pPpPp2p/P3P3/3P3N/6PP/R1B2R1K w - - 1 31"


def _top(path: str, level: int) -> dict[str, tuple[int, float, float]]:
    out = subprocess.run([str(PROBE), str(CKPT), f"{level}", "mps"],
                         input=FEN, capture_output=True, text=True,
                         timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    tops: dict[str, tuple[int, float, float]] = {}
    for line in out.stdout.splitlines():
        if line.startswith("TOP"):
            _, _pos, _n, uci, visits, q, prior = line.split()
            tops[uci] = (int(visits), float(q), float(prior))
    return tops


def test_no_stampede_mate_found() -> None:
    tops = _top(ROOT, 16)
    # pre-fix: h1g1 had 15 of 16 visits; f1f8 was absent from the top set
    assert "f1f8" in tops, "mate-in-1 not even visited at 16 sims"
    assert tops["f1f8"][0] > 10, "mate not dominating after its exact-Q visit"
    assert tops["f1f8"][1] == 1.0, "Rf8# must back up the exact win"
    if "h1g1" in tops:
        assert tops["h1g1"][0] < tops["f1f8"][0], \
            "gen-order child #0 out-visits the mate"
