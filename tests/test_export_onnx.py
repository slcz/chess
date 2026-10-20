"""
test_export_onnx.py — export→parity round trip on a tiny net.

The full-size gate runs via scripts/export_onnx.py (17 MB model, real
positions); this test keeps a fast always-on version in the suite: a
1-layer net, a handful of C-encoded positions, and the same assertions
the CLI gate makes (tolerances + the illegal-slot exact-zero contract).
Skips when onnxruntime is not installed.
"""
import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")     # noqa: F841 — presence check

from chess_ai.net.export_onnx import export_onnx, parity_gate  # noqa: E402
from chess_ai.net.model import ChessTransformer, NetConfig     # noqa: E402


def test_export_parity_roundtrip(tmp_path):
    torch.manual_seed(7)
    cfg = NetConfig(d_model=64, n_layers=1, n_heads=4, d_ff=128,
                    rope_pairs=1, rope_dirs=2, aux_reply=True)
    model = ChessTransformer(cfg)
    onnx_path = export_onnx(model, tmp_path / "tiny.onnx")
    assert onnx_path.exists()

    stats = parity_gate(model, onnx_path, n_positions=4, seed=1)
    # parity_gate already asserts the tolerances and illegal-zero contract;
    # argmax agreement is exact for identical math on untied random weights.
    assert stats["argmax_agree"] == stats["positions"]
