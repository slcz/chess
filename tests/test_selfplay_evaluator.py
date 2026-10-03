"""tests/test_selfplay_evaluator.py — the torch evaluator's spec grammar
and arch-safe checkpoint loading (Step 1).

The evaluator is the seam the C engine's embedded interpreter calls into;
a checkpoint that silently loads into the WRONG architecture would poison
self-play with garbage priors and no error anywhere. Gates:

- SPEC GRAMMAR: "torch:ckpt=<path>[,device=<dev>]" — the device field the
  Mac pilot (mps) and the cloud (cuda) need; unknown keys fail loudly.
- ARCH FROM net_config: the model is built from the checkpoint's saved
  net_config, not today's defaults.
- LEGACY PIN: a checkpoint saved before pos_scheme existed (the 42k-era
  Shaw window) rebuilds as Shaw — the same setdefault pin as Trainer.load.
  If the pin were wrong, strict load_state_dict fails on key mismatch.
"""
from dataclasses import asdict

import pytest
import torch

from chess_ai.net.model import ChessTransformer, NetConfig
from chess_ai.selfplay.sp_evaluator import TorchBatchEvaluator, make_evaluator


def _tiny_cfg(**kw) -> NetConfig:
    return NetConfig(d_model=64, n_layers=1, n_heads=4, d_ff=128, **kw)


def _save_payload(path, net_cfg: NetConfig, with_net_config=True):
    model = ChessTransformer(net_cfg)
    payload = {"model": model.state_dict()}
    if with_net_config:
        payload["net_config"] = asdict(net_cfg)
    torch.save(payload, path)
    return model


def test_ckpt_with_device_key_builds_from_net_config(tmp_path):
    ckpt = tmp_path / "tiny.pt"
    cfg = _tiny_cfg()
    _save_payload(ckpt, cfg)
    ev = make_evaluator(f"torch:ckpt={ckpt},device=cpu")
    assert isinstance(ev, TorchBatchEvaluator)
    # the architecture came from the checkpoint, not today's default
    assert ev.model.cfg.d_model == 64
    assert ev.model.cfg.n_layers == 1


def test_ckpt_without_pos_scheme_pins_shaw(tmp_path):
    ckpt = tmp_path / "legacy.pt"
    cfg = _tiny_cfg(pos_scheme="shaw")
    model = ChessTransformer(cfg)
    net_config = {k: v for k, v in asdict(cfg).items() if k != "pos_scheme"}
    torch.save({"model": model.state_dict(), "net_config": net_config}, ckpt)
    ev = make_evaluator(f"torch:ckpt={ckpt},device=cpu")
    assert ev.model.cfg.pos_scheme == "shaw"


def test_bare_state_dict_default_arch_loads(tmp_path):
    """No net_config in the payload → the arch is today's default; a
    default-arch state dict (the test fixtures' seed_net.pt pattern) loads."""
    ckpt = tmp_path / "bare.pt"
    torch.save(ChessTransformer().state_dict(), ckpt)
    ev = make_evaluator(f"torch:ckpt={ckpt},device=cpu")
    assert ev.model.cfg.d_model == ChessTransformer().cfg.d_model


def test_bare_state_dict_wrong_arch_fails_loudly(tmp_path):
    """The poison case a silent load would create: a state dict whose arch
    differs from the default must raise, never load garbage priors."""
    ckpt = tmp_path / "tiny_bare.pt"
    torch.save(ChessTransformer(_tiny_cfg()).state_dict(), ckpt)
    with pytest.raises(RuntimeError, match="size mismatch"):
        make_evaluator(f"torch:ckpt={ckpt},device=cpu")


def test_unknown_spec_key_fails_loudly():
    with pytest.raises(ValueError, match="unknown torch evaluator keys"):
        make_evaluator("torch:ckpt=x.pt,devices=mps")
    with pytest.raises(ValueError, match="unknown evaluator spec"):
        make_evaluator("onnx:ckpt=x.onnx")


def test_fixture_specs_still_work():
    assert type(make_evaluator("fixture")).__name__ == "FixtureBatchEvaluator"
    assert type(make_evaluator("fixture:white_wins")).__name__ == \
        "FixtureBatchEvaluator"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
