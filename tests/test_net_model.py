"""
tests/test_net_model.py — the network's structural contracts (Phase 2).

The big one is masking: illegal moves must carry probability EXACTLY zero
(risk register §9.3 — an unmasked prior silently poisons search). Also:
parameter budget (~6M, the CF-6M recipe), output shapes/ranges, the
aux-head fallback for finished games, one training step end-to-end, and
an MPS smoke test so device bugs surface now, not during the first cloud
run.

Round 4 (branch Rope-2D) adds the axial-2D-RoPE contracts: the shape/
masking suite runs under BOTH position schemes, and dedicated tests pin
the math the scheme is chosen for — scores depend only on (Δrank, Δfile),
offsets never alias, and the bilinear head breaks identical-endpoint
token ties (the 2026-09-24 gate failure, in test form).
"""
from __future__ import annotations

import pytest
import torch

from chess_ai.net.features import POLICY_SIZE
from chess_ai.net.model import (
    AxialRope,
    ChessTransformer,
    NetConfig,
    RopePolicyHead,
    count_parameters,
)


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(0)
    B = 4
    tokens = torch.randn(B, 64, 117)
    mask = torch.zeros(B, POLICY_SIZE, dtype=torch.bool)
    for b in range(B):
        mask[b, torch.randperm(POLICY_SIZE)[:30]] = True
    return tokens, mask


SCHEMES = ["shaw", "rope2d"]


def test_parameter_budget_default_config():
    # Default = rope2d trunk (round-4 winner, 2026-09-27) + pooled policy
    # head + aux (~2×1.1M). The "attention" head variant stays near the
    # original ~6.8M — assert both so a future head change can't silently
    # blow the budget. shaw twins must be strictly LARGER than their rope
    # twins: Shaw's learned rel_k/rel_v (~17k per attention × 8 layers,
    # plus the bilinear head's rel_k) are the difference; RoPE tables are
    # buffers, not parameters.
    n = count_parameters(ChessTransformer())          # pooled default
    assert 8_400_000 <= n <= 9_500_000, n
    n_shaw = count_parameters(ChessTransformer(NetConfig(pos_scheme="shaw")))
    assert 8_000_000 <= n_shaw <= 9_500_000, n_shaw
    assert n < n_shaw, (n, n_shaw)
    n_attn = count_parameters(ChessTransformer(NetConfig(policy_head="attention")))
    assert 6_500_000 <= n_attn <= 7_200_000, n_attn
    n_shaw_attn = count_parameters(
        ChessTransformer(NetConfig(pos_scheme="shaw", policy_head="attention")))
    assert 6_500_000 <= n_shaw_attn <= 7_200_000, n_shaw_attn
    assert n_attn < n_shaw_attn, (n_attn, n_shaw_attn)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_forward_shapes_and_ranges(batch, scheme):
    tokens, mask = batch
    model = ChessTransformer(NetConfig(n_layers=2, pos_scheme=scheme))
    model.eval()
    with torch.no_grad():
        out = model(tokens, mask)
    assert out["policy_logp"].shape == (4, POLICY_SIZE)
    assert out["wdl"].shape == (4, 3)
    assert out["value"].shape == (4,)
    assert out["aux_logp"].shape == (4, POLICY_SIZE)
    assert (out["value"].abs() <= 1).all()


@pytest.mark.parametrize("scheme", SCHEMES)
def test_illegal_moves_have_exactly_zero_probability(batch, scheme):
    tokens, mask = batch
    model = ChessTransformer(NetConfig(n_layers=2, pos_scheme=scheme))
    model.eval()
    with torch.no_grad():
        out = model(tokens, mask)
    p = out["policy_logp"].exp()
    assert (p[~mask] == 0).all()                    # exact zeros, not ~0
    masses = p.masked_fill(~mask, 0).sum(-1)
    assert torch.allclose(masses, torch.ones(4), atol=1e-5)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_aux_head_all_false_mask_stays_finite(batch, scheme):
    """Finished-game rows pass an all-False aux mask; the head must not
    produce NaN (their loss weight is zero)."""
    tokens, mask = batch
    model = ChessTransformer(NetConfig(n_layers=2, pos_scheme=scheme))
    model.eval()
    am = mask.clone()
    am[0] = False
    with torch.no_grad():
        out = model(tokens, mask, aux_mask=am)
    assert torch.isfinite(out["aux_logp"]).all()


def test_aux_head_optional():
    model = ChessTransformer(NetConfig(n_layers=1, aux_reply=False))
    assert model.aux is None
    out = model(torch.randn(2, 64, 117), torch.ones(2, POLICY_SIZE, dtype=torch.bool))
    assert "aux_logp" not in out


def test_one_train_step_is_finite():
    from chess_ai.train import TrainConfig, Trainer, build_tensors, generate_random_games

    games = generate_random_games(2, seed=3, max_plies=30)
    tensors = build_tensors(games)
    model = ChessTransformer(NetConfig(n_layers=1, d_model=64, n_heads=4, d_ff=128))
    trainer = Trainer(model, TrainConfig(lr=1e-3, batch_size=16, device="cpu"))
    from chess_ai.train.trainer import tensors_to_torch
    batch = tensors_to_torch(
        {k: v[:16] for k, v in tensors.items()}, torch.device("cpu")
    )
    metrics = trainer.train_step(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert 0.0 <= metrics["policy_acc"] <= 1.0


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_forward_and_step(scheme):
    from chess_ai.train import TrainConfig, Trainer, build_tensors, generate_random_games
    from chess_ai.train.trainer import tensors_to_torch

    games = generate_random_games(2, seed=3, max_plies=30)
    tensors = build_tensors(games)
    model = ChessTransformer(NetConfig(n_layers=2, d_model=128, n_heads=4, d_ff=256,
                                       pos_scheme=scheme))
    trainer = Trainer(model, TrainConfig(lr=1e-3, batch_size=16, device="mps"))
    batch = tensors_to_torch(
        {k: v[:16] for k, v in tensors.items()}, torch.device("mps")
    )
    metrics = trainer.train_step(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


# ---------------------------------------------------------------------------
# Axial 2D RoPE contracts (round 4, branch Rope-2D)
# ---------------------------------------------------------------------------

def _constant_content_scores(pairs: int = 4, d_head: int = 32):
    """Score matrix S[i, j] between squares i, j with the SAME content
    vector everywhere — every bit of variation is then pure geometry."""
    torch.manual_seed(0)
    rope = AxialRope(d_head, pairs)
    q0, k0 = torch.randn(d_head), torch.randn(d_head)
    q = q0.expand(1, 1, 64, d_head).contiguous()
    k = k0.expand(1, 1, 64, d_head).contiguous()
    S = (rope(q) @ rope(k).transpose(-2, -1)).squeeze()      # [64, 64]
    return S, q0, k0


def test_rope_scores_depend_only_on_offset():
    """Translation invariance: S[i, j] == S[i+t, j+t] for every in-board
    translation t — the attention score is a function of (Δrank, Δfile)
    only, which is the entire point of swapping Shaw for RoPE."""
    S, _, _ = _constant_content_scores()
    for dr in range(-7, 8):
        for df in range(-7, 8):
            if dr == 0 and df == 0:
                continue
            sq = torch.arange(64)
            f, r = sq % 8, sq // 8
            pair = ((r + dr) >= 0) & ((r + dr) <= 7) & ((f + df) >= 0) & ((f + df) <= 7)
            # both endpoints, AND the pair translated by (+1 rank, +1
            # file), on the board — one mask over all 64 squares
            t = pair & (r <= 6) & (f <= 6) & (r + dr <= 6) & (f + df <= 6)
            i, j = sq[t], sq[t] + dr * 8 + df
            i2, j2 = i + 9, j + 9
            assert torch.allclose(S[i, j], S[i2, j2], atol=1e-5), (dr, df)


def test_rope_no_offset_aliasing():
    """All 225 distinct (Δrank, Δfile) offsets must map to DISTINCT rotations.

    The JOINT encoding is injective for either finest wavelength: the λ ≥ 16
    pairs are individually alias-free over the 15 axis offsets, so they
    disambiguate the single wrap the finest pair can have (λ0 = 8 maps Δ and
    Δ+8 to the same angle). Shaw's flattened-1D-distance + clip ±16 merges
    offsets that geometry keeps apart — RoPE must not. Tested on the rotation
    ANGLES (a property of the ladder, independent of content), plus spot
    checks that the module's tables come from ITS configured ladder."""
    import math

    pairs = 4
    offs = torch.tensor([(dr, df) for dr in range(-7, 8) for df in range(-7, 8)],
                        dtype=torch.float32)                       # [225, 2]
    for lam0 in (16.0, 8.0):
        theta = 2 * math.pi / (lam0 * 2.0 ** torch.arange(pairs))
        fa = offs[:, 1:2] * theta                                  # file bank
        ra = offs[:, 0:1] * theta                                  # rank bank
        enc = torch.cat([fa.cos(), fa.sin(), ra.cos(), ra.sin()], dim=-1)
        assert enc.unique(dim=0).shape[0] == 225

    for lam0 in (16.0, 8.0):
        rope = AxialRope(32, pairs, 2, lam0)
        t0 = 2 * math.pi / lam0
        assert torch.allclose(rope.cos[1, 0], torch.cos(torch.tensor(t0)),
                              atol=1e-6)
        assert torch.allclose(rope.cos[8, pairs], torch.cos(torch.tensor(t0)),
                              atol=1e-6)


def test_rope_rotation_is_orthogonal():
    """R_i is a true rotation: the self-score S[i, i] = q·k exactly, for
    every square (also confirms the unrotated dims pass through)."""
    S, q0, k0 = _constant_content_scores()
    assert torch.allclose(torch.diagonal(S), (q0 * k0).sum(), atol=1e-5)


def test_rope_is_parameter_free_and_validated():
    assert count_parameters(AxialRope(32, 4)) == 0   # tables are buffers
    with pytest.raises(ValueError):                  # banks must fit the head
        AxialRope(8, 4)                              # 4·4=16 > 8 dims


def test_rope_policy_head_breaks_endpoint_ties():
    """The 2026-09-24 gate failure, in test form: tokens for squares that
    are empty across the whole 8-ply window are byte-identical, so with
    identical endpoint tokens a plain q·k head ties the logits exactly
    (e2e3 == e2e4 — the model cannot even express 'pushes two'). The
    rotated head must separate them from init, with no learned geometry
    parameters."""
    torch.manual_seed(1)
    d = 256
    head = RopePolicyHead(d, 8, rope_pairs=4)
    x = torch.randn(1, 1, d).expand(1, 64, d).contiguous()   # identical tokens
    logits = head(x).view(-1)                                  # [4240]
    ft = logits[:4096].view(64, 64)                            # from × to
    assert ft[12, 20] != ft[12, 28]      # e2e3 vs e2e4: Δrank 1 vs 2
    assert ft[12, 20] != ft[12, 19]      # e2d3 vs e2e3: Δ(1,−1) vs (1,0)
    assert ft.std() > 0                  # geometry moves logits at all


@pytest.mark.parametrize("scheme", SCHEMES)
def test_checkpoint_roundtrip_preserves_scheme(scheme, tmp_path):
    """net_config travels with the weights (Trainer.save); a rope2d
    checkpoint must rebuild as rope2d and reproduce its outputs — and old
    shaw checkpoints keep loading (the new keys default in NetConfig)."""
    from chess_ai.train import TrainConfig, Trainer

    torch.manual_seed(0)
    model = ChessTransformer(NetConfig(n_layers=1, d_model=64, n_heads=4,
                                       d_ff=128, aux_reply=False,
                                       pos_scheme=scheme))
    trainer = Trainer(model, TrainConfig(lr=1e-3, batch_size=4, device="cpu"))
    path = tmp_path / "ckpt.pt"
    trainer.save(path)

    loaded = Trainer.load(path)
    assert loaded.model.cfg.pos_scheme == scheme
    tokens = torch.randn(2, 64, 117)
    mask = torch.ones(2, POLICY_SIZE, dtype=torch.bool)
    model.eval()
    loaded.model.eval()
    with torch.no_grad():
        out_a = model(tokens, mask)
        out_b = loaded.model(tokens, mask)
    assert torch.allclose(out_a["policy_logp"], out_b["policy_logp"], atol=1e-6)


def test_legacy_checkpoint_without_scheme_key_loads_as_shaw(tmp_path):
    """Pre-round-4 checkpoints carry no pos_scheme in net_config; they
    must rebuild as SHAW even though rope2d is the default since
    2026-09-27 — a state_dict mismatch (rel_k/rel_v vs rope.cos/sin) is
    otherwise guaranteed. Simulated by stripping the key from a fresh
    shaw checkpoint, exactly what an old file contains."""
    from chess_ai.train import TrainConfig, Trainer

    torch.manual_seed(0)
    model = ChessTransformer(NetConfig(n_layers=1, d_model=64, n_heads=4,
                                       d_ff=128, aux_reply=False,
                                       pos_scheme="shaw"))
    trainer = Trainer(model, TrainConfig(lr=1e-3, batch_size=4, device="cpu"))
    path = tmp_path / "legacy.pt"
    trainer.save(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["net_config"]["pos_scheme"]
    torch.save(payload, path)

    loaded = Trainer.load(path)
    assert loaded.model.cfg.pos_scheme == "shaw"
