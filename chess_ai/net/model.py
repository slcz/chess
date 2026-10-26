"""
chess_ai/net/model.py — the chess transformer (Phase 2.1, subtask 2.1).

Architecture (the "Agreed net design" block in PLAN.md Phase 2, decided
2026-09-24; recipe: *Mastering Chess with a Transformer Model*, arXiv
2409.12272, their CF-6M config — adopted architecture and input encoding,
NOT their supervised training regime; our targets come from our own self-play
once Phase 3 lands):

    tokens [B, 64, 117]                     (features.py / the C encoder)
      │ input projection  Linear(117 → d)   (no absolute position embedding:
      ▼                                    relative positions replace it —
    8 × pre-LN encoder blocks               the CF ablation: relative > abs)
        ├─ LN → multi-head attention (8 heads): axial 2D RoPE on q/k
        │      (parameter-free; scores depend on (Δrank, Δfile) only,
        │      plain SDPA)
        └─ LN → FFN  Linear(d → 4d) → GELU → Linear(4d → d)
      │ final LN
      ├─ policy head (cfg `policy_head`):
      │     attention — bilinear from×to head with axial RoPE in its
      │       scores (~140k params, geometric sharing; the head we train)
      │     pooled — mean-pool 64 tokens → Linear(d → 4240); the
      │       AlphaZero/Lc0-style legacy head
      │   = 4240 slots (features.py move space, decision D1)
      ├─ value head: mean-pool → MLP → 3 logits, W/D/L ("WDL"; chess draw
      │   rates make a 3-way head richer than a tanh scalar; Lc0/CF
      │   practice). The scalar the search consumes is v = pW − pL.
      └─ aux opponent-reply head (KataGo adoption, knob `aux_reply`): a
          second bilinear head predicting the OPPONENT'S best policy for
          the reply — expressed in OUR mover-relative coordinates (the
          trunk is ours; the head learns the enemy's plan in our frame).

Illegal-move masking is this module's most important contract: policy
log-probs are log_softmax(logits masked to −∞ at illegal slots), so illegal
moves have probability EXACTLY zero — asserted in tests, because an
unmasked prior silently poisons search (risk register §9.3).

Training-recipe knobs kept deliberately out of run 1 (per the plan):
post-LN + DeepNet init + Mish (fallback recipe if pre-LN proves unstable),
soft-policy target (T=4), value-error self-prediction, scalar reward head.
The trunk interface is also the seam for the conv-ResNet fallback (same
forward contract, different block) if the checkpoint ladder stalls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_ai.net.features import POLICY_SIZE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NetConfig:
    """All architecture knobs in one place (the scale knobs for the expert
    run are n_layers/d_model/d_ff, per the plan's "same recipe scaled")."""

    token_dim: int = 117      # input features per square (features.py)
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 1024          # 4×d_model, the standard expansion
    aux_reply: bool = True    # KataGo aux opponent-reply policy head
    policy_head: str = "attention"  # "attention" = RopePolicyHead (the only
                                    # head we train now); "pooled" is legacy-only
    pos_scheme: str = "rope2d"  # axial 2D RoPE — the only scheme (the legacy
                              # learned-bucket encoder was removed)
    rope_pairs: int = 2       # RoPE frequency pairs per direction bank (2p
                              # dims each; 4 banks x 2 pairs = 16 of 32 rotated,
                              # the rest is unrotated content)
    rope_dirs: int = 4        # direction banks: 2 = axial (file/rank),
                              # 4 = + both diagonals (spiral); banks cost
                              # no parameters and no extra compute
    rope_lambda: float = 8.0  # finest wavelength λ₀ (ladder λ_i = λ₀·2^i)
    soft_policy: bool = False # KataGo auxiliary soft policy head: predicts
                              # normalize(pi^(1/T)) (T=4, nominal weight 8)


# ---------------------------------------------------------------------------
# Axial 2D RoPE (round-4 position scheme; PLAN §0 action item 2026-09-26)
# ---------------------------------------------------------------------------

class AxialRope(nn.Module):
    """Parameter-free axial 2D rotary position embedding for the 8×8 board.

    RoPE (Su et al. 2021) rotates each query/key by its own position so the
    dot product q·k depends only on the OFFSET between the two positions.
    For the board we split each head's channels into two banks and rotate
    them by DIFFERENT coordinates — axial RoPE (as used for images): with
    d_head = 32 and rope_pairs = 4,

        dims [0:8)    file bank — 4 pairs, rotated by the square's FILE
        dims [8:16)   rank bank — 4 pairs, rotated by the square's RANK
        dims [16:32)  unrotated content (partial rotary, the GPT-NeoX trick:
                      rotating half the dims leaves pure content capacity)

    Because the banks act on disjoint dim slices, the 2×2 block rotations
    compose independently and the attention score between squares i, j
    depends only on (Δrank, Δfile) = (r_j−r_i, f_j−f_i):

        (R_i q) · (R_j k)  =  q · (R_iᵀ R_j) k  =  q · R(Δrank, Δfile) k.

    Frequency ladder: within each bank, pair i has wavelength λ_i =
    rope_lambda·2^i (base 8 → λ = 8, 16, 32, 64). The maximum axis offset on
    8×8 is ±7 (15 values) and a diagonal bank projects to ±9.9, so only the
    λ ≥ 16 pairs are individually alias-free; the FINEST pair (λ = 8) wraps
    once (Δ and Δ+8 share an angle), which buys angular resolution near the
    origin at the cost of that wrap. Together the bank is still injective
    over every offset, and each further pair doubles the wavelength — a
    fixed multiscale basis over Δ, with neither the 1D flattening nor the
    ±16 clip-merge a learned per-bucket scheme suffers.

    Positions are the mover-relative square grid (sq % 8 = file,
    sq // 8 = rank) — the same `rel_square` frame as the tokens, so the
    vertical flip for Black is inherited unchanged and stays orthogonal
    to how position enters attention. No parameters, no per-layer cost
    beyond the rotation itself.
    """

    def __init__(self, d_head: int, pairs: int = 2, dirs: int = 4,
                 lam0: float = 8.0):
        super().__init__()
        if 2 * dirs * pairs > d_head:
            raise ValueError(
                f"axial RoPE needs 2·dirs·pairs ≤ d_head (got dirs={dirs}, "
                f"pairs={pairs}, d_head={d_head})")
        self.pairs = pairs
        self.dirs = dirs
        self.lam0 = lam0
        # Bank m encodes one DIRECTION phi_m = m*pi/dirs via the position's
        # projection onto it; dirs=2 is the classic axial (file, rank) pair,
        # dirs=4 adds both diagonals. theta_i = 2*pi/lambda_i with
        # lambda_i = 16*2^i stays alias-free for every direction (the largest
        # projection offset is (7+7)/sqrt(2) ~ 9.9 < lambda_0 = 16).
        theta = 2 * math.pi / (lam0 * 2.0 ** torch.arange(pairs))
        sq = torch.arange(64)
        file_ = (sq % 8).float()[:, None]
        rank_ = (sq // 8).float()[:, None]

        def _axis(v):    # snap cos/sin of the axis-aligned frames exact
            return 0.0 if abs(v) < 1e-12 else v

        ang = torch.stack(
            [((file_ * _axis(math.cos(m * math.pi / dirs))
               + rank_ * _axis(math.sin(m * math.pi / dirs))) * theta)
             for m in range(dirs)], dim=1)          # [64, dirs, pairs]
        self.register_buffer("cos", ang.cos().reshape(64, dirs * pairs))
        self.register_buffer("sin", ang.sin().reshape(64, dirs * pairs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate x [..., 64, d_head] in place (first 2·dirs·pairs dims only).

        One bank per direction, each rotated with its own cos/sin columns
        (half-split pairing inside the bank: the Llama/GPT-NeoX convention;
        q and k use the same tables, which is all the relative property
        needs). Tables are cast to x's dtype so bf16 autocast keeps SDPA's
        uniform-dtype contract.
        """
        p, d_bank = self.pairs, 2 * self.pairs
        c = self.cos.to(x.dtype)                  # [64, dirs*p]
        s = self.sin.to(x.dtype)
        banks = []
        for m in range(self.dirs):
            xb = x[..., m * d_bank:(m + 1) * d_bank]
            x1, x2 = xb.chunk(2, dim=-1)
            cb, sb = c[:, m * p:(m + 1) * p], s[:, m * p:(m + 1) * p]
            banks.append(torch.cat(
                [x1 * cb - x2 * sb, x2 * cb + x1 * sb], dim=-1))
        return torch.cat(banks + [x[..., self.dirs * d_bank:]], dim=-1)


class RopeAttention(nn.Module):
    """Plain multi-head attention with axial 2D RoPE on q/k.

    Standard qkv/proj with residual-branch zero-init (each block starts as
    the identity); geometry enters scores as q/k rotation, attention is
    fused SDPA.
    """

    def __init__(self, d_model: int, n_heads: int, rope_pairs: int = 2,
                 rope_dirs: int = 4, rope_lambda: float = 8.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.rope = AxialRope(self.dh, rope_pairs, rope_dirs, rope_lambda)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.rope(q.view(B, T, self.h, self.dh).transpose(1, 2))
        k = self.rope(k.view(B, T, self.h, self.dh).transpose(1, 2))
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)     # plain SDPA
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, -1))


class EncoderBlock(nn.Module):
    """Pre-LN transformer encoder block (the stable default; the CF paper's
    post-LN + DeepNet init + Mish recipe is the documented fallback if
    training proves unstable — swap here, nothing else changes)."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        if cfg.pos_scheme != "rope2d":
            raise ValueError(
                f"pos_scheme {cfg.pos_scheme!r}: the legacy learned-bucket "
                "encoder was removed; only rope2d checkpoints load")
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = RopeAttention(cfg.d_model, cfg.n_heads, cfg.rope_pairs,
                                  cfg.rope_dirs, cfg.rope_lambda)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        nn.init.zeros_(self.ff[2].weight)   # same residual-branch zero-init
        nn.init.zeros_(self.ff[2].bias)     # as RopeAttention.proj above

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

def _promo_slot_tables():
    """Index tables for the 144 underpromotion slots, in features.py order
    (single source of truth for every bilinear head):

      own:   from ∈ rel-rank-7 squares 48..55, dir ∈ {+7, +8, +9}
      enemy: from ∈ rel-rank-2 squares  8..15, dir ∈ {−7, −8, −9}
    each × (N, B, R). Board-edge slots (a/h-file captures) leave the 0..63
    range — never legal, hence always masked, so their gathered embedding
    is irrelevant; clamped to keep the gather in range. Returns
    (frm_idx, to_idx, ident[144, 6]) with ident = one-hot dir ⊕ piece.
    """
    slots = []
    for base, dirs in ((48, (7, 8, 9)), (8, (-7, -8, -9))):
        for frm in range(base, base + 8):
            for direction in dirs:
                for piece in range(3):
                    to = min(max(frm + direction, 0), 63)
                    slots.append((frm, to, direction, piece))
    assert len(slots) == 144
    frm_idx = torch.tensor([s[0] for s in slots])
    to_idx = torch.tensor([s[1] for s in slots])
    ident = torch.zeros(144, 6)
    for n, s in enumerate(slots):
        up, down = (7, 8, 9), (-7, -8, -9)
        ident[n, (up if s[2] > 0 else down).index(s[2])] = 1.0
        ident[n, 3 + s[3]] = 1.0
    return frm_idx, to_idx, ident

class RopePolicyHead(nn.Module):
    """Bilinear from×to policy with axial 2D RoPE — the rope2d twin of
    PolicyHead, and the round-4 action item's one genuinely untemplated
    piece (RoPE was designed for causal sequence attention; here it must
    score move ARROWS, not attending tokens).

    Same structure as PolicyHead — per-head q from the from-square, k
    from the to-square, summed scores, shared underpromo linear — but the
    rel_k bucket term is replaced by rotating q/k with the SAME AxialRope
    tables the trunk uses:

        logits[i, j] = Σ_h (R_i q_ih) · (R_j k_jh) / √dh
                     = Σ_h q_ih · R(Δrank, Δfile) k_ih / √dh

    The non-negotiable property survives (see PolicyHead for why it
    exists): a quiet move's destination is EMPTY, and squares empty for
    the whole 8-ply window have byte-identical tokens — plain q·k then
    gives exactly tied logits to e2e3 and e2e4. With rotation the two
    arrows have different (Δrank, Δfile), so the logits differ from
    init, with zero learned parameters. The tie structure that remains —
    identical endpoint tokens AND identical (Δrank, Δfile) — is the
    class of pairs no relative scheme can separate, and it is benign:
    those moves are geometrically indistinguishable.
    """

    def __init__(self, d_model: int, n_heads: int, rope_pairs: int = 2,
                 rope_dirs: int = 4, rope_lambda: float = 8.0):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.rope = AxialRope(self.dh, rope_pairs, rope_dirs, rope_lambda)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.underpromo = nn.Linear(2 * d_model + 6, 1)
        # The 144 promo slots in features.py order (see _promo_slot_tables).
        frm_idx, to_idx, ident = _promo_slot_tables()
        self.register_buffer("frm_idx", frm_idx)
        self.register_buffer("to_idx", to_idx)
        self.register_buffer("ident", ident)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.rope(self.w_q(x).view(B, T, self.h, self.dh).transpose(1, 2))
        k = self.rope(self.w_k(x).view(B, T, self.h, self.dh).transpose(1, 2))
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        logits_ft = scores.sum(dim=1).reshape(B, -1)   # from*64+to

        feats = torch.cat(
            [x[:, self.frm_idx], x[:, self.to_idx], self.ident.expand(B, -1, -1)],
            dim=-1,
        )
        logits_up = self.underpromo(feats).squeeze(-1)
        return torch.cat([logits_ft, logits_up], dim=-1)


class PooledPolicyHead(nn.Module):
    """AlphaZero-style pooled policy head — the run-1 legacy default:
    mean-pool the 64 tokens into one position vector, then a single linear
    to the full 4240-slot move space. Position-blind (scheme enters through
    the trunk only); kept for old checkpoints, nothing trains on it."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, POLICY_SIZE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=1))        # [B, 4240]


class ValueHead(nn.Module):
    """WDL head: mean-pool the 64 tokens, then a small MLP to 3 logits.
    The scalar the search consumes is v = p(W) − p(L) (net.forward output).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.mean(dim=1))           # [B, 3] W/D/L logits


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

def _make_head(cfg: NetConfig) -> nn.Module:
    """Policy/aux head factory (bilinear takes its rope sizing; pooled
    takes neither)."""
    if cfg.policy_head == "pooled":
        return PooledPolicyHead(cfg.d_model)
    return RopePolicyHead(cfg.d_model, cfg.n_heads, cfg.rope_pairs,
                          cfg.rope_dirs, cfg.rope_lambda)


class ChessTransformer(nn.Module):
    """Encoder-only transformer trunk + policy/WDL(+aux) heads.

    forward(tokens, mask, aux_mask) → dict with:
      "policy_logp"  [B, 4240]  log-probabilities, illegal slots = −inf
                              (prob exactly 0 after exp — the hard contract)
      "wdl"          [B, 3]    W/D/L logits (apply log_softmax in the loss)
      "value"        [B]       scalar v = pW − pL ∈ [−1, 1]
      "aux_logp"     [B, 4240] opponent-reply head, same masking contract
                              (only when NetConfig.aux_reply)

    `mask`/`aux_mask` are bool [B, 4240], True = legal. An all-False
    aux_mask row (no reply — the game ended) is legal to pass: the head
    falls back to an unmasked distribution for that row so log_softmax
    stays finite; such rows must carry zero loss weight (data.py does).
    """

    def __init__(self, cfg: NetConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or NetConfig()
        self.input_proj = nn.Linear(cfg.token_dim, cfg.d_model)
        self.blocks = nn.ModuleList(EncoderBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.policy = _make_head(cfg)
        self.value = ValueHead(cfg.d_model)
        self.aux = _make_head(cfg) if cfg.aux_reply else None
        self.soft = _make_head(cfg) if cfg.soft_policy else None

    def trunk(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @staticmethod
    def _masked_logp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log_softmax over legal slots only; illegal = −inf.

        The mask is non-empty by construction everywhere this runs: the C
        engine never evaluates a terminal position (no legal moves = the
        game is over), and the data exporter substitutes the row's own
        policy mask with aux_w = 0 for reply-less records
        (selfplay_data.py) — so no empty-mask fallback exists here. A
        data-dependent `if` on a GPU tensor would sync the device on
        every forward (measured 2026-10-03)."""
        neg_inf = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg_inf)
        return F.log_softmax(masked, dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        aux_mask: torch.Tensor | None = None,
    ) -> dict:
        x = self.trunk(tokens)
        out = {
            "policy_logp": self._masked_logp(self.policy(x), mask),
            "wdl": self.value(x),
        }
        w = F.softmax(out["wdl"], dim=-1)
        out["value"] = w[:, 0] - w[:, 2]          # v = pW − pL
        if self.aux is not None:
            if aux_mask is None:
                aux_mask = mask
            out["aux_logp"] = self._masked_logp(self.aux(x), aux_mask)
        if self.soft is not None:
            out["soft_logp"] = self._masked_logp(self.soft(x), mask)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
