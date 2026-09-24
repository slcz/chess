"""
chess_ai/net/model.py — the chess transformer (Phase 2.1, subtask 2.1).

Architecture (the "Agreed net design" block in PLAN.md Phase 2, decided
2026-09-24; recipe: *Mastering Chess with a Transformer Model*, arXiv
2409.12272, their CF-6M config — adopted architecture and input encoding,
NOT their supervised training regime; our targets come from our own self-play
once Phase 3 lands):

    tokens [B, 64, 117]                     (features.py / the C encoder)
      │ input projection  Linear(117 → d)   (no absolute position embedding:
      ▼                                    Shaw relpos replaces it — their
    8 × pre-LN encoder blocks               ablation: relative > absolute)
        ├─ LN → multi-head attention (8 heads) with Shaw relative
        │      position embeddings (clip k=16), added to the attention
        │      scores AND folded into the values — geometry enters every
        │      layer, matching how pieces actually move
        └─ LN → FFN  Linear(d → 4d) → GELU → Linear(4d → d)
      │ final LN
      ├─ policy head (config `policy_head`, run-1 default = "pooled"):
      │     pooled — mean-pool 64 tokens → Linear(d → 4240); the
      │       AlphaZero/Lc0-style robust choice (user decision
      │       2026-09-24, after the bilinear head's representation ties
      │       complicated the gate)
      │     attention — bilinear from×to head with Shaw relpos in its
      │       scores (~140k params, geometric sharing; kept for A/B —
      │       see PolicyHead for why relpos there is non-negotiable)
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
    relpos_clip: int = 16     # Shaw clip distance k: buckets = 2k+1
    aux_reply: bool = True    # KataGo aux opponent-reply policy head
    policy_head: str = "pooled"   # "pooled" (run-1 default) | "attention"


# ---------------------------------------------------------------------------
# Shaw relative-position multi-head attention
# ---------------------------------------------------------------------------

class ShawAttention(nn.Module):
    """Multi-head self-attention with Shaw et al. (2018) relative positions.

    For query token i and key token j at signed distance j − i (clipped to
    ±k), learned per-head embeddings E_k, E_v of shape [H, 2k+1, d_head]
    contribute:

        score[i,j] += (q_i · E_k[clip(j−i)]) / √d_head
        out[i]     += Σ_j A[i,j] · E_v[clip(j−i)]

    i.e. relative geometry modulates both WHERE attention looks and WHAT it
    reads out. This replaces an absolute per-square input embedding — the
    CF paper's ablation found relative positions strictly better for chess
    (pieces are defined by where they can go, not by their coordinates).
    With 64 tokens the gathered embedding tensor is tiny; no fancy
    factorization is needed.
    """

    def __init__(self, d_model: int, n_heads: int, clip: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.clip = clip
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # ^ residual-branch zero-init: each block starts as the identity.
        # With standard init instead, near-uniform attention at init makes
        # every block add almost the SAME token-averaged vector to every
        # token; 8 layers of that wash out cross-token identity (measured:
        # token std 0.10 → 0.02 while training), the bilinear head then
        # sees identical tokens and the policy sits at exactly uniform —
        # a true stationary point that the random-label memorization gate
        # cannot escape (the 2026-09-24 gate failure). Zeroing the branch
        # outputs makes the trunk exactly the input embeddings at init
        # (squares distinguishable by piece content); attention/FFN
        # weights grow from zero under gradient. GPT-2's residual-
        # projection init scaling, taken to its limit.
        n_buckets = 2 * clip + 1
        self.rel_k = nn.Parameter(torch.zeros(n_heads, n_buckets, self.dh))
        self.rel_v = nn.Parameter(torch.zeros(n_heads, n_buckets, self.dh))
        nn.init.normal_(self.rel_k, std=0.02)
        nn.init.normal_(self.rel_v, std=0.02)

        # bucket[j*64+i] = clip(j−i) + k  — fixed, saved with the module.
        pos = torch.arange(64)
        dist = pos[None, :] - pos[:, None]          # dist[i, j] = j − i
        self.register_buffer("bucket", (dist.clamp(-clip, clip) + clip).long())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        # [B, H, T, dh] views
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)

        # Relative embeddings gathered per (i, j) pair: [H, T, T, dh]
        rel_k = self.rel_k[:, self.bucket]
        rel_v = self.rel_v[:, self.bucket]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)
        scores = scores + torch.einsum("bhid,hijd->bhij", q, rel_k) / math.sqrt(self.dh)
        attn = scores.softmax(dim=-1)
        out = attn @ v + torch.einsum("bhij,hijd->bhid", attn, rel_v)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.proj(out)


class EncoderBlock(nn.Module):
    """Pre-LN transformer encoder block (the stable default; the CF paper's
    post-LN + DeepNet init + Mish recipe is the documented fallback if
    training proves unstable — swap here, nothing else changes)."""

    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = ShawAttention(cfg.d_model, cfg.n_heads, cfg.relpos_clip)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        nn.init.zeros_(self.ff[2].weight)   # same residual-branch zero-init
        nn.init.zeros_(self.ff[2].bias)     # as ShawAttention.proj above

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class PolicyHead(nn.Module):
    """Bilinear from×to policy **with Shaw relpos** + shared promotion linear.

    The from-to region (4096 slots) is an attention score between
    "from-square queries" and "to-square keys", summed over H sub-heads:

        logits[i, j] = Σ_h ( q_ih·k_jh + q_ih·relk_h[clip(j−i)] ) / √dh

    The relpos term is NOT optional decoration — it is what makes the
    policy representable at all. A quiet move's destination square is
    EMPTY, and squares empty for the whole 8-ply window have byte-identical
    input tokens (all piece dims zero; globals are broadcast equally), so
    plain q·k gives *exactly equal* logits to e2e3 and e2e4 — the model
    cannot even express "pawn pushes two" as different from "pawn pushes
    one", and the policy gradient for such tied moves is identically zero
    (the trunk's relpos can't rescue it: with zero-init residual branches
    it only grows if gradients demand it — a chicken-and-egg stall, which
    the 2026-09-24 gate failure traced back to). relk puts the GEOMETRY of
    the move (a knight's L, a pawn's +8) into the logit from init.

    Promotions (144 slots) cannot be expressed in from-to form (the same
    from/to pair is 4 different moves for a promoting pawn), so a small
    shared linear scores each (from, to, direction, promo-piece) slot from
    the concatenation of the two squares' embeddings plus a one-hot of the
    slot's (direction, piece) identity. Two mirrored regions, matching
    features.py: OUR pawns promote "upward" (rel rank 7 → 8); the ENEMY's
    promote downward in our frame (rel rank 2 → 1) — the aux-reply head's
    targets need those (our own moves never use them).
    """

    def __init__(self, d_model: int, n_heads: int, clip: int):
        super().__init__()
        self.h = n_heads
        self.dh = d_model // n_heads
        self.clip = clip
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.rel_k = nn.Parameter(torch.zeros(n_heads, 2 * clip + 1, self.dh))
        nn.init.normal_(self.rel_k, std=0.02)
        pos = torch.arange(64)
        dist = pos[None, :] - pos[:, None]
        self.register_buffer("bucket", (dist.clamp(-clip, clip) + clip).long())
        self.underpromo = nn.Linear(2 * d_model + 6, 1)

        # The 144 promo slots in the SAME order as features.py:
        #   own:   from ∈ rel-rank-7 squares 48..55, dir ∈ {+7, +8, +9}
        #   enemy: from ∈ rel-rank-2 squares  8..15, dir ∈ {−7, −8, −9}
        # each × (N, B, R). Board-edge slots (a/h-file captures) leave the
        # 0..63 range — never legal, hence always masked, so their gathered
        # embedding is irrelevant; clamped to keep the gather in range.
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
        ident = torch.zeros(144, 6)             # one-hot dir (3) ⊕ piece (3)
        for n, s in enumerate(slots):
            up, down = (7, 8, 9), (-7, -8, -9)
            ident[n, (up if s[2] > 0 else down).index(s[2])] = 1.0
            ident[n, 3 + s[3]] = 1.0
        self.register_buffer("frm_idx", frm_idx)
        self.register_buffer("to_idx", to_idx)
        self.register_buffer("ident", ident)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.h, self.dh).transpose(1, 2)   # [B,H,64,dh]
        k = self.w_k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        rel_k = self.rel_k[:, self.bucket]                            # [H,64,64,dh]
        scores = (q @ k.transpose(-2, -1)
                  + torch.einsum("bhid,hijd->bhij", q, rel_k)) / math.sqrt(self.dh)
        logits_ft = scores.sum(dim=1)            # [B, 64, 64]
        logits_ft = logits_ft.reshape(B, -1)     # from*64+to, matches features.py

        feats = torch.cat(
            [x[:, self.frm_idx], x[:, self.to_idx], self.ident.expand(B, -1, -1)],
            dim=-1,
        )                                        # [B, 144, 2d+6]
        logits_up = self.underpromo(feats).squeeze(-1)                 # [B, 144]
        return torch.cat([logits_ft, logits_up], dim=-1)               # [B, 4240]


class PooledPolicyHead(nn.Module):
    """AlphaZero-style pooled policy head — the run-1 default (decided
    2026-09-24 after the bilinear head's failure modes made the gate hard
    to reason about): mean-pool the 64 tokens into one position vector,
    then a single linear to the full 4240-slot move space.

    Trade-offs vs PolicyHead (the "attention" option, kept for A/B):
    ~1.1M params instead of ~140k and no built-in pair structure — but
    also none of its sharp edges: every move's logit is an independent
    weight row, so there is no representation-induced tie (the bilinear
    head needed rel_k in its scores because identical endpoint tokens
    force identical logits). This mirrors how the plan's conv-ResNet
    fallback consumes a trunk: global pooling + FC. Signature matches
    PolicyHead so the two swap by config alone.
    """

    def __init__(self, d_model: int, n_heads: int, clip: int):
        super().__init__()                     # n_heads/clip unused: uniform
        self.proj = nn.Linear(d_model, POLICY_SIZE)   # signature for swapping

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
        head_cls = PooledPolicyHead if cfg.policy_head == "pooled" else PolicyHead
        self.policy = head_cls(cfg.d_model, cfg.n_heads, cfg.relpos_clip)
        self.value = ValueHead(cfg.d_model)
        self.aux = head_cls(cfg.d_model, cfg.n_heads, cfg.relpos_clip) if cfg.aux_reply else None

    def trunk(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(tokens)
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @staticmethod
    def _masked_logp(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """log_softmax over legal slots only; illegal = −inf. A fully
        illegal row (should only ever be an aux row for finished games)
        falls back to unmasked so the tensor stays finite."""
        neg_inf = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg_inf)
        empty = ~mask.any(dim=-1)
        if empty.any():
            masked[empty] = logits[empty]
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
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
