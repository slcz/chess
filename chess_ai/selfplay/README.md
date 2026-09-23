# chess_ai/selfplay/ — batched self-play

The throughput core (Phase 3): a Python outer loop over N games in lockstep — C advances each game until it needs an NN evaluation and returns packed input planes for pending leaves; Python runs one batched forward pass per round; priors + values return as arrays. Also home to the game recorder, the replay-buffer writer, and the checkpoint-ladder match runner (net@A vs. net@B).

## Search algorithm (Phase 3.2 — agreed 2026-09-23)

Gumbel AlphaZero (Danihelka et al., ICLR 2022). n = 2–32 simulations per move. **No
Dirichlet noise, no move-selection temperature** — candidate sampling via Gumbel-top-m
and a visit-scaled σ take over both exploration jobs. (Authoritative copy + training
targets + discussion: PLAN.md, Phase 3, "Agreed search design".)

```text
GUMBEL_SEARCH(s, n):
    m      = min(n, 16, #legal)             # candidates; m <= n => all visited in round 1
    g(a)   ~ Gumbel(0,1)  per legal action
    z(a)   = logits(a) + g(a)
    A      = top-m actions by z(a)
    rounds = ceil(log2 m)                   # from the ORIGINAL m — do not recompute
    while |A| > 1:
        plays = ceil(n / (rounds * |A|))    # split the round's slice over survivors
        for a in A: plays x PUCT-descent(child(s,a)); back up, sign-flip every ply
        A = top ceil(|A|/2) of A by [ z(a) + (C_VISIT + max_b N(b)) * C_SCALE * q(a) ]
    return the single survivor              # the move; no visit sampling, no tau
                                            # C_VISIT=50, C_SCALE=1.0
```

Training target for the policy head is NOT the visit distribution (AGZ) but the
improved policy `pi' = softmax(logits + sigma(completedQ))` with
`completedQ(a) = q(a) if N(a)>0 else v_hat(s)` — distilled as KL (i.e. plain
target-weighted cross-entropy). Value target stays the game outcome z.

