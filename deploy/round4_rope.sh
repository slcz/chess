#!/bin/bash
# deploy/round4_rope.sh — round 4: the axial-2D-RoPE A/B (PLAN §0 action
# item 2026-09-26, branch Rope-2D).
#
# Retrains the supervised net with the trunk's Shaw relpos replaced by
# axial 2D RoPE (--pos-scheme rope2d; PLAN §0: split head channels into
# rank/file banks, parameter-free rotation, scores depend on (Δrank,
# Δfile), plain SDPA, no rel_v).
#
# SCHEDULE (user call 2026-09-26): ONE phase — 25000 steps at batch 4096
# bf16, lr 3e-4, fresh net. Sample-matched to the round-3/3b Shaw
# baseline (42k steps = ~22k @ b1024 + ~20k @ b4096 ≈ 105M samples; ours
# = 25k × 4096 ≈ 102M) — same exposure, so the A/B isolates the position
# scheme without paying the baseline's extra epochs. Same pack →
# byte-identical 5% val split and fixed val picks; b4096 measured
# fastest per-sample on the 5090 in round 3b.
#
# Baseline to match-or-beat (round 3b @ 42k): held-out policy top-1
# 50.7%, value acc 60.2% / NLL 0.794 (n=8192); mid-run @ 21.5k: policy
# 41.7%, value 60.5%. Decision rule (PLAN §0): no performance AND no
# convergence degradation → RoPE replaces Shaw as the default scheme.
#
# Churn-safe per §5: atomic checkpoints every 500 steps, HF push every
# 2000 (runs/policy_value_rope/latest.pt in chriszhang01/chess-value-net
# — separate dir from the Shaw baseline, lineage untouched), re-running
# this script resumes from whatever step latest.pt holds. Expected ~3 h
# on a 5090 (~$2; the $5/8h cap from PLAN §0 still applies).
#
# PREREQ (Mac, once — node-side pack build is only the fallback below):
#   scripts/salad.sh put data/lichess_2016-01_600k.pack /workspace/chess/data/
#
# Run (under tmux ON THE NODE — user rule 2026-09-26: the Mac may sleep or
# drop the ssh channel; the run must survive it. The script is idempotent
# and checkpoint-resumable, so even a tmux death (the 2026-09-24 field
# note claims ALL channel descendants die) loses ≤500 steps — just
# re-launch the same tmux line):
#   scripts/salad.sh ssh bash -c "'tmux new-session -d -s rope \
#     \"bash /workspace/chess/deploy/round4_rope.sh > /workspace/chess/runs/round4_rope.log 2>&1\"'"
#   watch: scripts/salad.sh ssh bash -c "'tail -n 40 /workspace/chess/runs/round4_rope.log'"
#   attach: scripts/salad.sh ssh -t ... 'tmux attach -t rope'  (or read the
#   tee'd runs/policy_value_rope_{a,b}.out)
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a
mkdir -p data runs

BIG_PACK=data/lichess_2016-01_600k.pack
MONTH=2016-01
OUT=runs/policy_value_rope
HF_REPO=chriszhang01/chess-value-net
TOTAL_STEPS=25000   # b4096 ≈ 102M samples ≈ the baseline's ~105M

# --- 0. pre-flight: the A/B net builds + forwards on THIS gpu (fail fast,
#        before any download; also prints the param delta vs Shaw's 8,729,891)
python3 - <<'EOF'
import torch
from chess_ai.net.model import ChessTransformer, NetConfig, count_parameters
m = ChessTransformer(NetConfig(pos_scheme="rope2d")).cuda().eval()
toks = torch.randn(8, 64, 117, device="cuda")
mask = torch.ones(8, 4240, dtype=torch.bool, device="cuda")
with torch.no_grad():
    out = m(toks, mask)
print(f"pre-flight OK: rope2d {count_parameters(m):,} params "
      f"(shaw twin 8,729,891), policy {tuple(out['policy_logp'].shape)}, "
      f"finite={torch.isfinite(out['policy_logp']).all().item()}, "
      f"gpu={torch.cuda.get_device_name(0)}")
EOF

# --- 1. data: the baseline's own pack (PREREQ scp). Fallback = node-side
#        build — deterministic single-threaded preprocess; the quarantined
#        worker path is NOT used. Sanity-check the size: ~212 MB, never KB.
if [ ! -s "$BIG_PACK" ]; then
    echo "== pack missing — FALLBACK node-side build (~15 min) =="
    if [ ! -f "data/lichess_${MONTH}.pgn.zst" ]; then
        echo "== [$(date -u +%H:%M:%S)] downloading lichess ${MONTH} (~872 MB) =="
        python3 -m chess_ai.supervised.train_supervised download --month "$MONTH"
    fi
    python3 -m chess_ai.supervised.train_supervised preprocess \
        --pgn "data/lichess_${MONTH}.pgn.zst" --out "$BIG_PACK" \
        --min-elo 1600 --max-games 600000
fi
ls -lh "$BIG_PACK"

steps_done () {
    [ -f "$OUT/latest.pt" ] && \
    python3 -c "import torch; print(torch.load('$OUT/latest.pt', map_location='cpu', weights_only=False)['steps'])" \
    || echo 0
}

# --- 2. train: one phase, b4096 bf16, to 25k (fresh or churn-resumed) --
# --resume with no latest.pt just starts fresh (cmd_train guards on the
# file), so the flag is unconditional and churn-recovery is symmetric.
DONE=$(steps_done)
if [ "$DONE" -lt "$TOTAL_STEPS" ]; then
    echo "== train [$(date -u +%H:%M:%S)]: step ${DONE} -> $TOTAL_STEPS @ b4096 bf16 =="
    mkdir -p "$OUT"
    python3 -u -m chess_ai.supervised.train_supervised train \
        --pack "$BIG_PACK" --out "$OUT" \
        --steps $((TOTAL_STEPS - DONE)) --batch-size 4096 --lr 3e-4 \
        --device cuda --bf16 --policy-weight 1.0 --aux-weight 0.0 \
        --pos-scheme rope2d \
        --n-workers 4 --val-every 500 --save-every 500 --push-every 2000 \
        --log-every 50 --hf-repo "$HF_REPO" --resume \
        2>&1 | tee "$OUT".out
fi

# --- 3. read-out: big val + the bot sanity matches, with the Shaw
#        baseline numbers inline for the A/B call
echo "== [$(date -u +%H:%M:%S)] final eval: 20k held-out positions =="
echo "== Shaw baseline @42k: policy top-1 0.5073, value acc 0.6016, NLL 0.7943 =="
python3 -m chess_ai.supervised.train_supervised eval \
    --ckpt "$OUT/final.pt" --pack "$BIG_PACK" --n-positions 20000

echo "== matches (Shaw baseline: policy-vs-greedy 17-0-3, value-vs-greedy 19-0-1, policy-vs-value 10-0) =="
python3 -m chess_ai.supervised.train_supervised match \
    --ckpt "$OUT/final.pt" --player policy --opponent greedy --n-games 20
python3 -m chess_ai.supervised.train_supervised match \
    --ckpt "$OUT/final.pt" --player value --opponent greedy --n-games 20
python3 -m chess_ai.supervised.train_supervised match \
    --ckpt "$OUT/final.pt" --player policy --opponent value --n-games 10

echo "== round 4 done: $OUT/final.pt (HF: $HF_REPO/policy_value_rope/final.pt) =="
