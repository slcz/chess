#!/bin/bash
# deploy/round2.sh — round 2: resume the 20k run for +5000 steps at batch
# 4096 (same LR 3e-4; Adam batch-scaling keeps-or-raises LR, never cuts).
# Run it in an OPEN terminal (salad proxy kills detached processes):
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/round2.sh
#
# Appends to the same runs/value_cloud/log.jsonl (val lines carry the step
# count, so the two stages read as one curve); final_20k.pt preserved.
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a
mkdir -p data runs

PACK=data/lichess_2016-01.pack
MONTH=2016-01

# --- dataset (a restarted container loses its writable layer) ---------------
if [ ! -f "$PACK" ]; then
    if [ ! -f "data/lichess_${MONTH}.pgn.zst" ]; then
        echo "== [$(date -u +%H:%M:%S)] downloading lichess ${MONTH} (~872 MB) =="
        python3 -m chess_ai.supervised.train_value download --month "$MONTH"
    fi
    echo "== [$(date -u +%H:%M:%S)] preprocessing (target 300k games) =="
    python3 -m chess_ai.supervised.train_value preprocess \
        --pgn "data/lichess_${MONTH}.pgn.zst" --out "$PACK" \
        --min-elo 1600 --max-games 300000
else
    echo "== pack already present: $PACK =="
fi

# --- training (foreground; resumes at 20k from the HF-restored latest.pt) ---
mkdir -p runs/value_cloud

RESUME_FLAG=""
[ -f runs/value_cloud/latest.pt ] && RESUME_FLAG="--resume"

echo "== round 2 [$(date -u +%H:%M:%S)]: +7000 steps, batch 3072 ${RESUME_FLAG:-(fresh)} =="
# batch 3072 (user experiment, 2026-09-25): measured 2048 = 31.9/32.6 GB,
# so 3072 likely exceeds fp32 VRAM — the failure mode is thrash, not a
# clean OOM, and it shows within the first 25 steps (log line never
# prints). If that happens: Ctrl-C, drop back to 2048. 7000 × 3072 =
# 21.5M samples ≈ 1 epoch more, total ~2.9.
python3 -u -m chess_ai.supervised.train_value train \
    --pack "$PACK" --out runs/value_cloud \
    --steps 7000 --batch-size 3072 --lr 3e-4 --device cuda \
    --n-workers 4 --val-every 250 --save-every 500 --log-every 25 \
    --hf-repo chriszhang01/chess-value-net $RESUME_FLAG \
    2>&1 | tee runs/value_cloud_round2.out

echo "== round 2 done: runs/value_cloud/final.pt is now the 25k net =="
