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

echo "== round 2 [$(date -u +%H:%M:%S)]: +5000 steps, batch 2048 ${RESUME_FLAG:-(fresh)} =="
# batch 2048, not 4096: the transformer's attention maps (batch*H*64*64
# floats, held per layer for backward) push 4096 to the 5090's VRAM
# ceiling — measured thrash: GPU pinned at 32.1/32.6 GB, <25 steps in
# 3+ min. 2048 fits with headroom; same LR (Adam).
python3 -u -m chess_ai.supervised.train_value train \
    --pack "$PACK" --out runs/value_cloud \
    --steps 5000 --batch-size 2048 --lr 3e-4 --device cuda \
    --n-workers 4 --val-every 250 --save-every 500 --log-every 25 \
    --hf-repo chriszhang01/chess-value-net $RESUME_FLAG \
    2>&1 | tee runs/value_cloud_round2.out

echo "== round 2 done: runs/value_cloud/final.pt is now the 25k net =="
