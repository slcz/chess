#!/bin/bash
# deploy/cloud_run.sh — dataset prep + training, FOREGROUND edition.
#
# The salad ssh proxy kills every descendant process when the channel
# closes (measured 2026-09-24: setsid+nohup+disown all die), so long
# jobs must run inside an OPEN interactive session. Run this from the
# Mac and keep the window open until training finishes:
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/cloud_run.sh
#
# Everything also tees to runs/value_cloud.out + log.jsonl on the node,
# so a second terminal (or the agent) can poll progress with short
# ssh calls while this one runs.
#
# Idempotent: existing pack short-circuits prep; existing latest.pt
# resumes training instead of restarting it.
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
mkdir -p data runs

PACK=data/lichess_2016-01.pack
MONTH=2016-01

# --- dataset ---------------------------------------------------------------
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

# --- training (foreground; this session IS the lifetime of the run) --------
set -a; source .env 2>/dev/null || true; set +a   # HF_TOKEN for ckpt push
mkdir -p runs/value_cloud

RESUME_FLAG=""
[ -f runs/value_cloud/latest.pt ] && RESUME_FLAG="--resume"

echo "== [$(date -u +%H:%M:%S)] training: 20k steps, batch 1024, cuda ${RESUME_FLAG:-(fresh)} =="
python3 -u -m chess_ai.supervised.train_value train \
    --pack "$PACK" --out runs/value_cloud \
    --steps 20000 --batch-size 1024 --lr 3e-4 --device cuda \
    --n-workers 4 --val-every 500 --save-every 500 --log-every 50 \
    --hf-repo chriszhang01/chess-value-net $RESUME_FLAG \
    2>&1 | tee runs/value_cloud.out

echo "== [$(date -u +%H:%M:%S)] done: runs/value_cloud/final.pt =="
