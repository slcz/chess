#!/bin/bash
# deploy/round2b.sh — bf16 A/B on the round-2 config (user experiment
# 2026-09-25): same batch 3072, same LR, same data, RESUMED from the
# round-2 checkpoint — only autocast differs. Read the pair:
#
#   fp32 baseline : round 2's own log (0.51 s/step at batch 3072)
#   bf16 arm      : this run's s/step + the val trajectory 30k -> 32k
#                   (continuing cleanly = bf16 numerics are fine)
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/round2b.sh
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a

PACK=data/lichess_2016-01.pack
[ -f "$PACK" ] || { echo "pack missing — run round2.sh first"; exit 1; }
[ -f runs/value_cloud/latest.pt ] || { echo "no checkpoint to resume"; exit 1; }

echo "== round 2b [$(date -u +%H:%M:%S)]: +2000 steps, batch 3072, BF16 =="
python3 -u -m chess_ai.supervised.train_value train \
    --pack "$PACK" --out runs/value_cloud \
    --steps 2000 --batch-size 3072 --lr 3e-4 --device cuda --bf16 \
    --n-workers 4 --val-every 250 --save-every 500 --log-every 25 \
    --hf-repo chriszhang01/chess-value-net --resume \
    2>&1 | tee runs/value_cloud_bf16.out

echo "== round 2b done — compare s/step vs round 2's 0.51, and val slope =="
