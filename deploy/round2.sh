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
set -a; source .env; set +a

echo "== round 2 [$(date -u +%H:%M:%S)]: +5000 steps, batch 4096, resume =="
python3 -u -m chess_ai.supervised.train_value train \
    --pack data/lichess_2016-01.pack --out runs/value_cloud \
    --steps 5000 --batch-size 4096 --lr 3e-4 --device cuda \
    --n-workers 4 --val-every 250 --save-every 500 --log-every 25 \
    --hf-repo chriszhang01/chess-value-net --resume \
    2>&1 | tee runs/value_cloud_round2.out

echo "== round 2 done: runs/value_cloud/final.pt is now the 25k net =="
