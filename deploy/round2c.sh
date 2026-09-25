#!/bin/bash
# deploy/round2c.sh — throughput arm 3: bf16 + torch.compile
# (reduce-overhead/CUDA graphs) + fused Adam, at batch 4096 (user call:
# utilization % measures kernel residency, not FLOP efficiency — with
# launch overhead gone and bf16 freeing ~6 GB, 4096 deserves a real
# measurement).
#
# Resumes from the 32k checkpoint; 1500 steps is plenty for the read.
# Failure signature if 4096 still doesn't fit: resume line prints, then
# silence past ~30 steps — Ctrl-C and rerun with --batch-size 3072.
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/round2c.sh
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a

PACK=data/lichess_2016-01.pack
[ -f "$PACK" ] || { echo "pack missing — run round2.sh first"; exit 1; }
[ -f runs/value_cloud/latest.pt ] || { echo "no checkpoint to resume"; exit 1; }

echo "== round 2c [$(date -u +%H:%M:%S)]: +1500 steps, batch 4096, bf16 + compile =="
python3 -u -m chess_ai.supervised.train_value train \
    --pack "$PACK" --out runs/value_cloud \
    --steps 1500 --batch-size 4096 --lr 3e-4 --device cuda --bf16 --compile \
    --n-workers 4 --val-every 250 --save-every 500 --log-every 25 \
    --hf-repo chriszhang01/chess-value-net --resume \
    2>&1 | tee runs/value_cloud_c.out

echo "== round 2c done — compare s/step: fp32/3072 0.51 | bf16/3072 0.25 | this =="
