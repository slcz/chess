#!/bin/bash
# deploy/round3.sh — side-track round 3: policy + value trained TOGETHER
# on human moves (aux head stays off, per user decision 2026-09-25).
# This is the reference paper's supervised regime: policy CE on the
# played move + value CE on the game outcome, shared trunk.
#
# Fresh net (not resumed from the value-only 30k run) so the policy read
# is clean. Batch 1024: measured fastest samples/s on this card (9.5k/s
# vs ~6.2k at 2048/3072 — this net is kernel/einsum-bound, not
# FLOP-bound, at fp32).
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/round3.sh
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a
mkdir -p data runs

PACK=data/lichess_2016-01.pack
MONTH=2016-01

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

mkdir -p runs/policy_value
RESUME_FLAG=""
[ -f runs/policy_value/latest.pt ] && RESUME_FLAG="--resume"

echo "== round 3 [$(date -u +%H:%M:%S)]: 20k steps, batch 1024, policy+value, bf16 ${RESUME_FLAG:-(fresh)} =="
python3 -u -m chess_ai.supervised.train_value train \
    --pack "$PACK" --out runs/policy_value \
    --steps 20000 --batch-size 1024 --lr 3e-4 --device cuda --bf16 \
    --policy-weight 1.0 --aux-weight 0.0 \
    --n-workers 4 --val-every 500 --save-every 500 --push-every 2000 --log-every 50 \
    --hf-repo chriszhang01/chess-value-net $RESUME_FLAG \
    2>&1 | tee runs/policy_value.out

echo "== round 3 done: runs/policy_value/final.pt =="
