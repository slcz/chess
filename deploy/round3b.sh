#!/bin/bash
# deploy/round3b.sh — round 3 extension: DOUBLE the data, keep training.
#
# Policy top-1 was still climbing ~+1.5pts/5k steps at the 21.5k cut, so
# instead of a second pass over the same positions (epoch-2 memorization
# risk), regenerate the pack at 600k games — deterministic preprocess
# reproduces the original 300k exactly, then 300k fresh games — and
# resume training over the bigger pack.
#
# NOTE: the val split is the last 5% of the pack, so val metrics jump to
# a fresh game sample at the boundary — still honest, different games.
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/round3b.sh
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a
mkdir -p data runs

SMALL_PACK=data/lichess_2016-01.pack
BIG_PACK=data/lichess_2016-01_600k.pack
MONTH=2016-01

if [ ! -f "$BIG_PACK" ]; then
    if [ ! -f "data/lichess_${MONTH}.pgn.zst" ]; then
        echo "== [$(date -u +%H:%M:%S)] downloading lichess ${MONTH} (~872 MB) =="
        python3 -m chess_ai.supervised.train_value download --month "$MONTH"
    fi
    echo "== [$(date -u +%H:%M:%S)] preprocessing 600k games (~10 min; first 300k identical to $SMALL_PACK) =="
    python3 -m chess_ai.supervised.train_value preprocess \
        --pgn "data/lichess_${MONTH}.pgn.zst" --out "$BIG_PACK" \
        --min-elo 1600 --max-games 600000
fi

mkdir -p runs/policy_value
RESUME_FLAG=""
[ -f runs/policy_value/latest.pt ] && RESUME_FLAG="--resume"

echo "== round 3b [$(date -u +%H:%M:%S)]: +20000 steps on 600k-game pack, b4096, bf16 ${RESUME_FLAG:-(fresh)} =="
# b4096 (user call 2026-09-25): measured to fit + be fastest per-sample
# in bf16 on the 5090 (13.2k samples/s vs 6.2k fp32/1024). 20k × 4096 =
# 82M samples ≈ 2 epochs of the 600k pack.
python3 -u -m chess_ai.supervised.train_value train \
    --pack "$BIG_PACK" --out runs/policy_value \
    --steps 20000 --batch-size 4096 --lr 3e-4 --device cuda --bf16 \
    --policy-weight 1.0 --aux-weight 0.0 \
    --n-workers 4 --val-every 500 --save-every 500 --push-every 2000 --log-every 50 \
    --hf-repo chriszhang01/chess-value-net $RESUME_FLAG \
    2>&1 | tee runs/policy_value_b.out

echo "== round 3b done: runs/policy_value/final.pt =="
