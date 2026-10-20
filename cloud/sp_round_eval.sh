#!/bin/bash
# sp_round_eval.sh <ckpt_a> <ckpt_b> [total_games=400] — one-shot match,
# all games in net-lockstep on one GPU (half per color), no promotion
# logic: just play and report. TOTAL line is score from A's perspective.
set -e
cd /workspace/chess
TOTAL=${3:-400}
HALF=$((TOTAL / 2))
python3 scripts/gate_match.py --ckpt-a "$1" --ckpt-b "$2" --games "$HALF" \
    --a-sims 200 --b-sims 200 --book data/noob_3moves.epd --device cuda \
    --rows 2048 2>&1 | tee "/workspace/eval-$(date +%m%d-%H%M).log" \
    | grep -E "^(GAME|SUMMARY|TOTAL)" | tail -6
