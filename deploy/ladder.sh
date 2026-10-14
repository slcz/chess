#!/bin/bash
# ladder.sh — the standing sims ladder (policy vs 100 .. 300 vs 400) via
# gate_match.py: two uci_mux processes of the SAME checkpoint, per-phase
# sims, net-lockstep waves, 100 games per rung (50 per color).
#
#   tmux new-session -d -s ladder \
#     "bash deploy/ladder.sh <ckpt> <games-per-wave> <book> <seed> [device] \
#      > /workspace/ladder.log 2>&1"
#
# a-sims 1 = the raw-policy player (think(1) + argmax prior). Numerics:
# bf16 + compile default on; exported so older images match.
set -e
CKPT=${1:?ckpt}
GAMES=${2:-50}
BOOK=${3:-data/prior_book_42k.epd}
SEED=${4:-42}
DEVICE=${5:-cuda}
cd "$(dirname "$0")/.."
export NN_BF16=1 NN_COMPILE=1

for rung in "1 100" "100 200" "200 300" "300 400"; do
    set -- $rung
    echo "== RUNG $1 vs $2 =="
    python3 scripts/gate_match.py --ckpt-a "$CKPT" --ckpt-b "$CKPT" \
        --a-sims "$1" --b-sims "$2" --games "$GAMES" --book "$BOOK" \
        --max-plys 250 --device "$DEVICE" --seed "$SEED"
done
echo "== LADDER DONE =="
