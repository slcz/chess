#!/bin/bash
# sp_gen_tmux.sh <run> <id> <games> — (re)start this node's generation
# tmux session (survives ssh disconnects; the salad gateway kills
# non-tmux descendants on channel close).
RUN=$1; ID=$2; GAMES=$3
tmux kill-session -t spgen 2>/dev/null || true
cd /workspace/chess
tmux new-session -d -s spgen \
    "bash scripts/sp_gen.sh $RUN $ID $GAMES > /workspace/gen-i$ID.log 2>&1"
echo "spgen started: $RUN-i$ID, $GAMES games, log /workspace/gen-i$ID.log"
