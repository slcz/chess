#!/bin/bash
# sp_gen_tmux.sh <run> <id> <games> — (re)start this node's generation
# tmux session (survives ssh disconnects; the salad gateway kills
# non-tmux descendants on channel close).
RUN=$1; ID=$2; GAMES=$3
tmux kill-session -t spgen 2>/dev/null || true
cd /workspace/chess
# -c pins the session start dir (a long-lived tmux server keeps its own
# original cwd for new sessions) and the script path is absolute — both
# harden against stale servers (node-2 incident, 2026-10-09).
tmux new-session -d -s spgen -c /workspace/chess \
    "bash /workspace/chess/scripts/sp_gen.sh $RUN $ID $GAMES > /workspace/gen-i$ID.log 2>&1"
echo "spgen started: $RUN-i$ID, $GAMES games, log /workspace/gen-i$ID.log"
