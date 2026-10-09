#!/bin/bash
# sp_uploaded.sh <run_name> — total games PUSHED to HF across the pool.
# Reads each node's pushed.json (written by sp_pusher after each upload).
RUN=${1:?run name}
cd "$(dirname "$0")/.."
N=$(grep -c "^CLOUD_" .cloud.env)
TOTAL=0
for i in $(seq 1 "$N"); do
    P=$(scripts/salad_cloud.sh "$i" ssh "cat /workspace/sp/$RUN-i$i/pushed.json 2>/dev/null || echo '{\"pushed\": 0}'" 2>/dev/null | tail -1)
    N_GAMES=$(echo "$P" | .venv/bin/python -c "import json,sys; print(json.load(sys.stdin).get('pushed', 0))" 2>/dev/null || echo 0)
    echo "  i$i: $N_GAMES uploaded"
    TOTAL=$((TOTAL + N_GAMES))
done
echo "TOTAL_UPLOADED $TOTAL"
