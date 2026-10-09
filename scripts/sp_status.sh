#!/bin/bash
# sp_status.sh <run_name> — generation progress across the .cloud.env pool.
# Per instance: games played (from gen.log), last pusher line, DONE flag.
RUN=${1:?run name}
cd "$(dirname "$0")/.."
N=$(grep -c "^CLOUD_" .cloud.env)
for i in $(seq 1 "$N"); do
    echo "--- instance $i ---"
    scripts/salad_cloud.sh "$i" ssh "echo games: \$(grep -c '^GAME' /workspace/sp/$RUN-i$i/gen.log 2>/dev/null); tail -1 /workspace/gen-i$i.log 2>/dev/null; [ -f /workspace/sp/$RUN-i$i/DONE.json ] && echo RUN_DONE"
done
