#!/bin/bash
# sp_gen.sh <run_name> <instance_id> [games=22500] [seed] — the generation
# stage of a round for ONE pool node: uninterrupted self-play + pusher
# uploading parts to units/<run>-i<id>/ on the HF dataset repo. Run inside
# tmux; exits when the run's DONE.json is uploaded (the tail part included).
#
#   scripts/sp_gen.sh sp-42k-r3 1 22500 9001
set -e
cd /workspace/chess
RUN=$1; ID=$2; GAMES=${3:-22500}; SEED=${4:-$((9000 + ID))}
set -a; . ./.env; set +a
export NN_BF16=1 NN_COMPILE=1

D=/workspace/sp/$RUN-i$ID
rm -rf "$D"; mkdir -p "$D"
echo "== [$RUN-i$ID] generate $GAMES games (pool 1024, sims 200, seed $SEED) =="
engine/build/selfplay runs/ckpt_a/net.pt "$D" "$GAMES" \
    200 1024 2048 30 300 4 0.25 0.3 cuda "$SEED" \
    > "$D/gen.log" 2>&1 &
GEN_PID=$!

echo "== [$RUN-i$ID] pusher: slice + upload every 1000 games =="
python3 scripts/sp_pusher.py "$D" chriszhang01/chess-selfplay \
    "units/$RUN-i$ID" --every 1000 --interval 60
wait "$GEN_PID"
tail -1 "$D/gen.log"
echo "== [$RUN-i$ID] COMPLETE =="
