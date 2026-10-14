#!/bin/bash
# sp_gen.sh <run_name> <instance_id> [games=22500] [seed] — the generation
# stage of a round for ONE pool node: uninterrupted self-play + pusher
# uploading parts to units/<run>-i<id>/ on the HF dataset repo. Run inside
# tmux on a bare node, or as the autonomous image's job (the container's
# CMD, deploy/sp_autorun.sh — no tmux there: container lifecycle = job
# lifecycle); exits when the run's DONE.json is uploaded (the tail part
# included).
#
#   scripts/sp_gen.sh sp-42k-r3 1 22500 9001
#
# Search-budget env knobs (defaults = the r3 uniform-200 behavior):
#   SIMS=600 FAST_SIMS=50 FAST_P=0.75 scripts/sp_gen.sh ...
# turns on playout cap randomization (KataGo Wu 2020 §4): each step draws
# one pool-wide budget — FAST_SIMS w.p. FAST_P, else SIMS — and only
# slow-cap plies become training samples (sp_round_train filters on
# ply.sims == game n_sims).
#
# SAMPLE_PLYS (default 10): the OPENING move-sampling window — moves are
# sampled ∝ visits (τ=1) for the first SAMPLE_PLYS plies, argmax after.
# This diversifies openings only; it is unrelated to which plies become
# training samples (every ply is recorded; the train filter is the PCR
# slow-cap rule above).
set -e
cd /workspace/chess
RUN=$1; ID=$2; GAMES=${3:-22500}; SEED=${4:-$((9000 + ID))}
SIMS=${SIMS:-200}; FAST_SIMS=${FAST_SIMS:-0}; FAST_P=${FAST_P:-0.75}
SAMPLE_PLYS=${SAMPLE_PLYS:-10}
# .env is the Mac-side secret file; in the autonomous image the secrets
# arrive as container env (HF_TOKEN) — source only if present.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export NN_BF16=1 NN_COMPILE=1

D=/workspace/sp/$RUN-i$ID
rm -rf "$D"; mkdir -p "$D"
echo "== [$RUN-i$ID] generate $GAMES games (pool 1024, sims $SIMS, \
fast $FAST_SIMS @$FAST_P, sample-plies $SAMPLE_PLYS, seed $SEED) =="
engine/build/selfplay runs/gen1.pt "$D" "$GAMES" \
    "$SIMS" 1024 2048 "$SAMPLE_PLYS" 300 4 0.25 0.3 cuda "$SEED" "$FAST_SIMS" "$FAST_P" \
    > "$D/gen.log" 2>&1 &
GEN_PID=$!

echo "== [$RUN-i$ID] pusher: slice + upload every 1000 games =="
python3 scripts/sp_pusher.py "$D" chriszhang01/chess-selfplay \
    "units/$RUN-i$ID" --every 1000 --interval 60
wait "$GEN_PID"
tail -1 "$D/gen.log"
echo "== [$RUN-i$ID] COMPLETE =="
