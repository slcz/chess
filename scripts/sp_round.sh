#!/bin/bash
# sp_round.sh <run_name> [games=10000] — one full self-play round on the
# GPU node. Run INSIDE tmux (runbook): the three stages chain in one
# session —
#   1. generate: one uninterrupted selfplay run (pool 1024, bf16+compile)
# 2. pusher:   slices ~1K-game parts from the live shard, uploads each
#              to the HF dataset repo (units/<run>/part-XXXX) as they
#              complete — the simulation is never interrupted
#   3. train:   download units → 2 epochs @ batch 512, lr 2e-4 →
#              per-epoch checkpoints saved AND pushed to the ckpt repo
#   4. eval:    400 games iter1-e2 vs the base net (runs/gen1.pt), one
#              shot; swap the arg to iter1-e1.pt for the epoch-1 candidate
#   scripts/sp_round.sh sp-42k-d1 10000
#
# Search-budget env knobs — see sp_gen.sh (SIMS / FAST_SIMS / FAST_P;
# defaults = uniform 200, PCR off). The train stage only ever samples
# slow-cap plies regardless.
set -e
cd /workspace/chess
RUN=${1:?run name, e.g. sp-42k-d1}
GAMES=${2:-10000}
SIMS=${SIMS:-200}; FAST_SIMS=${FAST_SIMS:-0}; FAST_P=${FAST_P:-0.75}
SAMPLE_PLYS=${SAMPLE_PLYS:-10}
# .env is the Mac-side secret file; in the autonomous image the secrets
# arrive as container env (HF_TOKEN) — source only if present.
if [ -f ./.env ]; then set -a; . ./.env; set +a; fi
export NN_BF16=1 NN_COMPILE=1

rm -rf "/workspace/sp/$RUN"; mkdir -p "/workspace/sp/$RUN"
echo "== 1/4 generate $GAMES games (pool 1024, sims $SIMS, \
fast $FAST_SIMS @$FAST_P, sample-plies $SAMPLE_PLYS) =="
engine/build/selfplay runs/gen1.pt "/workspace/sp/$RUN" "$GAMES" \
    "$SIMS" 1024 2048 "$SAMPLE_PLYS" 300 4 0.25 0.3 cuda 7 "$FAST_SIMS" "$FAST_P" \
    > "/workspace/sp/$RUN/gen.log" 2>&1 &
GEN_PID=$!

echo "== 2/4 pusher: slice + upload every 1000 games =="
python3 scripts/sp_pusher.py "/workspace/sp/$RUN" \
    chriszhang01/chess-selfplay "units/$RUN" --every 1000 --interval 60
wait "$GEN_PID"
tail -1 "/workspace/sp/$RUN/gen.log"

echo "== 3/4 train: 2 epochs, batch 512, lr 2e-4 =="
python3 scripts/sp_round_train.py --prefix "units/$RUN" \
    --base-ckpt runs/gen1.pt --out runs/iter1 \
    --epochs 2 --lr 2e-4 --batch 512

echo "== 4/4 eval: iter1-e2 vs base, 400 games =="
scripts/sp_round_eval.sh runs/iter1-e2.pt runs/gen1.pt 400
echo "== round $RUN complete =="
