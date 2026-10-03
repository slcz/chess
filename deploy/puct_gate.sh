#!/bin/bash
# deploy/puct_gate.sh — the PUCT-gated run (user call 2026-10-02: "training
# in Gumbel MCTS, evaluating in PUCT MCTS"): FRESH from the RoPE 25k seed,
# the v3 training recipe (dense pi_prime + AdamW wd 1e-4 + outlook
# downweight, n=16 Gumbel self-play), with the GATE swapped to the C
# AZ-PUCT walk (--gate-search az, n-eval 128 = the fleet-measured
# PUCT-128 config, 100 games at the upgraded protocol).
#
# Why: the 2026-10-02 cross-implementation matrix showed the Gumbel gate
# measures "calibration under full sigma-commitment" and DECOUPLES from
# strength (v3-r70: raw policy +83, PUCT +70, Gumbel gate -112) — the
# PUCT gate tracks policy quality, so this run's question is "does the
# PUCT ladder climb from the baseline" (can the loop evolve under an
# instrument that reads real strength).
#
# Threshold 0.55 (not the old 0.60): AlphaZero's 55% promotion rule; PUCT
# gates are draw-heavy (~78%), so 0.60 demands a near-unconvertible
# margin before any promotion. False accepts self-correct at the next
# gate (the lucky champion is re-tested against its own continuation).
#
# Mirror of eq9fix.sh's bootstrap. Runs INSIDE the chess container
# (/workspace/chess, .env alongside), idempotent at every step; the gated
# runner itself is crash-safe (champion.pt + state.json resume).
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/puct_gate.sh [--c-visit 15 ...]
#
# Watch:  scripts/salad.sh ssh tail -20 /workspace/chess/runs/puct_gate.out
# Gates:  scripts/salad.sh ssh cat /workspace/chess/runs/puct_gate/gated/gated_log.jsonl
#         (per-game gate progress: runs/puct_gate/gated/eval_live.log)
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a
: "${HF_TOKEN:?HF_TOKEN must be set (in .env on the node)}"
HF_REPO="${HF_REPO:-chriszhang01/chess-selfplay-pilot}"

echo "== [$(date -u +%H:%M:%S)] 0/4 system deps (python headers, tmux, toolchain) =="
if [ ! -f /usr/include/python3.10/Python.h ] || ! command -v tmux >/dev/null 2>&1 \
   || ! command -v make >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq python3.10-dev tmux build-essential
fi

echo "== [$(date -u +%H:%M:%S)] 1/4 build C engine (x86_64, incl. az.c) =="
make -C engine

echo "== [$(date -u +%H:%M:%S)] 2/4 test slice (loop + data + az C-parity + ladder) =="
python3 -m pytest -q tests/test_selfplay_cli.py tests/test_selfplay_loop.py \
    tests/test_selfplay_data.py tests/test_az_c.py tests/test_ladder.py

echo "== [$(date -u +%H:%M:%S)] 3/4 seed checkpoint from HF =="
mkdir -p runs
python3 - <<'EOF'
import os, shutil
from huggingface_hub import hf_hub_download
if os.path.exists("runs/policy_value_net_rope_25k.pt"):
    print("seed ckpt already present")
else:
    p = hf_hub_download("chriszhang01/chess-value-net", "policy_value_rope/final.pt",
                        token=os.environ["HF_TOKEN"])
    shutil.copy(p, "runs/policy_value_net_rope_25k.pt")
    print("fetched", p, "-> runs/policy_value_net_rope_25k.pt")
EOF

echo "== [$(date -u +%H:%M:%S)] 4/4 launch the PUCT-gated run under tmux =="
if tmux has-session -t gated 2>/dev/null; then
    echo "tmux session 'gated' already running — nothing to do"
    exit 0
fi
tmux new-session -d -s gated \
    "PYTHONPATH=/workspace/chess python3 -m scripts.gated_run \
       --out runs/puct_gate \
       --champion-init runs/policy_value_net_rope_25k.pt \
       --policy-target pi_prime --lr 2e-5 --weight-decay 1e-4 --outlook-downweight \
       --train-steps 768 --n 16 --rounds-per-iter 10 --games-per-iter 1024 \
       --max-plies 256 \
       --gate-search az --n-eval 128 --gate-games 100 --gate-threshold 0.55 \
       --iterations 20 \
       --hf-repo $HF_REPO \
       $* > runs/puct_gate.out 2>&1"
echo "launched: tmux attach -t gated   (or watch runs/puct_gate.out)"
echo "stop:    tmux kill-session -t gated"
