#!/bin/bash
# deploy/pi_prime_fix.sh — recipe-v3 gated run (2026-10-02): dense π' policy target —
# eq9 policy target, n=16 (restored for playout speed), AdamW weight decay
# 1e-4, KataGo-style outlook sample downweighting, FRESH from the RoPE seed
# (the round-30 n64 net carries the saturated value head these fixes target).
#
# Mirror of pilot.sh's bootstrap, ending in scripts.gated_run instead of
# the bare loop. Runs INSIDE the chess container (/workspace/chess, .env
# alongside), idempotent at every step. The gated runner itself is
# crash-safe: champion.pt + state.json resume; killed iterations re-run.
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/pi_prime_fix.sh [--n 32 ...]
#
# Watch:  scripts/salad.sh ssh tail -20 /workspace/chess/runs/pi_prime_fix.out
# Gates:  scripts/salad.sh ssh cat /workspace/chess/runs/pi_prime_fix/gated/gated_log.jsonl
#         (per-game gate progress: runs/pi_prime_fix/gated/eval_live.log)
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

echo "== [$(date -u +%H:%M:%S)] 1/4 build C engine (x86_64) =="
make -C engine

echo "== [$(date -u +%H:%M:%S)] 2/4 fast test slice (binary + loop + data) =="
python3 -m pytest -q tests/test_selfplay_cli.py tests/test_selfplay_loop.py \
    tests/test_selfplay_data.py

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

echo "== [$(date -u +%H:%M:%S)] 4/4 launch the gated run under tmux =="
if tmux has-session -t gated 2>/dev/null; then
    echo "tmux session 'gated' already running — nothing to do"
    exit 0
fi
tmux new-session -d -s gated \
    "PYTHONPATH=/workspace/chess python3 -m scripts.gated_run \
       --out runs/pi_prime_fix \
       --champion-init runs/policy_value_net_rope_25k.pt \
       --policy-target pi_prime --lr 2e-5 --weight-decay 1e-4 --outlook-downweight \
       --train-steps 768 --n 16 --rounds-per-iter 10 --games-per-iter 1024 \
       --max-plies 256 --gate-games 32 --gate-threshold 0.60 --iterations 20 \
       --hf-repo $HF_REPO \
       $* > runs/pi_prime_fix.out 2>&1"
echo "launched: tmux attach -t gated   (or watch runs/pi_prime_fix.out)"
echo "stop:    tmux kill-session -t gated"
