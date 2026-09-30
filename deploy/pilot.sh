#!/bin/bash
# deploy/pilot.sh — the chess self-play pilot on a salad node (Steps 2–3:
# 4.1/4.2 cloud orchestration, n-sweep, 5–20k-game run, churn drill).
#
# Runs INSIDE the chess container (repo at /workspace/chess), after .env
# has been scp'd alongside it. Idempotent at every step: re-running after
# a churn kill rebuilds nothing that exists, re-uses the fetched seed
# checkpoint, and the loop itself resumes from latest.pt + unit shards
# (pushed to HF after every round — fresh-box state comes from remote
# alone, PLAN §5 rule 1).
#
#   scripts/salad.sh ssh bash /workspace/chess/deploy/pilot.sh [--n 8 ...]
#
# Anything after the defaults is passed to the loop (later flags win),
# e.g. `pilot.sh --n 8` for the n-sweep, `pilot.sh --out runs/chess_pilot_n16`
# for a separate arm. Long job ⇒ tmux (the ssh proxy kills every descendant
# when the channel closes — runbook 0); watch with:
#   scripts/salad.sh ssh tail -20 /workspace/chess/runs/chess_pilot.out
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
set -a; source .env 2>/dev/null || true; set +a
: "${HF_TOKEN:?HF_TOKEN must be set (in .env on the node)}"
HF_REPO="${HF_REPO:-chriszhang01/chess-selfplay-pilot}"

echo "== [$(date -u +%H:%M:%S)] 1/4 build C engine (x86_64) =="
# everything: libchess (FFI), libchess_sp (the loop's seam), the binary,
# the test binaries the slice runs
make -C engine

echo "== [$(date -u +%H:%M:%S)] 2/4 fast test slice (binary + loop) =="
python3 -m pytest -q tests/test_selfplay_cli.py tests/test_selfplay_loop.py

echo "== [$(date -u +%H:%M:%S)] 3/4 seed checkpoint from HF =="
# The side track's repos use per-run PREFIXES inside chess-value-net
# (policy_value_rope/…), not one repo per run.
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

echo "== [$(date -u +%H:%M:%S)] 4/4 launch the loop under tmux =="
# The ssh proxy kills every descendant when the channel closes (runbook 0)
# — long jobs need tmux, and a fresh image may not carry it.
if ! command -v tmux >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq tmux
fi
if tmux has-session -t chesspilot 2>/dev/null; then
    echo "tmux session 'chesspilot' already running — nothing to do"
    exit 0
fi
tmux new-session -d -s chesspilot \
    "python3 -m chess_ai.train.selfplay_loop \
       --out runs/chess_pilot --hf-repo $HF_REPO \
       --device cuda --eval-device cuda \
       --workers 8 --buffer-requests 128 \
       --games-per-iter 128 --train-steps 96 \
       --resign -1 --start-plies 8 \
       --save-every 20 --iters 400 \
       $* > runs/chess_pilot.out 2>&1"
echo "launched: tmux attach -t chesspilot   (or watch runs/chess_pilot.out)"
echo "stop:    tmux kill-session -t chesspilot"
