#!/bin/bash
# provision_gate.sh — reprovision a fresh salad node for gate matches
# (puct_mcts branch). Idempotent: safe to rerun after any restart.
#
#   scp the repo tarball first (runbook §3a), then:
#   ssh root@<node> bash /workspace/chess/deploy/provision_gate.sh
#
# Assumes data/noob_3moves.epd was scp'd into data/ (gitignored) and
# HF_TOKEN is in /workspace/chess/.env (written by the caller).
set -e
set -o pipefail
cd /workspace/chess

echo "== 1/5 engine + drivers =="
# Python.h for the embedded-Python evaluators (not in the base image)
python3 -c "import sysconfig; assert __import__('os').path.exists(
    sysconfig.get_paths()['include'] + '/Python.h')" 2>/dev/null \
    || { apt-get update -qq && apt-get install -y -qq python3-dev; }
# tmux: long runs must survive the ssh channel (runbook field notes).
# stockfish: the Elo gauntlet — apt puts it at /usr/games/stockfish,
# which is NOT on a tmux shell PATH; pass that absolute path.
apt-get install -y -qq tmux stockfish 2>&1 | tail -1
make -C engine 2>&1 | tail -1          # libchess.so (the Python FFI) + CLI
make -C engine build/uci_mux build/uci_puct build/match_puct \
     build/sf_gauntlet build/puct_probe 2>&1 | tail -1

echo "== 2/5 engine test gate =="
make -C engine check 2>&1 | tail -4

echo "== 3/5 python deps =="
python3 -c "import torch" 2>/dev/null || pip3 install torch --index-url https://download.pytorch.org/whl/cu128
python3 -c "import chess, cffi, pytest, huggingface_hub" 2>/dev/null \
    || pip3 install -r requirements.txt huggingface_hub
python3 -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"

echo "== 4/5 book + HF token =="
[ -f data/noob_3moves.epd ] \
    && echo "book: $(wc -l < data/noob_3moves.epd) positions" \
    || { echo "book MISSING — scp data/noob_3moves.epd"; exit 1; }
grep -q HF_TOKEN .env || { echo "HF_TOKEN missing in .env"; exit 1; }

echo "== 5/5 gate checkpoints =="
# ckpt_a = THE 42k net (the ladder + goldens reference). NEVER fetch it
# from HF by name - policy_value/latest.pt is a DIFFERENT checkpoint
# (wrong-net incident 2026-10-10). Ship it by scp:
#   scripts/salad.sh put runs/policy_value_net_42k.pt \
#     /workspace/chess/runs/ckpt_a/net.pt
SHA=$(sha256sum runs/ckpt_a/net.pt 2>/dev/null | cut -c1-16)
[ "$SHA" = "2a3fb75b55bc48de" ] \
    && echo "ckpt_a sha256 $SHA = the 42k reference net" \
    || { echo "ckpt_a MISSING or not the 42k net (sha256 $SHA)";
         echo "scp runs/policy_value_net_42k.pt first"; exit 1; }
set -a; . ./.env; set +a
python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import os, shutil

got = hf_hub_download("chriszhang01/chess-value-net",
                      "policy_value_rope/final.pt",
                      token=os.environ["HF_TOKEN"],
                      local_dir="/workspace/restore")
os.makedirs("runs/ckpt_b", exist_ok=True)
shutil.copy(got, "runs/ckpt_b/net.pt")
print("restored runs/ckpt_b/net.pt <- policy_value_rope/final.pt")
EOF
ls -la runs/ckpt_a runs/ckpt_b
echo "== provision done =="
