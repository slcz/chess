#!/bin/bash
# scripts/recover_and_launch.sh — ONE command after every salad restart.
#
# The node loses its writable layer (and sometimes its mind) on every
# restart; this script rebuilds everything from remote state and launches
# the training in the foreground of YOUR terminal (the salad proxy kills
# detached processes, so the session must stay open):
#
#   scripts/recover_and_launch.sh
#
# Steps 1–3 retry through the flaky gateway; step 4 (the long run) is
# idempotent — if it dies, rerun this script and it continues from the
# last checkpoint (HF-restored or on-node).
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a
S=scripts/salad.sh

retry() {  # retry a remote step through the flaky gateway
    local n=1
    until "$@"; do
        n=$((n + 1))
        [ $n -gt 4 ] && { echo "step failed after 4 attempts"; exit 1; }
        echo "  ...attempt $n failed, retrying in 20 s"
        sleep 20
    done
}

echo "== 1/4 clone repo on node =="
retry $S ssh bash -c "'cd / && rm -rf /workspace/chess && git clone --depth 1 -b supervised_learning https://slcz:$GITHUB_KEY@github.com/slcz/chess.git /workspace/chess 2>&1 | tail -1'"

echo "== 2/4 bootstrap: engine + tests + CUDA =="
retry $S ssh bash -c "'cd /workspace/chess && bash deploy/bootstrap.sh 2>&1 | tail -4'"

echo "== 3/4 restore round-1 checkpoint (step 20000) from HuggingFace =="
retry $S ssh bash -c "'cd /workspace/chess && echo HF_TOKEN=$HF_TOKEN > .env && HF_TOKEN=$HF_TOKEN python3 -c \"from huggingface_hub import hf_hub_download; print(hf_hub_download(\\\"chriszhang01/chess-value-net\\\", \\\"value_cloud/latest.pt\\\", local_dir=\\\"/workspace/restore\\\"))\" && mkdir -p runs/value_cloud && cp /workspace/restore/value_cloud/latest.pt runs/value_cloud/latest.pt && ls -la runs/value_cloud/'"

echo "== 4/4 dataset (if missing) + round-2 training, foreground =="
echo "   keep this terminal open until training completes (25k steps)"
$S ssh bash /workspace/chess/deploy/round2.sh
