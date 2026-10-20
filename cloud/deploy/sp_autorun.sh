#!/bin/bash
# deploy/sp_autorun.sh — the autonomous self-play container's MAIN process
# (Dockerfile CMD). A salad instance boots the image, this script resolves the
# APPROVED checkpoint, then runs cloud/sp_gen.sh (generation + pusher) and
# exits — container lifecycle = job lifecycle, no tmux, no provisioning.
#
# Approved checkpoint: the lineage lives in the model repo under LINEAGE_DIR
# (default selfplay-scratch/) and is addressed by ONE pointer,
# <LINEAGE_DIR>/approved.json = {generation, file, sha256, ...}. This script
# downloads the file that pointer names and verifies its sha256 before using
# it; approving a new checkpoint = replacing that JSON, never copying files.
# A missing pointer (or a hash mismatch) is FATAL: silently falling back to a
# baked checkpoint would mislabel a whole round's data.
#
# Identity: STAMP = <SALAD_INSTANCE_ID>-<boot epoch>, unique per (instance,
# start). ASSUME A RESTARTED CONTAINER IS CLEAN and may even keep the same
# salad id — so the boot epoch is folded in: SEED = FNV-1a-64(STAMP), upload
# prefix = <DATA_ROOT>/<RUN>-i<STAMP>/. A restart therefore starts a FRESH
# unit. Partial units already on HF stay valid; ingest dedups by
# (unit_seed, game_index).
#
# Container-group env (salad console):
#   SSH_PUBKEY   key-only shell for monitoring (entrypoint.sh)
#   HF_TOKEN     required — the pusher's upload credential
#   RUN          optional, default g<generation>-<ckpt8>-sp<ROUND>k
#   ROUND        optional, the WHOLE round's target games (name only)
#   GAMES        optional, games this unit generates; 0 (default) = forever:
#                the container runs until you stop the group
#   LINEAGE_DIR  optional, default selfplay-scratch
#   DATA_ROOT    optional HF dataset dir, default selfplay-scratch/units
#   SIMS / FAST_SIMS / FAST_P   optional PCR knobs (image bakes 300/50/0.75)
#   SAMPLE_PLYS  optional opening move-sampling window (image bakes 10)
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
export GAMES=${GAMES:-0}
export SIMS=${SIMS:-300} FAST_SIMS=${FAST_SIMS:-50} FAST_P=${FAST_P:-0.75}
export SAMPLE_PLYS=${SAMPLE_PLYS:-10}
export PI_TARGET=${PI_TARGET:-learn}
LINEAGE_DIR=${LINEAGE_DIR:-selfplay-scratch}
DATA_ROOT=${DATA_ROOT:-selfplay-scratch/units}

# ---- approved checkpoint (download + sha256 verify) ----------------------
python3 - "$LINEAGE_DIR" /workspace/approved.pt <<'PY'
import hashlib, json, os, shutil, sys
from huggingface_hub import hf_hub_download
lineage, out = sys.argv[1], sys.argv[2]
err = "[autorun] FATAL"
repo = os.environ.get("HF_MODEL_REPO", "")
if not repo:
    print(f"{err}: HF_MODEL_REPO is not set", flush=True)
    sys.exit(1)
try:
    m = json.loads(open(hf_hub_download(repo, f"{lineage}/approved.json",
                                        repo_type="model")).read())
    src = hf_hub_download(repo, f"{lineage}/{m['file']}", repo_type="model")
except Exception as e:
    raise SystemExit(f"{err}: cannot resolve {lineage}/approved.json — {e}")
h = hashlib.sha256()
with open(src, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
if h.hexdigest() != m["sha256"]:
    raise SystemExit(f"{err}: sha256 mismatch for {m['file']}")
shutil.copy(src, out)
open("/workspace/approved.env", "w").write(
    f"APPROVED_GEN={m['generation']}\nAPPROVED_SHA={m['sha256'][:8]}\nCKPT={out}\n")
print(f"[autorun] approved gen{m['generation']} {m['file']} "
      f"sha256 {m['sha256'][:16]}…")
PY
. /workspace/approved.env
export CKPT APPROVED_GEN APPROVED_SHA
RUN=${RUN:-g${APPROVED_GEN}-${APPROVED_SHA}${ROUND:+-sp$((ROUND / 1000))k}}
export RUN DATA_ROOT ROUND

ID=${SALAD_INSTANCE_ID:-${INSTANCE_ID:-}}
if [ -z "$ID" ]; then
    ID=$(hostname)
    echo "[autorun] WARNING: no SALAD_INSTANCE_ID/INSTANCE_ID — falling " \
         "back to hostname '$ID' as identity" >&2
fi
STAMP="${ID}-$(date +%s)"
SEED=$(python3 - "$STAMP" <<'EOF'
import sys
h = 0xcbf29ce484222325
for b in sys.argv[1].encode():
    h = ((h ^ b) * 0x100000001b3) % (1 << 64)
print(h)
EOF
)

echo "[autorun] instance $STAMP | run $RUN | seed $SEED | \
     games $([ "$GAMES" -le 0 ] && echo forever || echo "$GAMES")"
echo "[autorun] PCR: slow $SIMS, fast $FAST_SIMS @ p=$FAST_P | move sampling: first $SAMPLE_PLYS plies | pi target: $PI_TARGET"

# provenance: a JSON beside the shards, so every unit is self-describing
export STAMP
python3 - <<'PY'
import json, os
e = os.environ
json.dump({
    "run": e["RUN"], "generation": int(e["APPROVED_GEN"]),
    "ckpt_sha256_prefix": e["APPROVED_SHA"], "stamp": e["STAMP"],
    "games_target": int(e.get("GAMES", 0)),
    "round_target": int(e.get("ROUND", 0)), "sims": int(e.get("SIMS", 0)),
    "fast_sims": int(e.get("FAST_SIMS", 0)), "fast_p": float(e.get("FAST_P", 0)),
    "sample_plys": int(e.get("SAMPLE_PLYS", 0)),
    "pi_target": e.get("PI_TARGET", ""),
}, open("/workspace/provenance.json", "w"), indent=2)
PY

echo "[autorun] preflight: HF_TOKEN (the pusher's upload credential)"
[ -n "${HF_TOKEN:-}" ] || {
    echo "[autorun] FATAL: HF_TOKEN not set — GPU hours with no upload" >&2
    exit 1
}

echo "[autorun] preflight: CUDA"
python3 - <<'EOF'
import torch
if not torch.cuda.is_available():
    raise SystemExit("[autorun] FATAL: CUDA not visible in-container")
print("[autorun] GPU:", torch.cuda.get_device_name(0))
EOF

echo "[autorun] starting generation (log: /workspace/sp/$RUN-i$STAMP/gen.log)"
exec cloud/sp_gen.sh "$RUN" "$STAMP" "$GAMES" "$SEED"
