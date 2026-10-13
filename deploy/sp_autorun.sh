#!/bin/bash
# deploy/sp_autorun.sh — the autonomous self-play container's MAIN process
# (Dockerfile CMD). A salad instance boots the image, this script preflights,
# then runs scripts/sp_gen.sh (generation + pusher) and exits — container
# lifecycle = job lifecycle, no tmux, no provisioning.
#
# Identity: STAMP = <SALAD_INSTANCE_ID>-<boot epoch>, unique per (instance,
# start). ASSUME A RESTARTED CONTAINER IS CLEAN (nothing on its fs survives)
# and may even keep the same salad id — so the boot epoch is folded in:
# SEED = FNV-1a-64(STAMP), upload prefix = units/<RUN>-i<STAMP>/. A restart
# therefore starts a FRESH unit — never the same game sequence, never
# overwriting a previous unit's files. Partial units already on HF stay
# valid; ingest dedups by (unit_seed, game_index). Round state is read from
# HF (scripts/sp_progress.py), never from the container.
#
# Container-group env (salad console):
#   SSH_PUBKEY   key-only shell for monitoring (entrypoint.sh)
#   HF_TOKEN     required — the pusher's upload credential
#   RUN          optional, default sp-42k-pcr
#   GAMES        optional, per-instance target, default 6250
#                (pick ceil(25000 / n_instances); total ≈ 25k)
#   SIMS / FAST_SIMS / FAST_P   optional PCR knobs, default 600 / 50 /
#                0.75 (the image may bake tighter defaults via ENV;
#                runtime env still wins)
#   SAMPLE_PLYS  optional opening move-sampling window (τ=1 plies then
#                argmax), default 30 — the image may bake a tighter
#                default via ENV; runtime env still wins
#
# The base checkpoint is BAKED into the image at
# runs/policy_value_net_42k.pt (= runs/ckpt_a/net.pt, the path sp_gen.sh
# uses) — the container needs nothing from HF at boot.
set -e
cd /workspace/chess
export PYTHONUNBUFFERED=1
RUN=${RUN:-sp-42k-pcr}
GAMES=${GAMES:-6250}
export SIMS=${SIMS:-600} FAST_SIMS=${FAST_SIMS:-50} FAST_P=${FAST_P:-0.75}
export SAMPLE_PLYS=${SAMPLE_PLYS:-10}

# Identity source, first available: salad's injected id, a manual env
# override, else the container HOSTNAME (in Docker the hostname IS the
# container id — so even an orchestrator that injects nothing still yields
# a unique-per-boot stamp once the epoch is folded in below). A collision
# would only duplicate work (ingest dedups by (unit_seed, game_index)),
# never corrupt; refusing to boot over a missing env would be worse.
ID=${SALAD_INSTANCE_ID:-${INSTANCE_ID:-}}
if [ -z "$ID" ]; then
    ID=$(hostname)
    echo "[autorun] WARNING: no SALAD_INSTANCE_ID/INSTANCE_ID — falling " \
         "back to hostname '$ID' as identity" >&2
fi
STAMP="${ID}-$(date +%s)"

# Seed: FNV-1a 64 of the stamp (fits the engine's uint64 seed).
SEED=$(python3 - "$STAMP" <<'EOF'
import sys
h = 0xcbf29ce484222325
for b in sys.argv[1].encode():
    h = ((h ^ b) * 0x100000001b3) % (1 << 64)
print(h)
EOF
)

echo "[autorun] instance $STAMP | run $RUN | seed $SEED | games $GAMES"
echo "[autorun] PCR: slow $SIMS, fast $FAST_SIMS @ p=$FAST_P | move sampling: first $SAMPLE_PLYS plies"

echo "[autorun] preflight: base ckpt"
CKPT=runs/policy_value_net_42k.pt
[ -f "$CKPT" ] || { echo "[autorun] FATAL: $CKPT missing from image" >&2; exit 1; }
mkdir -p runs/ckpt_a
[ -e runs/ckpt_a/net.pt ] || ln -s "../policy_value_net_42k.pt" runs/ckpt_a/net.pt

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
exec scripts/sp_gen.sh "$RUN" "$STAMP" "$GAMES" "$SEED"
