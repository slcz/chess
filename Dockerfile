# chess value-net image on a Salad.com RTX 5090 (Blackwell, sm_120).
# Adapted from the Qwen-LoRA 5090 Dockerfile.
#
# TWO modes, selected by the container CMD:
#   - autonomous self-play (default, deploy/sp_autorun.sh): the image
#     carries the repo, the C engine compiled IN-image for x86_64, and the
#     one base checkpoint (runs/policy_value_net_42k.pt) — an instance
#     boots straight into generation + HF push, no scp/bootstrap step.
#     Config is container-env only (SSH_PUBKEY, HF_TOKEN, RUN, GAMES, and
#     the optional SIMS/FAST_SIMS/FAST_P PCR knobs); identity comes from
#     salad's injected SALAD_INSTANCE_ID.
#   - interactive (CMD ["sleep", "infinity"]): the old env-only behavior
#     for supervised runs (runbook steps 3–7: scp a tarball, bootstrap,
#     train) — still fully supported, nothing else changed.
#
#   docker buildx build --platform linux/amd64 -t ghcr.io/slcz/chess-value:latest --push .
#
# Salad deploy: expose TCP 22, set env SSH_PUBKEY (and HF_TOKEN for the
# autonomous mode), then ssh root@<host> → /workspace is the working area.

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf \
    TORCH_HOME=/workspace/hf/torch

# system tools: python, C toolchain (the engine builds IN-image —
# python3-dev provides the unversioned python3-config the Makefile calls),
# zstd (Lichess .pgn.zst), ssh server (key-only login), git (fallback clone)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        gcc make build-essential python3-dev \
        zstd \
        git curl ca-certificates \
        openssh-server \
    && rm -rf /var/lib/apt/lists/*

# PyTorch with Blackwell (sm_120) kernels: needs the cu128 wheels
# (torch >= 2.7). Pinned to 2.8.0 = the version the Mac side develops
# against. Installed FIRST and separately so this heavy layer caches
# independently of the small project-deps layer.
RUN pip3 install --no-cache-dir torch==2.8.0 \
        --index-url https://download.pytorch.org/whl/cu128

# project deps (torch pinned above satisfies requirements.txt's torch)
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && pip3 install --no-cache-dir huggingface_hub   # ckpt push-off-instance

# workspace layout + sshd hardening (key-only login)
RUN mkdir -p /var/run/sshd /workspace/hf /workspace/chess \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

COPY deploy/entrypoint.sh /entrypoint.sh
COPY deploy/bootstrap.sh /bootstrap.sh
RUN chmod +x /entrypoint.sh /bootstrap.sh

# ---- autonomous self-play ------------------------------------------------
# The repo is COPYed per .dockerignore (which re-includes exactly one
# checkpoint: runs/policy_value_net_42k.pt). The engine links the image's
# own python3.10 + torch via python3-config --embed, so the baked-in
# PUCT_REPO_ROOT / PUCT_PURELIB paths match the runtime layout. The usage
# gate proves the binary resolves its shared libs (a missing libpython
# would exit 127 and fail the build).
WORKDIR /workspace/chess
COPY . /workspace/chess/
# `all` builds libchess.so too — the engine's EMBEDDED python imports
# chess_ai.* at attach time, and that import loads the shared lib.
RUN make -C engine all build/selfplay \
    && { ./engine/build/selfplay 2>/dev/null; [ $? -eq 2 ]; }
# sp_gen.sh's canonical checkpoint path = runs/ckpt_a/net.pt (a symlink to
# the baked 42k net — one file, two names, zero duplication)
RUN mkdir -p runs/ckpt_a \
    && ln -snf ../policy_value_net_42k.pt runs/ckpt_a/net.pt \
    && chmod +x deploy/sp_autorun.sh

EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
# default = the self-play job: preflight → generate → push to HF → exit;
# the container stopping IS the run finishing. Override with
# ["sleep", "infinity"] for a plain interactive container.
CMD ["/workspace/chess/deploy/sp_autorun.sh"]
