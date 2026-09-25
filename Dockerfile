# Environment for the chess value-net side track on a Salad.com RTX 5090
# (Blackwell, sm_120). Adapted from the Qwen-LoRA 5090 Dockerfile.
#
# Only the base environment is baked in. Code and data are NOT part of the
# image — after the instance is up:
#   1. scp the repo tarball to /workspace (project rule: scp, not rsync)
#   2. the C engine compiles ON the instance (deploy/bootstrap.sh runs
#      `make -C engine`): the Mac's arm64 dylib cannot run on x86_64, and
#      per PLAN §4.1 binaries never ship anyway
#   3. the dataset is downloaded+preprocessed ON the instance (the Lichess
#      dump comes from lichess directly — faster than uploading from the
#      Mac); or scp an existing data/*.pack
#
#   docker build -t chess-value:latest .
#   docker tag  chess-value:latest <registry>/<user>/chess-value:latest
#   docker push <registry>/<user>/chess-value:latest
#
# Salad deploy: expose TCP 22, set env SSH_PUBKEY to your public key, then
#   ssh root@<host>  →  /workspace is the working area.

FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf \
    TORCH_HOME=/workspace/hf/torch

# system tools: python, C toolchain (engine builds from source), zstd
# (Lichess .pgn.zst), ssh server (key-only login), git (fallback clone)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        gcc make build-essential \
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

WORKDIR /workspace/chess
EXPOSE 22
ENTRYPOINT ["/entrypoint.sh"]
# default command keeps the container alive; override for one-off jobs
CMD ["sleep", "infinity"]
