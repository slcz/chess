#!/bin/bash
# Salad container entrypoint: key-only sshd + GPU banner + keep-alive.
# Salad passes SSH_PUBKEY as container env; port 22 is exposed in the
# container-group config. With no key set the container still runs (jobs
# could be baked into the CMD), it just has no interactive access.
set -e
mkdir -p /run/sshd /workspace

if [ -n "$SSH_PUBKEY" ]; then
    mkdir -p /root/.ssh
    printf '%s\n' "$SSH_PUBKEY" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
    echo "[entrypoint] ssh public key installed"
else
    echo "[entrypoint] WARNING: SSH_PUBKEY not set - no ssh access"
fi

/usr/sbin/sshd

echo "[entrypoint] GPUs visible in-container:"
nvidia-smi --list-gpus 2>/dev/null || echo "[entrypoint] nvidia-smi unavailable"

exec "$@"
