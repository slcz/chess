#!/bin/bash
# scripts/salad.sh — ssh/scp convenience for the salad training node.
#
# SSH_HOST / SSH_PORT come from .env (set when the instance comes up).
# The container's entrypoint installed SSH_PUBKEY for key-only root
# login, so no passwords anywhere.
#
#   scripts/salad.sh ssh 'nvidia-smi | head -12'      # run on the node
#   scripts/salad.sh ssh                              # interactive shell
#   scripts/salad.sh put /tmp/chess.tgz /workspace/   # scp TO the node
set -e
cd "$(dirname "$0")/.."
set -a; source .env; set +a
: "${SSH_HOST:?SSH_HOST not set in .env}"
: "${SSH_PORT:?SSH_PORT not set in .env}"

# Salad nodes are disposable: a replaced container at the same address
# presents a NEW host key, which strict checking rejects. Purge the old
# entry before connecting (TOFU per connection is the right trust model
# for cattle, not pets).
ssh-keygen -R "[${SSH_HOST}]:${SSH_PORT}" >/dev/null 2>&1 || true

OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)

case "${1:-}" in
    ssh) shift
         if [ $# -eq 0 ]; then
             exec ssh -p "$SSH_PORT" "${OPTS[@]}" "root@${SSH_HOST}"
         else
             exec ssh -p "$SSH_PORT" "${OPTS[@]}" "root@${SSH_HOST}" "$@"
         fi ;;
    put) shift
         [ $# -ge 2 ] || { echo "usage: salad.sh put LOCAL REMOTE" >&2; exit 2; }
         # A bare REMOTE path is relative to the node's root — prepend the
         # host spec (scp with two bare paths silently degrades to a LOCAL
         # copy, which is how the 2026-09-26 ship failed confusingly).
         REMOTE=$2
         case "$REMOTE" in
             *:*) ;;                                  # already [user@]host:path
             *)  REMOTE="root@${SSH_HOST}:${REMOTE}" ;;
         esac
         exec scp -P "$SSH_PORT" "${OPTS[@]}" "$1" "$REMOTE" ;;
    *)   echo "usage: salad.sh ssh [cmd...] | salad.sh put LOCAL REMOTE" >&2
         exit 2 ;;
esac
