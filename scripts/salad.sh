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
         exec scp -P "$SSH_PORT" "${OPTS[@]}" "$@" ;;
    *)   echo "usage: salad.sh ssh [cmd...] | salad.sh put LOCAL REMOTE" >&2
         exit 2 ;;
esac
