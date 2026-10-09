#!/bin/bash
# salad_cloud.sh — ssh/scp wrapper for the .cloud.env instance pool.
#
#   .cloud.env lines:  CLOUD_<i>=<host>:<port>     (i = 1..N)
#
#   scripts/salad_cloud.sh <i|all> ssh 'command'
#   scripts/salad_cloud.sh <i|all> put <local> <remote>
set -e
cd "$(dirname "$0")/.."
[ -f .cloud.env ] || { echo ".cloud.env missing" >&2; exit 1; }
set -a; source .cloud.env; set +a

MODE=$1; shift
if [ "$MODE" = "all" ]; then
    INSTANCES=$(grep -E '^CLOUD_[0-9]+' .cloud.env | cut -d= -f1 | sort -u)
else
    INSTANCES="CLOUD_$MODE"
fi
[ -n "$INSTANCES" ] || { echo "no CLOUD_ entries in .cloud.env" >&2; exit 1; }

run_one() {
    VAR=$1; shift
    ADDR=${!VAR:?missing $VAR in .cloud.env}
    HOST=${ADDR%%:*}
    PORT=${ADDR##*:}
    ssh-keygen -R "[$HOST]:$PORT" >/dev/null 2>&1 || true
    OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
    case "$MODE" in
        ssh) ssh -p "$PORT" "${OPTS[@]}" "root@$HOST" bash -c "\"$*\"" ;;
        put) scp -P "$PORT" "${OPTS[@]}" "$1" "root@$HOST:$2" ;;
        *) echo "unknown mode $MODE" >&2; exit 2 ;;
    esac
}

for VAR in $INSTANCES; do
    echo "== $VAR =="
    run_one "$VAR" "$@"
done
