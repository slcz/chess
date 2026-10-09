#!/bin/bash
# salad_cloud.sh — ssh/scp wrapper for the .cloud.env instance pool.
#
#   .cloud.env lines:  CLOUD_<i>=<host>:<port>     (i = 1..N)
#
#   scripts/salad_cloud.sh <i> ssh 'command'
#   scripts/salad_cloud.sh <i> put <local> <remote>
#   scripts/salad_cloud.sh all  ssh 'command'    # every instance, sequential
set -e
cd "$(dirname "$0")/.."
[ -f .cloud.env ] || { echo ".cloud.env missing" >&2; exit 1; }
set -a; source .cloud.env; set +a

IDX=$1; shift
if [ "$IDX" = "all" ]; then
    INSTANCES=$(grep -E '^CLOUD_[0-9]+' .cloud.env | cut -d= -f1 | sort -u)
else
    INSTANCES="CLOUD_$IDX"
fi
[ -n "$INSTANCES" ] || { echo "no CLOUD_ entries in .cloud.env" >&2; exit 1; }

for VAR in $INSTANCES; do
    ADDR=${!VAR:?missing $VAR in .cloud.env}
    HOST=${ADDR%%:*}
    PORT=${ADDR##*:}
    ssh-keygen -R "[$HOST]:$PORT" >/dev/null 2>&1 || true
    OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
    case "$1" in
        ssh) shift; ssh -p "$PORT" "${OPTS[@]}" "root@$HOST" bash -c "\"$*\"" ;;
        put) shift
             REMOTE=$1
             case "$REMOTE" in
                 *:*) ;;
                 *) REMOTE="root@$HOST:$REMOTE" ;;
             esac
             scp -P "$PORT" "${OPTS[@]}" "$REMOTE" >/dev/null 2>&1 || true
             scp -P "$PORT" "${OPTS[@]}" "$1" "root@$HOST:$2"
             ;;
        *) echo "usage: salad_cloud.sh <i|all> <ssh|put> ..." >&2; exit 2 ;;
    esac
done
