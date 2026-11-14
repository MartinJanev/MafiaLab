#!/bin/bash
# Copy MafiaLab between the laptop and the FINKI GPGPU cluster. Run on the laptop, from anywhere.
#
#   bash cluster/sync.sh push        committed code (git HEAD) -> ~/ABS/MafiaLab
#   bash cluster/sync.sh push-data   data/games, data/experiments, data/cache -> the cluster
#   bash cluster/sync.sh pull        the same data folders and the job logs -> the laptop
#
# pull overwrites the laptop's copies (the reply cache too), so don't play games on the laptop
# between push-data and pull. Set CLUSTER to change the login (default martin.janev@10.10.65.6).
set -euo pipefail

HOST=${CLUSTER:-martin.janev@10.10.65.6}
REMOTE=ABS/MafiaLab
DATA=(data/games data/experiments data/cache)
cd "$(dirname "$0")/.."

case ${1:-} in
  push)
    git diff --quiet HEAD || echo "Uncommitted changes are not sent, only the last commit."
    # code folders are replaced so files deleted on the laptop go away there too; .venv stays
    git archive --format=tar HEAD | ssh "$HOST" "mkdir -p $REMOTE && cd $REMOTE \
      && rm -rf src prompts configs tests cluster && tar -x && echo $(git rev-parse --short HEAD) > COMMIT"
    echo "Pushed $(git rev-parse --short HEAD) to $HOST:~/$REMOTE"
    ;;
  push-data)
    present=()
    for d in "${DATA[@]}"; do [[ -e $d ]] && present+=("$d"); done
    tar -c "${present[@]}" | ssh "$HOST" "mkdir -p $REMOTE && tar -x -C $REMOTE"
    echo "Pushed ${present[*]}"
    ;;
  pull)
    ssh "$HOST" "cd $REMOTE && tar -c \$(ls -d ${DATA[*]} 2>/dev/null)" | tar -x
    mkdir -p data/cluster-logs
    ssh "$HOST" "cd ABS/results && tar -c ." | tar -x -C data/cluster-logs
    echo "Pulled the data folders, and the job logs into data/cluster-logs/"
    ;;
  *)
    echo "usage: bash cluster/sync.sh push | push-data | pull" >&2
    exit 2
    ;;
esac
