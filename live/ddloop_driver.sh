#!/usr/bin/env bash
# Process a queue of dd-loop experiments N-way parallel. Crash-safe + idempotent (ddloop_run skips done tags).
# queue file: one experiment per line "tag<TAB>ENV=val ENV2=val2"  ('#' lines and blanks ignored)
# usage: ddloop_driver.sh <queue_file> [parallelism=4]
set -uo pipefail
ROOT=/home/yuqing/ctaNew; cd "$ROOT"
Q=$1; N=${2:-4}
log(){ echo "[$(date -u '+%F %T')] $*"; }
log "driver start: queue=$Q parallelism=$N"
running=0
while IFS=$'\t' read -r tag env || [ -n "$tag" ]; do
  case "$tag" in ''|\#*) continue;; esac
  # throttle
  while [ "$(jobs -rp | wc -l)" -ge "$N" ]; do sleep 2; done
  ( bash live/ddloop_run.sh "$tag" $env ) &
done < "$Q"
wait
log "driver done: $Q"
echo "QUEUEDONE $Q"
