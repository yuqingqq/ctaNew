#!/usr/bin/env bash
# Convexity v1 real-time FALLBACK watchdog.
#
# The COLLECTOR is the primary trigger: the instant a 4h bar closes it flushes and spawns
# convexity_v1_cycle_once.sh directly (push, ~few-second latency). This watchdog is only a SAFETY NET — if
# the collector ever misses a boundary (restarted exactly at the close, crash, etc.), it runs cycle_once a
# few minutes later. cycle_once is flock-guarded + idempotent, so a redundant run is a no-op (it sees the
# boundary already booked and exits). So running both is always safe.
# Launch:  tmux new -d -s cvx1fb 'bash /home/yuqing/ctaNew/live/run_convexity_v1_realtime.sh'
set -uo pipefail
ROOT=/home/yuqing/ctaNew; export PYTHONPATH=$ROOT; cd "$ROOT"; PY=$ROOT/.venv/bin/python
LOG=$ROOT/live/state/convexity_v1/realtime.log
log(){ echo "[$(date -u '+%F %T')] [fallback] $*" | tee -a "$LOG"; }

log "== convexity v1 FALLBACK watchdog (collector push is the primary trigger) =="
while true; do
  $PY live/wait_bar_ready.py >> "$LOG" 2>&1 || { log "wait err — sleep 600"; sleep 600; continue; }
  sleep 180                                                   # give the collector's direct trigger time first
  bash live/convexity_v1_cycle_once.sh fallback || log "cycle_once (fallback) returned nonzero"
done
