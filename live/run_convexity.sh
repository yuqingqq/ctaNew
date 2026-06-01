#!/usr/bin/env bash
# convexity_portable paper-bot tmux supervisor.
#
# Loops every 4h:
#   1. refresh klines + rebuild panel
#   2. run --cycle (process new cycles since last_open_time, append to logs)
#   3. sleep until next 4h boundary
#
# Launch:   tmux new -d -s convexity 'bash /home/yuqing/ctaNew/live/run_convexity.sh'
# Attach:   tmux attach -t convexity
# Detach:   Ctrl-b d
# Logs:     /home/yuqing/ctaNew/live/state/convexity/run.log
#
set -uo pipefail   # NOT -e: a single cycle failure shouldn't kill the loop
ROOT=/home/yuqing/ctaNew
LOG=$ROOT/live/state/convexity/run.log
PY=python3
export PYTHONPATH=$ROOT
cd $ROOT
mkdir -p $(dirname $LOG)

log(){ echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a $LOG; }

log "== convexity paper-bot supervisor started =="
while true; do
  log "-- cycle start --"
  log "refresh panel..."
  $PY -m live.refresh_convexity_panel --days-back 14 >> $LOG 2>&1 \
    && log "refresh OK" || log "refresh FAILED (continuing — bot may run on stale panel)"

  log "run --cycle..."
  $PY -m live.convexity_paper_bot --cycle >> $LOG 2>&1 \
    && log "cycle OK" || log "cycle FAILED"

  # sleep until next 4h boundary (00, 04, 08, 12, 16, 20 UTC)
  NOW=$(date -u '+%s')
  NEXT_4H=$(python3 -c "
import datetime as dt
now=dt.datetime.now(dt.timezone.utc); hr=(now.hour//4 + 1)*4
nxt=now.replace(hour=hr%24, minute=0, second=0, microsecond=0)
if hr>=24: nxt+=dt.timedelta(days=1)
print(int(nxt.timestamp()))")
  SLEEP=$(( NEXT_4H - NOW + 60 ))   # +60s grace
  if [ $SLEEP -lt 60 ]; then SLEEP=60; fi
  log "sleeping ${SLEEP}s until next 4h boundary"
  sleep $SLEEP
done
