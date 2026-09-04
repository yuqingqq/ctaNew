#!/usr/bin/env bash
# PREFLIGHT -> SCORE, one command, no thinking required at the boundary.
#
# WHAT IT DOES IF THE MASK HAS NOT LANDED: it REFUSES CLEANLY AND RETRIES on a
# fixed interval, and it never scores. A governed day without its mask is not
# a day that can be scored badly -- `de_admissible_windows` refuses it at the
# population gate -- so the only question is whether we discover that in one
# second or after 28 minutes of replay. The preflight answers in one second,
# so retrying is cheap and waiting costs nothing but time.
#
# IT WILL NOT WAIT FOREVER. After --deadline it exits 3 with the blockers
# named, because a run that waits silently through the night is the failure
# this whole class of work is about: nothing ran and nothing said so.
set -u
DAY="${1:-}"
OUTDIR="${2:-}"
INTERVAL="${SCORE_RETRY_INTERVAL_S:-600}"
DEADLINE="${SCORE_DEADLINE_S:-43200}"        # 12 h, then give up LOUDLY
MEM="${SCORE_MEMORY_MAX:-8G}"                # race days; 2G OOM-killed two
PY=/home/yuqing/pricer-sol/venv/bin/python3
WT=/home/yuqing/ctaNew-wt-be

if [ -z "$DAY" ] || [ -z "$OUTDIR" ]; then
  echo "usage: be_score_forward_day.sh <YYYYMMDD> <outdir>" >&2; exit 2
fi
if [ -e "$OUTDIR" ] && [ -n "$(ls -A "$OUTDIR" 2>/dev/null)" ]; then
  echo "REFUSED: $OUTDIR is not empty. A scored day must land in a NEW" \
       "outdir; reusing one is how a half-written feed gets read as a day." >&2
  exit 2
fi

cd "$WT" || exit 3
started=$(date -u +%s)
attempt=0
while :; do
  attempt=$((attempt + 1))
  now=$(date -u +%s)
  echo "[$(date -u +%H:%M:%SZ)] preflight attempt $attempt for $DAY"
  if "$PY" -m live.pm_research.be_forward_preflight --day "$DAY" \
        > "/tmp/preflight_$DAY.json" 2>&1; then
    echo "[$(date -u +%H:%M:%SZ)] GO — scoring $DAY at $MEM"
    mkdir -p "$OUTDIR"
    systemd-run --user --scope --slice=research.slice -p MemoryMax="$MEM" -q \
      "$PY" -m live.pm_research.be_forward_day \
        --forward-day "$DAY" --outdir "$OUTDIR"
    rc=$?
    echo "[$(date -u +%H:%M:%SZ)] scoring rc=$rc"
    exit $rc
  fi
  # NO-GO: say WHICH blocker, every time, so a watcher sees it change.
  "$PY" -c 'import json,sys
try: d=json.load(open(sys.argv[1]))
except Exception: print("  (preflight produced no JSON)"); raise SystemExit
for b in d.get("blockers", []): print("  NO-GO:", b[:150])' \
    "/tmp/preflight_$DAY.json"
  if [ $((now - started)) -ge "$DEADLINE" ]; then
    echo "[$(date -u +%H:%M:%SZ)] GIVING UP after ${DEADLINE}s with blockers" \
         "still present. NOTHING WAS SCORED. This exits 3 rather than" \
         "waiting silently, because a run that waits through the night and" \
         "reports nothing is indistinguishable from one that never started." >&2
    exit 3
  fi
  sleep "$INTERVAL"
done
