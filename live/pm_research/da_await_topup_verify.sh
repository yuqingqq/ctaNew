#!/bin/bash
# DA: wait for BE's top-up dataset, then run the pre-registered verifier.
#
# RUN AS A SYSTEMD USER UNIT, not as a session background job. Two harness
# background watches were killed mid-poll today; a watch that dies silently
# while the thing it guards is still coming is the fail-open shape this
# programme keeps paying for. R-147(2) already ruled the remedy in another
# context -- the PM collectors survived the crash because they were systemd
# units and the nohup ones did not.
#
#   systemd-run --user --unit=da-topup-verify --slice=research.slice \
#     -p OOMScoreAdjust=1000 \
#     /home/yuqing/ctaNew/live/pm_research/da_await_topup_verify.sh
#   systemctl --user status da-topup-verify
#   cat /home/yuqing/ctaNew/data/pm_5min/derived/.da_topup_verify.out
set -u
F=/home/yuqing/ctaNew/data/pm_5min/derived/harmful_exposure_rows_v3_topup.json
LOG=/home/yuqing/ctaNew/data/pm_5min/derived/.da_topup_verify.out
DEADLINE=$(( $(date +%s) + 21600 ))     # 6h cap
# NEWER-THAN guard. On the first arming the artifact did not exist, so mere
# presence was a safe trigger. It is not safe any more: a REJECTED build can
# still be sitting at this path awaiting rename, and firing on it would render
# a verdict on a superseded artifact -- which is exactly the error avoided by
# luck at 15:44 today, when the file on disk had already been replaced by a
# rebuild while my verdict named the 1.95GB original. BASELINE is passed in by
# the arming command; only an artifact strictly newer than it is verified.
BASELINE=${BASELINE:-0}
# APPEND, never truncate. The first version did `: > "$LOG"`, so re-arming the
# watcher ERASED the rejection verdict the previous run had written -- and it
# did so minutes after I filed Q-DA-78(2) against BE for overwriting a rejected
# artifact in place. A log that clears itself on re-arm destroys exactly the
# evidence it exists to hold.
{ echo; echo "======== re-armed $(date -u +%FT%TZ) ========"; } >> "$LOG"
echo "armed $(date -u +%FT%TZ) as a systemd unit; awaiting $F" >> "$LOG"
echo "will IGNORE any artifact with mtime <= $BASELINE ($(date -u -d @$BASELINE +%FT%TZ 2>/dev/null))" >> "$LOG"
while : ; do
  if [ -f "$F" ]; then
    MT=$(stat -c%Y "$F" 2>/dev/null || echo 0)
    [ "$MT" -gt "$BASELINE" ] && break
  fi
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT: no artifact newer than baseline" >> "$LOG"; exit 2; }
  sleep 20
done
echo "appeared $(date -u +%FT%TZ); awaiting size stability" >> "$LOG"
# BE's builder writes the final path directly (Q-DA-77(6)), so presence is not
# completeness. Require three consecutive equal, non-zero sizes.
prev=-1; stable=0
while [ "$stable" -lt 3 ]; do
  sleep 15
  cur=$(stat -c%s "$F" 2>/dev/null || echo -1)
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ]; then stable=$((stable+1)); else stable=0; fi
  prev=$cur
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT awaiting stability" >> "$LOG"; exit 2; }
done
echo "stable at $prev bytes $(date -u +%FT%TZ); verifying" >> "$LOG"
cd /home/yuqing/ctaNew/live/pm_research || exit 3
/home/yuqing/pricer-sol/venv/bin/python3 da_topup_population_verify.py verify >> "$LOG" 2>&1
rc=$?
echo "VERIFIER_EXIT=$rc  ($(date -u +%FT%TZ))" >> "$LOG"
exit $rc
