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
: > "$LOG"
echo "armed $(date -u +%FT%TZ) as a systemd unit; awaiting $F" >> "$LOG"
while [ ! -f "$F" ]; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT: artifact never appeared" >> "$LOG"; exit 2; }
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
