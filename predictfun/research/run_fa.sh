#!/usr/bin/env bash
# FA driver (2026-11-12, session ce68ebe0): self-tests, the 9 replay days, report. Every job through run_light.sh in the claude slot,
# one at a time; waits for a free slot (retry 60 s, up to 4 h) when the slot / cap is busy; any other failure stops it.
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/fa_20261112; mkdir -p $OUT/logs
R="env PF_SLOT=claude predictfun/research/run_light.sh"; P="python3 predictfun/research/pf_false_alarm.py"
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if $R fa_$tag -- $P "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); echo "$(date -u +%H:%M:%SZ) slot busy; retry in 60 s" >> $OUT/logs/driver.log; sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1
  done; echo "$(date -u +%H:%M:%SZ) done $tag" >> $OUT/logs/driver.log; sleep 3; }
job selftest selftest
for d in $DAYS; do job day_$d day $d; done
job report report
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
