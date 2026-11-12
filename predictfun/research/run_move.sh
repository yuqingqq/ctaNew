#!/usr/bin/env bash
# MOVE driver (2026-11-12, session ce68ebe0): per-day builds of pf_move_info.py (public tape, 9 consumed replay days), then the report.
# Every step through run_light.sh in the claude slot, one at a time; waits for a free slot (up to 4 h per step); stops at the first
# other failure. Refuses to start over an existing day output.
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/move_20261112; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
for d in $DAYS; do [ -e $OUT/day_$d.npz ] && { echo "REFUSED: day_$d.npz exists" | tee -a $OUT/logs/driver.log; exit 1; }; done
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); echo "$(date -u +%H:%M:%SZ) slot busy ($(head -1 $OUT/logs/$tag.log)); retry in 60 s" >> $OUT/logs/driver.log; sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1
  done; sleep 3; }
for d in $DAYS; do job move_day_$d python3 predictfun/research/pf_move_info.py day $d; done
job move_report python3 predictfun/research/pf_move_info.py report
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
