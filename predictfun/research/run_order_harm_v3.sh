#!/usr/bin/env bash
# ORDER-HARM v3 driver (2026-11-14, session ce68ebe0): selftest, per-day keep-all labels with the option target's fields, chronological
# fitted-Q folds (test days 10-27 .. 11-03) with the PULL-DEPTH refit and calibration, report; claude slot, waits for a free slot.
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/order_harm_v3_client50_20261114; mkdir -p $OUT/logs
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
TESTS="20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1
  done; sleep 3; }
job oh3_client50_controls python3 -m unittest predictfun.research.test_order_harm_client50
job oh3_selftest python3 predictfun/research/pf_order_harm_v3.py selftest
# label days already built by the identical label code are reused (USER: "Existing label files remain reusable"); fits read them via their receipts
for d in $DAYS; do if [ -f $OUT/rows_$d.parquet ] && [ -f $OUT/rows_$d.json ]; then echo "$(date -u +%H:%M:%SZ) reuse rows $d" >> $OUT/logs/driver.log; else job oh3_rows_$d python3 predictfun/research/pf_order_harm_v3.py rows $d; fi; done
for d in $TESTS; do job oh3_fit_$d python3 predictfun/research/pf_order_harm_v3.py fit $d; done
job oh3_report python3 predictfun/research/pf_order_harm_v3.py report
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
