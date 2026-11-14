#!/usr/bin/env bash
# ORDER-HARM WHY driver (2026-11-14, session ce68ebe0; POST HOC): why POLICY v2's keep-vs-cancel model loses, against the top makers, and
# what the model saw at its losing keeps. Needs POLICY v2's outputs (run_order_harm_policy_v2.sh). claude slot, waits for a free slot.
set -u
cd /home/yuqing/ctaNew
OUT=$(python3 -c "import sys; sys.path.insert(0, 'predictfun/research'); import pf_order_harm_why as M; print(M.OUTD)") || exit 1; mkdir -p $OUT/logs   # the folder comes from the script (no drift)
TESTS="20261027 20261031 20261101 20261102 20261103"
P2OUT=$(python3 -c "import sys; sys.path.insert(0, 'predictfun/research'); import pf_order_harm_policy_v2 as P; print(P.OUTD)") || exit 1
for d in $TESTS; do [ -f $P2OUT/fills_$d.parquet ] || { echo "REFUSED: POLICY v2 fills $d missing in $P2OUT"; exit 1; }; done
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1
  done; sleep 3; }
job ohw_selftest python3 predictfun/research/pf_order_harm_why.py selftest
for d in $TESTS; do job ohw_day_$d python3 predictfun/research/pf_order_harm_why.py day $d; done
job ohw_report python3 predictfun/research/pf_order_harm_why.py report
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
