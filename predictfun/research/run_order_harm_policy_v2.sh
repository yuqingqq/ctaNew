#!/usr/bin/env bash
# ORDER-HARM POLICY v2 driver (2026-11-14, session ce68ebe0): selftest, input parity and the policy replay on ORDER-HARM v3's five chronological
# test days, report. Needs ORDER-HARM v3's outputs (run_order_harm_v3.sh) first. claude slot, waits for a free slot.
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/order_harm_policy_v2_client50_20261114; mkdir -p $OUT/logs
TESTS="20261027 20261031 20261101 20261102 20261103"
for d in $TESTS; do [ -f data/predict_fun/research/order_harm_v3_client50_20261114/fit_$d.json ] || { echo "REFUSED: ORDER-HARM v3 (client50) fold $d missing"; exit 1; }; done
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1
  done; sleep 3; }
job ohp2_client50_controls python3 -m unittest predictfun.research.test_order_harm_client50
job ohp2_selftest python3 predictfun/research/pf_order_harm_policy_v2.py selftest
for d in $TESTS; do job ohp2_parity_$d python3 predictfun/research/pf_order_harm_policy_v2.py parity $d; done
for d in $TESTS; do job ohp2_eval_$d python3 predictfun/research/pf_order_harm_policy_v2.py eval $d; done
job ohp2_report python3 predictfun/research/pf_order_harm_policy_v2.py report
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
