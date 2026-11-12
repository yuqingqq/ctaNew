#!/usr/bin/env bash
# QR2 driver (2026-11-12, session ce68ebe0): self-tests, the 9 replay days, report.
# Every job through run_light.sh in the claude slot (core 9), one at a time; stops at the first failure. A 30 s gap between jobs
# lets the Codex driver start its next stage.
set -u
cd /home/yuqing/ctaNew
OUT=data/predict_fun/research/qr2_20261112; mkdir -p $OUT/logs
R="env PF_SLOT=claude predictfun/research/run_light.sh"; P="python3 predictfun/research/pf_qr2_v3_early.py"
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
job() { local tag=$1; shift; echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
  $R qr2_$tag -- $P "$@" > $OUT/logs/$tag.log 2>&1 || { echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; exit 1; }
  echo "$(date -u +%H:%M:%SZ) done $tag" >> $OUT/logs/driver.log; sleep 30; }
job selftest selftest
for d in $DAYS; do job day_$d day $d; done
job report report
echo "$(date -u +%H:%M:%SZ) ALL DONE" >> $OUT/logs/driver.log
