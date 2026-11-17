#!/usr/bin/env bash
# GAIN / LOSS DECOMPOSITION driver (2026-11-17, session aea5bfb9; USER "Next priority: predict expected loss amount, especially severe losses,
# and test its calibration before changing the controller"): run_gain_loss.sh all -> selftest, label set L on Oct26 / Oct27 (every market,
# training selection rule), fit (Oct23-25), evaluate (set L + the audited 861), report (G1 / G2 / PASS / G3 as declared in pf_gain_loss.py).
set -u
cd /home/yuqing/ctaNew
PHASE="${1:?all}"
OUT=data/predict_fun/research/gain_loss_20261117; mkdir -p $OUT/logs
job() { local tag=$1; shift; local n=0
  while :; do echo "$(date -u +%H:%M:%SZ) start $tag" >> $OUT/logs/driver.log
    if env PF_SLOT=claude predictfun/research/run_light.sh $tag -- "$@" > $OUT/logs/$tag.log 2>&1; then break; fi
    if grep -q -E "^REFUSED: (research slot|[0-9]+ predictfun research processes)" $OUT/logs/$tag.log && [ $n -lt 240 ]; then
      n=$((n+1)); sleep 60; continue; fi
    echo "$(date -u +%H:%M:%SZ) FAILED $tag" >> $OUT/logs/driver.log; return 1
  done; }
case "$PHASE" in
  all)
    job gl_selftest python3 predictfun/research/pf_gain_loss.py selftest || exit 1
    for d in 20261026 20261027; do job gl_label_$d python3 predictfun/research/pf_gain_loss.py label $d || exit 1; done
    job gl_fit python3 predictfun/research/pf_gain_loss.py fit || exit 1
    job gl_evaluate python3 predictfun/research/pf_gain_loss.py evaluate || exit 1
    job gl_report python3 predictfun/research/pf_gain_loss.py report || exit 1;;
  *) echo "unknown phase $PHASE" >&2; exit 1;;
esac
echo "$(date -u +%H:%M:%SZ) GAIN-LOSS $PHASE DONE" >> $OUT/logs/driver.log
