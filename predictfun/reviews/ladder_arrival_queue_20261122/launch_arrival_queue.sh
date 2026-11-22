#!/bin/bash
# ARRIVAL-QUEUE-DECOMPOSITION-v1 launcher (session aea5bfb9, 2026-11-22; declaration.md). Stages, in order:
#   manifest | selftest | run | checks | readings
# run_light, absolute paths, one job at a time; saved data only (no tape load, no replay).
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_arrival_queue_20261122; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
SYN=$R/data/predict_fun/research/r2_teacher_selftest_world
rl() { env PF_SLOT=$1 PF_TEACHER_SYN=$SYN ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude aq_manifest $P/pf_ladder_arrival_queue.py manifest ;;
  selftest) rl claude aq_selftest_frozen $P/pf_ladder_arrival_queue.py selftest ;;
  run) rl claude aq_run $P/pf_ladder_arrival_queue.py run ;;
  checks) rl claude aq_checks $D/aq_checks.py ;;
  readings) rl claude aq_readings $D/tools/read_arrival_queue.py ;;
  *) echo "usage: launch_arrival_queue.sh manifest | selftest | run | checks | readings"; exit 2 ;;
esac
