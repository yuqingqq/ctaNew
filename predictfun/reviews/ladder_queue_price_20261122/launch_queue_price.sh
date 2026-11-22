#!/bin/bash
# QUEUE-PRICE-STATE-AUDIT-v1 launcher (session aea5bfb9, 2026-11-22; declaration.md). Stages, in order:
#   manifest | selftest | project | run | readings | checks
# run_light, absolute paths, one job at a time; saved data only (no tape load, no replay). run refuses unless the write-once projection fits.
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_queue_price_20261122; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude qp_manifest $P/pf_ladder_queue_price.py manifest ;;
  selftest) rl claude qp_selftest_frozen $P/pf_ladder_queue_price.py selftest ;;
  project) rl claude qp_projection $D/tools/projection_qp.py ;;
  run)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/projection.json'))['run_full'] is True else 1)" || { echo "run REFUSED by the projection" >> $J; exit 1; }
    rl claude qp_run $P/pf_ladder_queue_price.py run ;;
  readings) rl claude qp_readings $D/tools/read_queue_price.py ;;
  checks) rl claude qp_checks $D/qp_checks.py ;;
  *) echo "usage: launch_queue_price.sh manifest | selftest | project | run | readings | checks"; exit 2 ;;
esac
