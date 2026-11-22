#!/bin/bash
# FEASIBLE-CANCEL-HEADROOM-v1 launcher (session aea5bfb9, 2026-11-22; declaration.md). Stages, in order:
#   manifest | selftest | timing | project | run | checks | identity
# run_light, absolute paths, PF_SLOT claude; run refuses unless the write-once projection fits (the builder checks it too).
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_cancel_headroom_20261122; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
rl() { env PF_SLOT=claude ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $1 -- python3 "${@:2}" > $L/$1.log 2>&1; echo "$1 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl ch_manifest $P/pf_ladder_cancel_headroom.py manifest ;;
  selftest) rl ch_selftest_frozen $P/pf_ladder_cancel_headroom.py selftest ;;
  timing) rl ch_timing $P/pf_ladder_cancel_headroom.py timing ;;
  project) rl ch_projection $P/pf_ladder_cancel_headroom.py project ;;
  run)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/projection.json'))['fits'] is True else 1)" || { echo "run REFUSED by the projection" >> $J; exit 1; }
    rl ch_run $P/pf_ladder_cancel_headroom.py run ;;
  checks) rl ch_checks $D/ch_checks.py ;;
  identity) rl ch_identity $D/tools/async_identity.py ;;
  *) echo "usage: launch_cancel_headroom.sh manifest | selftest | timing | project | run | checks | identity"; exit 2 ;;
esac
