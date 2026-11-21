#!/bin/bash
# PI0-MOVEMENT-SHRINK-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order: fit | score | checks
# (the manifest is built before the freeze; the fit is committed before score runs). Jobs only through run_light with absolute paths.
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/movement_shrink_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  fit) rl claude shr_fit $P/pf_movement_shrink.py fit ;;
  score) rl claude shr_score $P/pf_movement_shrink.py score ;;
  checks) rl claude shr_checks $D/shrink_checks.py ;;
  *) echo "usage: launch_shrink.sh fit | score | checks"; exit 2 ;;
esac
