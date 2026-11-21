#!/bin/bash
# PI0-MOVEMENT-CALIBRATION-AUDIT-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order: pilot | project | rest | measure | checks.
# Jobs only through run_light with absolute paths; partitions 1-11 on four slots after the write-once pilot projection allows them.
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/movement_audit_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
part() { rl $1 mov_part_$(printf %03d $2) $P/pf_movement_audit.py run $2; }
case "${1:-}" in
  pilot) part claude 0 ;;
  project) rl claude mov_pilot_projection $D/tools/pilot_projection_mov.py ;;
  rest)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/pilot_projection.json'))['run_rest'] is True else 1)" || { echo "rest REFUSED by the pilot projection" >> $J; exit 1; }
    ( for k in 1 5 9; do part claude $k; done ) & p1=$!
    ( for k in 2 6 10; do part claude2 $k; done ) & p2=$!
    ( for k in 3 7 11; do part claude3 $k; done ) & p3=$!
    ( for k in 4 8; do part claude4 $k; done ) & p4=$!
    wait $p1 $p2 $p3 $p4 ;;
  measure) rl claude mov_measure $P/pf_movement_audit.py measure ;;
  checks) rl claude mov_checks $D/movement_checks.py ;;
  *) echo "usage: launch_movement.sh pilot | project | rest | measure | checks"; exit 2 ;;
esac
