#!/bin/bash
# LADDER_REST_INVENTORY_v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order: manifest | pilot | project | rest | measure | checks.
# Jobs only through run_light with absolute paths, the four development slots (one day per slot); the rest runs only if the write-once projection allows it.
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_rest_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude lr_manifest $P/pf_ladder_rest.py manifest ;;
  pilot|rest)
    st=$1; [ $st = rest ] && { python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/pilot_projection.json'))['run_rest'] is True else 1)" || { echo "rest REFUSED by the pilot projection" >> $J; exit 1; }; }
    cmd=$([ $st = pilot ] && echo pilot || echo run)
    ( rl claude lr_${st}_20261024 $P/pf_ladder_rest.py $cmd 20261024 ) & p1=$!
    ( rl claude2 lr_${st}_20261025 $P/pf_ladder_rest.py $cmd 20261025 ) & p2=$!
    ( rl claude3 lr_${st}_20261026 $P/pf_ladder_rest.py $cmd 20261026 ) & p3=$!
    ( rl claude4 lr_${st}_20261027 $P/pf_ladder_rest.py $cmd 20261027 ) & p4=$!
    wait $p1 $p2 $p3 $p4 ;;
  project) rl claude lr_projection $D/tools/pilot_projection_lr.py ;;
  measure) rl claude lr_measure $P/pf_ladder_rest.py measure ;;
  checks) rl claude lr_checks $D/ladder_checks.py ;;
  *) echo "usage: launch_ladder_rest.sh manifest | pilot | project | rest | measure | checks"; exit 2 ;;
esac
