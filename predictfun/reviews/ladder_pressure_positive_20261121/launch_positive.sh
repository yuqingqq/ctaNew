#!/bin/bash
# POSITIVE-PRESSURE-GUARD-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order:
#   manifest | pilot | project | rehearse | rest | screen | measure | checks
# run_light, absolute paths; the pilot is all seven Oct 24 markets (reused); the rest (Oct 25-27, one date per slot) runs only if the write-once
# whole-round projection allows it. The rehearsal copies the pilot outputs into <OUT>/rehearsal/pilot and runs the screen, the measurement and
# the checker's rehearse mode there (code paths, not acceptance).
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_pressure_positive_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
O=$R/data/predict_fun/research/ladder_pressure_positive_v1_20261121
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude gp2_manifest $P/pf_ladder_pressure_positive.py manifest ;;
  pilot) rl claude gp2_pilot $P/pf_ladder_pressure_positive.py pilot ;;
  project) rl claude gp2_projection $D/tools/pilot_projection_positive.py ;;
  rehearse)
    [ -e $O/rehearsal ] && { echo "rehearse REFUSED: $O/rehearsal exists" >> $J; exit 1; }
    mkdir -p $O/rehearsal/pilot && cp $O/pilot/20261024.json $O/pilot/20261024.jsonl.gz $O/pilot/20261024_arrays.npz $O/rehearsal/pilot/
    rl claude gp2_rehearse_screen $P/pf_ladder_pressure_positive.py screen $O/rehearsal
    rl claude gp2_rehearse_measure $P/pf_ladder_pressure_positive.py measure $O/rehearsal pilot
    rl claude gp2_rehearse_checks $D/positive_checks.py rehearse ;;
  rest)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/pilot_projection.json'))['run_rest'] is True else 1)" || { echo "rest REFUSED by the pilot projection" >> $J; exit 1; }
    ( rl claude gp2_rest_20261025 $P/pf_ladder_pressure_positive.py rest 20261025 ) & p1=$!
    ( rl claude2 gp2_rest_20261026 $P/pf_ladder_pressure_positive.py rest 20261026 ) & p2=$!
    ( rl claude3 gp2_rest_20261027 $P/pf_ladder_pressure_positive.py rest 20261027 ) & p3=$!
    wait $p1 $p2 $p3 ;;
  screen) rl claude gp2_screen $P/pf_ladder_pressure_positive.py screen ;;
  measure) rl claude gp2_measure $P/pf_ladder_pressure_positive.py measure ;;
  checks) rl claude gp2_checks $D/positive_checks.py ;;
  *) echo "usage: launch_positive.sh manifest | pilot | project | rehearse | rest | screen | measure | checks"; exit 2 ;;
esac
