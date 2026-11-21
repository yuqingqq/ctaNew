#!/bin/bash
# PRESSURE-LEAD-SENSITIVITY-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order:
#   manifest | pilot | project | rehearse | rest | selection | measure | checks
# run_light, absolute paths, one day per development slot; the rest runs only if the write-once projection allows it. The rehearsal copies the
# pilot day outputs into <OUT>/rehearsal/days and runs selection, measurement and the checker's rehearse mode there (code paths, not acceptance).
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_pressure_lag_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
O=$R/data/predict_fun/research/ladder_pressure_lag_v1_20261121
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude gl_manifest $P/pf_ladder_pressure_lag.py manifest ;;
  pilot|rest)
    st=$1; [ $st = rest ] && { python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/pilot_projection.json'))['run_rest'] is True else 1)" || { echo "rest REFUSED by the pilot projection" >> $J; exit 1; }; }
    cmd=$([ $st = pilot ] && echo pilot || echo run)
    ( rl claude gl_${st}_20261024 $P/pf_ladder_pressure_lag.py $cmd 20261024 ) & p1=$!
    ( rl claude2 gl_${st}_20261025 $P/pf_ladder_pressure_lag.py $cmd 20261025 ) & p2=$!
    ( rl claude3 gl_${st}_20261026 $P/pf_ladder_pressure_lag.py $cmd 20261026 ) & p3=$!
    ( rl claude4 gl_${st}_20261027 $P/pf_ladder_pressure_lag.py $cmd 20261027 ) & p4=$!
    wait $p1 $p2 $p3 $p4 ;;
  project) rl claude gl_projection $D/tools/pilot_projection_lag.py ;;
  rehearse)
    [ -e $O/rehearsal ] && { echo "rehearse REFUSED: $O/rehearsal exists" >> $J; exit 1; }
    mkdir -p $O/rehearsal/days && cp $O/pilot/2026102?.json $O/pilot/2026102?.jsonl.gz $O/pilot/2026102?_arrays.npz $O/rehearsal/days/
    rl claude gl_rehearse_selection $P/pf_ladder_pressure_lag.py selection $O/rehearsal
    rl claude gl_rehearse_measure $P/pf_ladder_pressure_lag.py measure $O/rehearsal
    rl claude gl_rehearse_checks $D/lag_checks.py rehearse $O/rehearsal ;;
  selection) rl claude gl_selection $P/pf_ladder_pressure_lag.py selection ;;
  measure) rl claude gl_measure $P/pf_ladder_pressure_lag.py measure ;;
  checks) rl claude gl_checks $D/lag_checks.py ;;
  *) echo "usage: launch_lag.sh manifest | pilot | project | rehearse | rest | selection | measure | checks"; exit 2 ;;
esac
