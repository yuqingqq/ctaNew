#!/bin/bash
# BINANCE-PRESSURE-GUARD-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order:
#   manifest | pilot | project | rehearse | rest | selection | measure | checks
# run_light, absolute paths, one day per development slot; the rest runs only if the write-once projection allows it. The rehearsal copies the
# pilot day outputs into <OUT>/rehearsal/days and runs selection, measurement and the checker's rehearse mode there (code paths, not acceptance).
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_pressure_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
O=$R/data/predict_fun/research/ladder_pressure_v1_20261121
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude gp_manifest $P/pf_ladder_pressure.py manifest ;;
  pilot|rest)
    st=$1; [ $st = rest ] && { python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/pilot_projection.json'))['run_rest'] is True else 1)" || { echo "rest REFUSED by the pilot projection" >> $J; exit 1; }; }
    cmd=$([ $st = pilot ] && echo pilot || echo run)
    ( rl claude gp_${st}_20261024 $P/pf_ladder_pressure.py $cmd 20261024 ) & p1=$!
    ( rl claude2 gp_${st}_20261025 $P/pf_ladder_pressure.py $cmd 20261025 ) & p2=$!
    ( rl claude3 gp_${st}_20261026 $P/pf_ladder_pressure.py $cmd 20261026 ) & p3=$!
    ( rl claude4 gp_${st}_20261027 $P/pf_ladder_pressure.py $cmd 20261027 ) & p4=$!
    wait $p1 $p2 $p3 $p4 ;;
  project) rl claude gp_projection $D/tools/pilot_projection_pressure.py ;;
  rehearse)
    [ -e $O/rehearsal ] && { echo "rehearse REFUSED: $O/rehearsal exists" >> $J; exit 1; }
    mkdir -p $O/rehearsal/days && cp $O/pilot/2026102?.json $O/pilot/2026102?.jsonl.gz $O/pilot/2026102?_arrays.npz $O/rehearsal/days/
    rl claude gp_rehearse_selection $P/pf_ladder_pressure.py selection $O/rehearsal
    rl claude gp_rehearse_measure $P/pf_ladder_pressure.py measure $O/rehearsal
    rl claude gp_rehearse_checks $D/pressure_checks.py rehearse $O/rehearsal ;;
  selection) rl claude gp_selection $P/pf_ladder_pressure.py selection ;;
  measure) rl claude gp_measure $P/pf_ladder_pressure.py measure ;;
  checks) rl claude gp_checks $D/pressure_checks.py ;;
  *) echo "usage: launch_pressure.sh manifest | pilot | project | rehearse | rest | selection | measure | checks"; exit 2 ;;
esac
