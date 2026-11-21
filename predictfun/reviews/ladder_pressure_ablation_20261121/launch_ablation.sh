#!/bin/bash
# PRESSURE-ACTION-ATTRIBUTION-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order:
#   manifest | pilot | project | rehearse | rest | measure | checks
# run_light, absolute paths, one date per development slot; the rest runs only if the write-once projection allows it (a date without rest
# members writes DAY_EMPTY and loads nothing). The rehearsal copies the pilot outputs into <OUT>/rehearsal/pilot and runs the measurement and
# the checker's rehearse mode there (code paths, not acceptance).
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_pressure_ablation_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
O=$R/data/predict_fun/research/ladder_pressure_ablation_v1_20261121
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude ga_manifest $P/pf_ladder_pressure_ablation.py manifest ;;
  pilot|rest)
    st=$1; [ $st = rest ] && { python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/pilot_projection.json'))['run_rest'] is True else 1)" || { echo "rest REFUSED by the pilot projection" >> $J; exit 1; }; }
    ( rl claude ga_${st}_20261024 $P/pf_ladder_pressure_ablation.py $st 20261024 ) & p1=$!
    ( rl claude2 ga_${st}_20261025 $P/pf_ladder_pressure_ablation.py $st 20261025 ) & p2=$!
    ( rl claude3 ga_${st}_20261026 $P/pf_ladder_pressure_ablation.py $st 20261026 ) & p3=$!
    ( rl claude4 ga_${st}_20261027 $P/pf_ladder_pressure_ablation.py $st 20261027 ) & p4=$!
    wait $p1 $p2 $p3 $p4 ;;
  project) rl claude ga_projection $D/tools/pilot_projection_ablation.py ;;
  rehearse)
    [ -e $O/rehearsal ] && { echo "rehearse REFUSED: $O/rehearsal exists" >> $J; exit 1; }
    mkdir -p $O/rehearsal/pilot && cp $O/pilot/2026102?.json $O/rehearsal/pilot/ && cp $O/pilot/2026102?.jsonl.gz $O/pilot/2026102?_arrays.npz $O/rehearsal/pilot/ 2>/dev/null
    rl claude ga_rehearse_measure $P/pf_ladder_pressure_ablation.py measure $O/rehearsal pilot
    rl claude ga_rehearse_checks $D/ablation_checks.py rehearse $O/rehearsal ;;
  measure) rl claude ga_measure $P/pf_ladder_pressure_ablation.py measure ;;
  checks) rl claude ga_checks $D/ablation_checks.py ;;
  *) echo "usage: launch_ablation.sh manifest | pilot | project | rehearse | rest | measure | checks"; exit 2 ;;
esac
