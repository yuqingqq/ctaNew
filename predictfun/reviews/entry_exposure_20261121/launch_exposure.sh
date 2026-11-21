#!/bin/bash
# ENTRY-EXPOSURE-v1 (review 3d8bf5ad): jobs via run_light (absolute paths), one process per slot. Its own launcher; never edited while it runs.
#   pilot | rest | summarize | checks
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; RC=$R/predictfun/reviews/entry_exposure_20261121; L=$RC/logs; mkdir -p $L; J=$L/jobs_rc.txt
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
part() { rl $1 exposure_part_$(printf %03d $2) $P/pf_entry_exposure.py run $2; }
case "$1" in
  pilot) part claude 0 ;;
  rest) ( for k in 1 5 9; do part claude $k; done ) & p1=$!
        ( for k in 2 6 10; do part claude2 $k; done ) & p2=$!
        ( for k in 3 7 11; do part claude3 $k; done ) & p3=$!
        ( for k in 4 8; do part claude4 $k; done ) & p4=$!
        wait $p1 $p2 $p3 $p4 ;;
  summarize) rl claude exposure_summarize $P/pf_entry_exposure.py summarize ;;
  checks) rl claude exposure_checks $RC/exposure_checks.py ;;
esac
