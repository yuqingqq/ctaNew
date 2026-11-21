#!/bin/bash
# PI0-COVERAGE-v1 launcher (session aea5bfb9, 2026-11-21; declaration.md). Stages, in order: manifest | pilot | project | rest | measure | checks.
# Jobs only through run_light with absolute paths, four development slots; the rest runs only if the write-once projection allows it.
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/coverage_20261121; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
rl() { env PF_SLOT=$1 ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
pilot() { rl $1 cov_pilot_$2 $P/pf_coverage.py pilot $2; }
part() { rl $1 cov_part_$(printf %03d $2) $P/pf_coverage.py run $2; }
case "${1:-}" in
  manifest) rl claude cov_manifest $P/pf_coverage.py manifest ;;
  pilot)
    ( pilot claude 20261024; pilot claude 20261102 ) & p1=$!
    ( pilot claude2 20261025; pilot claude2 20261103 ) & p2=$!
    ( pilot claude3 20261031 ) & p3=$!
    ( pilot claude4 20261101 ) & p4=$!
    wait $p1 $p2 $p3 $p4 ;;
  project) rl claude cov_projection $D/tools/pilot_projection_cov.py ;;
  rest)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/pilot_projection.json'))['run_rest'] is True else 1)" || { echo "rest REFUSED by the pilot projection" >> $J; exit 1; }
    N=$(python3 -c "import json; print(len(json.load(open('$D/manifest.json'))['partitions']))")
    ( for k in $(seq 0 4 $((N - 1))); do part claude $k; done ) & p1=$!
    ( for k in $(seq 1 4 $((N - 1))); do part claude2 $k; done ) & p2=$!
    ( for k in $(seq 2 4 $((N - 1))); do part claude3 $k; done ) & p3=$!
    ( for k in $(seq 3 4 $((N - 1))); do part claude4 $k; done ) & p4=$!
    wait $p1 $p2 $p3 $p4 ;;
  measure) rl claude cov_measure $P/pf_coverage.py measure ;;
  checks) rl claude cov_checks $D/coverage_checks.py ;;
  *) echo "usage: launch_coverage.sh manifest | pilot | project | rest | measure | checks"; exit 2 ;;
esac
