#!/bin/bash
# LONG-VALUE-RETENTION-DISCRIMINATION-v1 launcher (session aea5bfb9, 2026-11-22; declaration.md). Stages, in order:
#   manifest | selftest | prefix | project | rest <day> <slot> | measure | checks
# run_light, absolute paths; rest refuses unless the write-once projection says run_rest (the builder checks it too); at most 4 jobs.
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_retention_discrimination_20261122; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
SYN=$R/data/predict_fun/research/r2_teacher_selftest_world
rl() { env PF_SLOT=$1 PF_TEACHER_SYN=$SYN ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude rd_manifest $P/pf_ladder_retention_discrimination.py manifest ;;
  selftest) rl claude rd_selftest_frozen $P/pf_ladder_retention_discrimination.py selftest ;;
  prefix) rl claude rd_prefix $P/pf_ladder_retention_discrimination.py prefix ;;
  project) rl claude rd_projection $P/pf_ladder_retention_discrimination.py project ;;
  rest)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/projection.json'))['run_rest'] is True else 1)" || { echo "rest $2 REFUSED by the projection" >> $J; exit 1; }
    rl "${3:-claude}" "rd_rest_$2" $P/pf_ladder_retention_discrimination.py rest "$2" ;;
  measure) rl claude rd_measure $P/pf_ladder_retention_discrimination.py measure ;;
  checks) rl claude rd_checks $D/rd_checks.py ;;
  *) echo "usage: launch_retention.sh manifest | selftest | prefix | project | rest <day> <slot> | measure | checks"; exit 2 ;;
esac
