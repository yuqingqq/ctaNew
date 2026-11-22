#!/bin/bash
# QUEUE-PRIORITY-SENSITIVITY-v1 launcher (session aea5bfb9, 2026-11-22; declaration.md). Stages, in order:
#   manifest | selftest | pilot <day> <slot> | project | rest <day> <slot> | measure | checks
# run_light, absolute paths; at most four day jobs at once on PF_SLOT claude..claude4; rest refuses unless the write-once projection fits.
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; D=$R/predictfun/reviews/ladder_queue_front_20261122; L=$D/logs; J=$L/jobs_rc.txt; mkdir -p $L
SYN=$R/data/predict_fun/research/r2_teacher_selftest_world
rl() { env PF_SLOT=$1 PF_TEACHER_SYN=$SYN ARROW_DEFAULT_MEMORY_POOL=system $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "${1:-}" in
  manifest) rl claude qf_manifest $P/pf_ladder_queue_front.py manifest ;;
  selftest) rl claude qf_selftest_frozen $P/pf_ladder_queue_front.py selftest ;;
  pilot) rl "${3:-claude}" "qf_pilot_$2" $P/pf_ladder_queue_front.py pilot "$2" ;;
  project) rl claude qf_projection $P/pf_ladder_queue_front.py project ;;
  rest)
    python3 -c "import json, sys; sys.exit(0 if json.load(open('$L/projection.json'))['run_rest'] is True else 1)" || { echo "rest $2 REFUSED by the projection" >> $J; exit 1; }
    rl "${3:-claude}" "qf_rest_$2" $P/pf_ladder_queue_front.py rest "$2" ;;
  measure) rl claude qf_measure $P/pf_ladder_queue_front.py measure ;;
  checks) rl claude qf_checks $D/fq_checks.py ;;
  *) echo "usage: launch_queue_front.sh manifest | selftest | pilot <day> <slot> | project | rest <day> <slot> | measure | checks"; exit 2 ;;
esac
