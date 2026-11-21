#!/bin/bash
# R4E ADV2-ENTRY-SMOOTH-v1 (review 371aab56): every job via run_light (absolute paths). A new launcher for this round: never edit a launcher
# a running job executes (bash re-reads its script from its byte offset; feedback 2026-11-21). The round's ceiling: 8 core-hours.
#   selftests <slot>        -> the three modules' selftests (synthetic; the cached synthetic world)
#   mutants <slot>...       -> the three mutation suites (one mutant per job) and their summaries
#   fit                     -> the two SMOOTH POST fits (fit.json: the model hashes)
#   eval [slot]             -> the offline entry diagnostic (refuses unless fit.json is committed)
#   replay <slot> <k>...    -> replay partitions k (refuses unless fit.json is committed); partition 0 is the cost pilot
#   summarize | checks      -> the replay summary; the declared checks and their falsifier
set -u
R=/home/yuqing/ctaNew; P=$R/predictfun/research; RC=$R/predictfun/reviews/r4e_entry_smooth_20261121; L=$RC/logs; mkdir -p $L; J=$L/jobs_rc.txt
SYN=$R/data/predict_fun/research/r2_teacher_selftest_world
rl() { env PF_SLOT=$1 PF_TEACHER_SYN=$SYN $P/run_light.sh $2 -- python3 "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $J; }
case "$1" in
  selftests) for m in pf_adv2_entry pf_adv2_entry_eval pf_adv2_entry_replay; do rl $2 ${m}_selftest $P/$m.py selftest; tail -1 $L/${m}_selftest.log | cut -c1-160; done ;;
  mutants) shift 1; for h in entry entry_eval entry_replay; do n=$(python3 -c "import sys; sys.path.insert(0, '$P'); import pf_adv2_${h}_mutants as H; print(len(H.MUTANTS))")
      env PF_TEACHER_SYN=$SYN bash $P/pf_mutant_queue.sh $P/pf_adv2_${h}_mutants.py mut_$h $L/mutants_$h $n "$@" > /dev/null 2>&1
      python3 $P/pf_mutant_summary.py $P/pf_adv2_${h}_mutants.py $L/mutants_$h mut_$h > $L/mutants_${h}_summary.log 2>&1
      tail -1 $L/mutants_${h}_summary.log | cut -c1-200; done ;;
  fit) rl claude r4e_fit $P/pf_adv2_entry.py fit ;;
  eval) rl ${2:-claude} r4e_eval $P/pf_adv2_entry_eval.py eval ;;
  replay) S=$2; shift 2; for k in "$@"; do rl $S r4e_replay_$k $P/pf_adv2_entry_replay.py run $k; done ;;
  summarize) rl claude r4e_replay_summary $P/pf_adv2_entry_replay.py summarize ;;
  checks) rl claude r4e_checks $RC/r4e_checks.py ;;
esac
