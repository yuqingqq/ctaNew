#!/bin/bash
# R3 ADV2-RICH v1 (review 315979d2): the evaluation label sample on consumed Oct 26-27 through pf_adv2_sample.py (the frozen runner with
# the day-guarded identity). Every job via run_light (absolute paths). The round's 40 core-hour ceiling: the label build stops (STOP file
# checked before every partition) when its projection from the finished partitions passes 28 core-hours, leaving 12 for the fits,
# the evaluation and the bounded replay.
#   queue <slot> <k>...   -> evaluation partitions k on one slot, in order (stops at the STOP file)
#   projection            -> the projected label core-hours from the finished partitions; writes STOP above 28
#   finalize              -> finalize the evaluation sample (accepted.json: statuses and weight totals, no label read)
#   selftests <slot>      -> the four modules' selftests (synthetic; the cached synthetic world)
#   mutants <slot>...     -> the candidate / evaluation / replay mutation suites (one mutant per job) and their summaries
#   extract | fit         -> the Oct 24-25 training extraction, then the 12 fits (fit.json: the model hashes)
#   eval [slot]           -> the offline prediction diagnostic (refuses unless fit.json is committed)
#   replay <slot> <k>...  -> replay partitions k (refuses unless fit.json is committed); partition 0 is the cost pilot
#   summarize | checks    -> the replay summary; the round's declared checks (r3_checks.py)
set -u
R=/home/yuqing/ctaNew; M=$R/predictfun/research/pf_adv2_sample.py; D=$R/data/predict_fun/research/adv2_rich_v1_20261121/eval_labels_v1
L=$R/predictfun/reviews/r3_adv2_20261121/logs; mkdir -p $L; STOP=$L/STOP_LABEL_28_CORE_HOURS
SYN=$R/data/predict_fun/research/r2_teacher_selftest_world
lr() { env PF_SLOT=$1 $R/predictfun/research/run_light.sh $2 -- python3 $M "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $L/jobs_rc.txt; }
case "$1" in
  queue) S=$2; shift 2; for k in "$@"; do [ -f $STOP ] && { echo "r3_eval_$k skipped (STOP)" >> $L/jobs_rc.txt; continue; }; lr $S r3_eval_$k run $D $k; done ;;
  projection) python3 - "$L" "$D" <<'PY'
import json, re, sys, glob
from pathlib import Path
L, P = Path(sys.argv[1]), Path(sys.argv[2]); n = len(json.loads((P / 'plan.json').read_text())['partitions'])
cpu = [float(m.group(1)) for f in glob.glob(str(L / 'r3_eval_*.log')) for m in [re.search(r'rc=0 peak_rss_mib=\d+ wall_s=[\d.]+ cpu_user_s=([\d.]+)', Path(f).read_text())] if m]
proj = sum(cpu) / len(cpu) * n / 3600. if cpu else None
print(json.dumps(dict(finished=len(cpu), partitions=n, mean_cpu_s=(sum(cpu) / len(cpu) if cpu else None), projected_core_hours=proj, ceiling=28.)))
if proj is not None and proj > 28.:
    (L / 'STOP_LABEL_28_CORE_HOURS').write_text(json.dumps(dict(finished=len(cpu), projected_core_hours=proj)) + '\n')
PY
  ;;
  finalize) lr claude r3_eval_finalize finalize $D ;;
  selftests) S=$2; for m in pf_adv2_sample pf_adv2 pf_adv2_eval pf_adv2_replay; do
      env PF_SLOT=$S PF_TEACHER_SYN=$SYN $R/predictfun/research/run_light.sh ${m}_selftest -- python3 $R/predictfun/research/$m.py selftest > $L/${m}_selftest.log 2>&1
      echo "${m}_selftest rc=$?" >> $L/jobs_rc.txt; tail -1 $L/${m}_selftest.log | cut -c1-160; done ;;
  mutants) shift 1; for h in adv2 adv2_eval adv2_replay; do n=$(python3 -c "import sys; sys.path.insert(0, '$R/predictfun/research'); import pf_${h}_mutants as H; print(len(H.MUTANTS))")
      env PF_TEACHER_SYN=$SYN bash $R/predictfun/research/pf_mutant_queue.sh $R/predictfun/research/pf_${h}_mutants.py mut_${h#adv2_} $L/mutants_${h#adv2_} $n "$@" > /dev/null 2>&1
      python3 $R/predictfun/research/pf_mutant_summary.py $R/predictfun/research/pf_${h}_mutants.py $L/mutants_${h#adv2_} mut_${h#adv2_} > $L/mutants_${h#adv2_}_summary.log 2>&1
      tail -1 $L/mutants_${h#adv2_}_summary.log | cut -c1-200; done ;;
  extract) env PF_SLOT=claude $R/predictfun/research/run_light.sh r3_extract -- python3 $R/predictfun/research/pf_adv2.py extract > $L/r3_extract.log 2>&1; echo "r3_extract rc=$?" >> $L/jobs_rc.txt ;;
  fit) env PF_SLOT=claude $R/predictfun/research/run_light.sh r3_fit -- python3 $R/predictfun/research/pf_adv2.py fit > $L/r3_fit.log 2>&1; echo "r3_fit rc=$?" >> $L/jobs_rc.txt ;;
  eval) env PF_SLOT=${2:-claude} $R/predictfun/research/run_light.sh r3_eval_readings -- python3 $R/predictfun/research/pf_adv2_eval.py eval > $L/r3_eval_readings.log 2>&1; echo "r3_eval_readings rc=$?" >> $L/jobs_rc.txt ;;
  replay) S=$2; shift 2; for k in "$@"; do env PF_SLOT=$S $R/predictfun/research/run_light.sh r3_replay_$k -- python3 $R/predictfun/research/pf_adv2_replay.py run $k > $L/r3_replay_$k.log 2>&1; echo "r3_replay_$k rc=$?" >> $L/jobs_rc.txt; done ;;
  summarize) env PF_SLOT=claude $R/predictfun/research/run_light.sh r3_replay_summary -- python3 $R/predictfun/research/pf_adv2_replay.py summarize > $L/r3_replay_summary.log 2>&1; echo "r3_replay_summary rc=$?" >> $L/jobs_rc.txt ;;
  checks) env PF_SLOT=claude $R/predictfun/research/run_light.sh r3_checks -- python3 $R/predictfun/reviews/r3_adv2_20261121/r3_checks.py > $L/r3_checks.log 2>&1; echo "r3_checks rc=$?" >> $L/jobs_rc.txt ;;
esac
