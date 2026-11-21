#!/bin/bash
# R3 ADV2-RICH v1 (review 315979d2): the evaluation label sample on consumed Oct 26-27 through pf_adv2_sample.py (the frozen runner with
# the day-guarded identity). Every job via run_light (absolute paths). The round's 40 core-hour ceiling: the label build stops (STOP file
# checked before every partition) when its projection from the finished partitions passes 28 core-hours, leaving 12 for the fits,
# the evaluation and the bounded replay.
#   queue <slot> <k>...   -> evaluation partitions k on one slot, in order (stops at the STOP file)
#   projection            -> the projected label core-hours from the finished partitions; writes STOP above 28
#   finalize              -> finalize the evaluation sample (accepted.json: statuses and weight totals, no label read)
set -u
R=/home/yuqing/ctaNew; M=$R/predictfun/research/pf_adv2_sample.py; D=$R/data/predict_fun/research/adv2_rich_v1_20261121/eval_labels_v1
L=$R/predictfun/reviews/r3_adv2_20261121/logs; mkdir -p $L; STOP=$L/STOP_LABEL_28_CORE_HOURS
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
esac
