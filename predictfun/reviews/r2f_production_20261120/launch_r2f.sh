#!/bin/bash
# R2-F (review 38febc0b): the parity run and the full production label sample on consumed Oct 24-25. Every job via run_light (absolute
# paths). The 40 core-hour rule: each slot queue checks a STOP file before every partition; `projection` writes it when the finished
# partitions' mean CPU x the remaining partitions would pass the ceiling.
#   plans                 -> the parity plan (pilot run set) and the production plan (the full sample), audit every 35
#   parity <slot> <k>...  -> parity partitions k on one slot
#   queue <slot> <k>...   -> production partitions k on one slot, in order (stops at the STOP file)
#   projection            -> the projected production core-hours from the finished partitions; writes STOP above 40
#   finalize              -> finalize the parity and production outputs
set -u
R=/home/yuqing/ctaNew; M=$R/predictfun/research/pf_r2_label_sample.py; D=$R/data/predict_fun/research/r2_label_sample_v2_20261120
L=$R/predictfun/reviews/r2f_production_20261120/logs; mkdir -p $L; STOP=$L/STOP_40_CORE_HOURS
lr() { env PF_SLOT=$1 $R/predictfun/research/run_light.sh $2 -- python3 $M "${@:3}" > $L/$2.log 2>&1; echo "$2 rc=$?" >> $L/jobs_rc.txt; }
case "$1" in
  plans) lr claude2 r2f_plan_parity plan $D/parity_pilot_e35 pilot & lr claude3 r2f_plan_production plan $D/production_v1 & wait ;;
  parity) S=$2; shift 2; for k in "$@"; do lr $S r2f_parity_run_$k run $D/parity_pilot_e35 $k; done ;;
  queue) S=$2; shift 2; for k in "$@"; do [ -f $STOP ] && { echo "r2f_run_$k skipped (STOP)" >> $L/jobs_rc.txt; continue; }; lr $S r2f_run_$k run $D/production_v1 $k; done ;;
  projection) python3 - "$L" "$D/production_v1" <<'PY'
import json, re, sys, glob
from pathlib import Path
L, P = Path(sys.argv[1]), Path(sys.argv[2]); n = len(json.loads((P / 'plan.json').read_text())['partitions'])
cpu = [float(m.group(1)) for f in glob.glob(str(L / 'r2f_run_*.log')) for m in [re.search(r'rc=0 peak_rss_mib=\d+ wall_s=[\d.]+ cpu_user_s=([\d.]+)', Path(f).read_text())] if m]
proj = sum(cpu) / len(cpu) * n / 3600. if cpu else None
print(json.dumps(dict(finished=len(cpu), partitions=n, mean_cpu_s=(sum(cpu) / len(cpu) if cpu else None), projected_core_hours=proj, ceiling=40.)))
if proj is not None and proj > 40.:
    (L / 'STOP_40_CORE_HOURS').write_text(json.dumps(dict(finished=len(cpu), projected_core_hours=proj)) + '\n')
PY
  ;;
  finalize) lr claude r2f_finalize_parity finalize $D/parity_pilot_e35; lr claude r2f_finalize finalize $D/production_v1 ;;
esac
