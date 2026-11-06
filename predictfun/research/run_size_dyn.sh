#!/usr/bin/env bash
# Size x stop grid (STATUS 2026-11-04 10:32Z) + dynamic sizes (10:37Z). Takes over from run_size_stop.sh (stopped after its first
# day-run); a day-run whose summary.json exists is skipped. Order: grid k=60, dynamic k=60, report; grid 50 / 70, dynamic 50 / 70, report.
set -u
cd /home/yuqing/ctaNew
while ps -eo args | grep -q "[p]f_step4_live 2026"; do sleep 10; done
R=predictfun/research/run_light.sh; L=data/predict_fun/research/logs; O=data/predict_fun/research/step4
run() { local tag=$1; shift
  while true; do
    $R "$tag" -- "$@" > "$L/$tag.log" 2>&1 && return 0
    grep -qE '^REFUSED: (a predictfun research process|MemAvailable|disk free)' "$L/$tag.log" || { echo "FAILED $tag"; tail -3 "$L/$tag.log"; exit 1; }
    sleep 60
  done; }
export PF_RL_LPLACE=40 PF_RL_LCANCEL=22 PF_RL_LPENDCANCEL=95 PF_RL_LFILL=100 PF_RL_DDEC=2 PF_TIMING=e2
B=G4+G2+G6+G5+G3+G1; GZ=""; DY=""
for sz in A1:5:25:0 A1:10:50:0 A1:20:100:0; do for s in 60 120 180 240; do GZ="${GZ:+$GZ;}MICRO|$B+V0+XNA+XMIN1+S$s|mid|$sz"; done; done
for sz in 25 50 100 AV:10:50:0.03 AV:20:100:0.03 AV:20:100:0.06 A3:5:100:0.2; do for s in 60 240; do DY="${DY:+$DY;}MICRO|$B+V0+XNA+XMIN1+S$s|mid|$sz"; done; done
DAYS="20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103"
day() { local k=$1 tag=$2 arms=$3 d=$4
  if [ -f "$O/live_tokyoreal_custom_e2_preds_g${k}_$tag/$d/summary.json" ]; then echo "skip ${tag}${k}_$d"; return 0; fi
  PF_PREDS=preds_g$k PF_ARMS="$arms" PF_TAG=$tag run "${tag}${k}_$d" python3 -m predictfun.research.pf_step4_live "$d" tokyoreal custom; }
for d in $DAYS; do day 60 gz "$GZ" $d; done
for d in $DAYS; do day 60 gzd "$DY" $d; done
( run size_stop_report_60 python3 -m predictfun.research.pf_size_stop_report ) || echo "REPORT FAILED after k=60 (chain continues)"
for k in 50 70; do for d in $DAYS; do day $k gz "$GZ" $d; done; done
for k in 50 70; do for d in $DAYS; do day $k gzd "$DY" $d; done; done
( run size_stop_report_all python3 -m predictfun.research.pf_size_stop_report ) || echo "REPORT FAILED (final)"
echo ALL_DONE
