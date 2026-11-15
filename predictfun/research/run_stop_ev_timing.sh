#!/usr/bin/env bash
set -euo pipefail
TASK_HERE=$(cd "$(dirname "$0")" && pwd)
TASK_OUT=${1:?new output directory}
TASK_FREEZE=${2:?committed freeze path}
TASK_PY=/home/yuqing/pricer-sol/venv/bin/python3
mkdir -p "$TASK_OUT/logs"
for TASK_DAY in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
  PF_SLOT=codex "$TASK_HERE/run_light.sh" "stop_ev_timing_$TASK_DAY" -- "$TASK_PY" "$TASK_HERE/pf_stop_ev_timing_build.py" day --day "$TASK_DAY" --out "$TASK_OUT" --freeze "$TASK_FREEZE" > "$TASK_OUT/logs/$TASK_DAY.log" 2>&1
done
PF_SLOT=codex "$TASK_HERE/run_light.sh" stop_ev_timing_report -- "$TASK_PY" "$TASK_HERE/pf_stop_ev_timing_build.py" report --out "$TASK_OUT" --freeze "$TASK_FREEZE" > "$TASK_OUT/logs/report.log" 2>&1
