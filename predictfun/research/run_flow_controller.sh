#!/usr/bin/env bash
set -euo pipefail
FLOW_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLOW_OUT=/home/yuqing/ctaNew/data/predict_fun/research/flow_control_20261116_v2
FLOW_PY=/home/yuqing/pricer-sol/venv/bin/python3
FLOW_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
mkdir -p "$FLOW_OUT/logs"
for stage in bank smoke pilot; do
  if [[ ! -s "$FLOW_OUT/$stage/receipt.json" ]]; then
    PF_SLOT=codex "$FLOW_WRAP" "flow_control_$stage" -- "$FLOW_PY" "$FLOW_DIR/pf_flow_control_run.py" "$stage" >"$FLOW_OUT/logs/$stage.log" 2>&1
  fi
done
printf 'Controller pilot complete\n'
