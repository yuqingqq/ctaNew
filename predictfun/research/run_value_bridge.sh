#!/usr/bin/env bash
set -euo pipefail
BRIDGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE_PY=/home/yuqing/pricer-sol/venv/bin/python3
BRIDGE_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
BRIDGE_OUT=/home/yuqing/ctaNew/data/predict_fun/research/value_bridge_20261116_v1
mkdir -p "$BRIDGE_OUT/logs"
for day in 20261026 20261027; do
  PF_SLOT=codex "$BRIDGE_WRAP" "value_bridge_actions_$day" -- "$BRIDGE_PY" "$BRIDGE_DIR/pf_value_bridge_run.py" actions "$day" >"$BRIDGE_OUT/logs/actions_$day.log" 2>&1
done
PF_SLOT=codex "$BRIDGE_WRAP" value_bridge_pilot -- "$BRIDGE_PY" "$BRIDGE_DIR/pf_value_bridge_run.py" pilot >"$BRIDGE_OUT/logs/pilot.log" 2>&1
PF_SLOT=codex "$BRIDGE_WRAP" value_bridge_audit -- "$BRIDGE_PY" "$BRIDGE_DIR/pf_value_bridge_run.py" audit >"$BRIDGE_OUT/logs/audit.log" 2>&1
printf 'Value bridge comparison and audit complete\n'
