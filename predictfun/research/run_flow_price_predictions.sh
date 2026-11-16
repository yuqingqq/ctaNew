#!/usr/bin/env bash
set -euo pipefail
FLOW_SCRIPTS="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FLOW_PY=/home/yuqing/pricer-sol/venv/bin/python3
FLOW_WRAPPER=/home/yuqing/ctaNew/predictfun/research/run_light.sh
FLOW_OUT=/home/yuqing/ctaNew/data/predict_fun/research/flow_price_20261116_v1
FLOW_LOGS="$FLOW_OUT/logs"
mkdir -p "$FLOW_LOGS"
for day in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
  printf 'Building %s\n' "$day"
  PF_SLOT=codex "$FLOW_WRAPPER" "flow_price_data_$day" -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_data.py" "$day" >"$FLOW_LOGS/data_$day.log" 2>&1
done
for day in 20261027 20261031 20261101 20261102 20261103; do
  printf 'Fitting %s\n' "$day"
  PF_SLOT=codex "$FLOW_WRAPPER" "flow_price_fit_$day" -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_model.py" "$day" >"$FLOW_LOGS/fit_$day.log" 2>&1
done
printf 'Reporting predictions\n'
PF_SLOT=codex "$FLOW_WRAPPER" flow_price_report -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_report.py" >"$FLOW_LOGS/report.log" 2>&1
printf 'Auditing artifacts\n'
PF_SLOT=codex "$FLOW_WRAPPER" flow_price_audit -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_check.py" audit >"$FLOW_LOGS/audit.log" 2>&1
printf 'Prediction experiment complete\n'
