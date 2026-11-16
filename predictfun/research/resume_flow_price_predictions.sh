#!/usr/bin/env bash
# Resume the original frozen prediction study after the controller pilot releases core8.
set -euo pipefail
FLOW_SCRIPTS=/tmp/cta-flow-price-model-20261116/predictfun/research
FLOW_PY=/home/yuqing/pricer-sol/venv/bin/python3
FLOW_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
FLOW_OUT=/home/yuqing/ctaNew/data/predict_fun/research/flow_price_20261116_v1
FLOW_LOGS="$FLOW_OUT/logs"
for day in 20261101 20261102 20261103; do
  if [[ ! -s "$FLOW_OUT/$day/receipt.json" ]]; then
    [[ ! -e "$FLOW_OUT/$day" ]] || { printf 'REFUSED: incomplete data output %s retained\n' "$day" >&2; exit 1; }
    PF_SLOT=codex "$FLOW_WRAP" "flow_price_data_$day" -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_data.py" "$day" >"$FLOW_LOGS/data_$day.log" 2>&1
  fi
  if [[ ! -s "$FLOW_OUT/fit_$day/receipt.json" ]]; then
    [[ ! -e "$FLOW_OUT/fit_$day" ]] || { printf 'REFUSED: incomplete fit output %s retained\n' "$day" >&2; exit 1; }
    PF_SLOT=codex "$FLOW_WRAP" "flow_price_fit_$day" -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_model.py" "$day" >"$FLOW_LOGS/fit_$day.log" 2>&1
  fi
done
PF_SLOT=codex "$FLOW_WRAP" flow_price_report -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_report.py" >"$FLOW_LOGS/report.log" 2>&1
PF_SLOT=codex "$FLOW_WRAP" flow_price_audit -- "$FLOW_PY" "$FLOW_SCRIPTS/pf_flow_price_check.py" audit >"$FLOW_LOGS/audit.log" 2>&1
printf 'Remaining frozen prediction folds and audit complete\n'
