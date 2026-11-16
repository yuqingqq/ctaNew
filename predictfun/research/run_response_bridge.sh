#!/usr/bin/env bash
set -euo pipefail
RESPONSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESPONSE_PY=/home/yuqing/pricer-sol/venv/bin/python3
RESPONSE_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
RESPONSE_OUT=/home/yuqing/ctaNew/data/predict_fun/research/response_bridge_20261116_v1
mkdir -p "$RESPONSE_OUT/logs"
for stage in selftest fit predictions; do
  PF_SLOT=codex "$RESPONSE_WRAP" "response_$stage" -- "$RESPONSE_PY" "$RESPONSE_DIR/pf_response_eval.py" "$stage" >"$RESPONSE_OUT/logs/$stage.log" 2>&1
done
for day in 20261026 20261027; do
  PF_SLOT=codex "$RESPONSE_WRAP" "response_actions_$day" -- "$RESPONSE_PY" "$RESPONSE_DIR/pf_response_eval.py" actions "$day" >"$RESPONSE_OUT/logs/actions_$day.log" 2>&1
done
for stage in pilot audit; do
  PF_SLOT=codex "$RESPONSE_WRAP" "response_$stage" -- "$RESPONSE_PY" "$RESPONSE_DIR/pf_response_eval.py" "$stage" >"$RESPONSE_OUT/logs/$stage.log" 2>&1
done
printf 'Response model comparison complete\n'
