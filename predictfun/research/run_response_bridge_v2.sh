#!/usr/bin/env bash
set -euo pipefail
RESPONSE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESPONSE_PY=/home/yuqing/pricer-sol/venv/bin/python3
RESPONSE_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
RESPONSE_OUT=/home/yuqing/ctaNew/data/predict_fun/research/response_bridge_20261116_v2
mkdir -p "$RESPONSE_OUT/logs"
for stage in selftest fit predictions; do
  PF_SLOT=codex "$RESPONSE_WRAP" "response_v2_$stage" -- "$RESPONSE_PY" "$RESPONSE_DIR/pf_response_eval_v2.py" "$stage" >"$RESPONSE_OUT/logs/$stage.log" 2>&1
done
for day in 20261026 20261027; do
  PF_SLOT=codex "$RESPONSE_WRAP" "response_v2_actions_$day" -- "$RESPONSE_PY" "$RESPONSE_DIR/pf_response_eval_v2.py" actions "$day" >"$RESPONSE_OUT/logs/actions_$day.log" 2>&1
done
for stage in pilot audit; do
  PF_SLOT=codex "$RESPONSE_WRAP" "response_v2_$stage" -- "$RESPONSE_PY" "$RESPONSE_DIR/pf_response_eval_v2.py" "$stage" >"$RESPONSE_OUT/logs/$stage.log" 2>&1
done
PF_SLOT=codex "$RESPONSE_WRAP" response_v2_oracle -- "$RESPONSE_PY" "$RESPONSE_DIR/pf_response_oracle_v2.py" run >"$RESPONSE_OUT/logs/oracle.log" 2>&1
PF_SLOT=codex "$RESPONSE_WRAP" response_v2_record -- "$RESPONSE_PY" "$RESPONSE_DIR/pf_response_record_v2.py" >"$RESPONSE_OUT/logs/record.log" 2>&1
printf 'Response model comparison complete\n'
