#!/usr/bin/env bash
set -euo pipefail
DECOMP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DECOMP_PY=/home/yuqing/pricer-sol/venv/bin/python3
DECOMP_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
DECOMP_OUT=/home/yuqing/ctaNew/data/predict_fun/research/value_decomposition_20261117_v1
mkdir -p "$DECOMP_OUT/logs"
PF_SLOT=codex "$DECOMP_WRAP" value_decomposition_selftest -- "$DECOMP_PY" "$DECOMP_DIR/pf_value_decomposition_run.py" selftest >"$DECOMP_OUT/logs/selftest.log" 2>&1
for day in 20261026 20261027; do
  PF_SLOT=codex "$DECOMP_WRAP" "value_decomposition_$day" -- "$DECOMP_PY" "$DECOMP_DIR/pf_value_decomposition_run.py" collect "$day" >"$DECOMP_OUT/logs/$day.log" 2>&1
done
PF_SLOT=codex "$DECOMP_WRAP" value_decomposition_audit -- "$DECOMP_PY" "$DECOMP_DIR/pf_value_decomposition_run.py" audit >"$DECOMP_OUT/logs/audit.log" 2>&1
printf 'Value decomposition complete\n'
