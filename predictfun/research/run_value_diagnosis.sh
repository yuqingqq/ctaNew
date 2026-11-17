#!/usr/bin/env bash
set -euo pipefail
DIAG_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DIAG_PY=/home/yuqing/pricer-sol/venv/bin/python3
DIAG_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
DIAG_OUT=/home/yuqing/ctaNew/data/predict_fun/research/value_diagnosis_20261117_v1
mkdir -p "$DIAG_OUT/logs"
for part in calibration controller; do
  PF_SLOT=codex "$DIAG_WRAP" "value_diagnosis_$part" -- "$DIAG_PY" "$DIAG_DIR/pf_value_diagnosis_$part.py" run >"$DIAG_OUT/logs/$part.log" 2>&1
done
PF_SLOT=codex "$DIAG_WRAP" value_diagnosis_audit -- "$DIAG_PY" "$DIAG_DIR/pf_value_diagnosis_audit.py" >"$DIAG_OUT/logs/audit.log" 2>&1
printf 'Value diagnosis complete\n'
