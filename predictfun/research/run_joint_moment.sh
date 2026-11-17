#!/usr/bin/env bash
set -euo pipefail
MOMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MOMENT_PY=/home/yuqing/pricer-sol/venv/bin/python3
MOMENT_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
MOMENT_OUT=/home/yuqing/ctaNew/data/predict_fun/research/joint_moment_20261117_v2
mkdir -p "$MOMENT_OUT/logs"
PF_SLOT=codex "$MOMENT_WRAP" moment_selftest -- "$MOMENT_PY" "$MOMENT_DIR/pf_joint_moment_fit.py" selftest >"$MOMENT_OUT/logs/selftest.log" 2>&1
for day in 20261023 20261024 20261025 20261026 20261027; do
  PF_SLOT=codex "$MOMENT_WRAP" "moment_data_$day" -- "$MOMENT_PY" "$MOMENT_DIR/pf_joint_moment_data.py" build "$day" >"$MOMENT_OUT/logs/data_$day.log" 2>&1
done
PF_SLOT=codex "$MOMENT_WRAP" moment_fit -- "$MOMENT_PY" "$MOMENT_DIR/pf_joint_moment_fit.py" fit >"$MOMENT_OUT/logs/fit.log" 2>&1
for day in 20261026 20261027; do
  PF_SLOT=codex "$MOMENT_WRAP" "moment_eval_$day" -- "$MOMENT_PY" "$MOMENT_DIR/pf_joint_moment_fit.py" evaluate "$day" >"$MOMENT_OUT/logs/eval_$day.log" 2>&1
done
PF_SLOT=codex "$MOMENT_WRAP" moment_audit -- "$MOMENT_PY" "$MOMENT_DIR/pf_joint_moment_fit.py" audit >"$MOMENT_OUT/logs/audit.log" 2>&1
printf 'Joint moment prediction screen complete\n'
