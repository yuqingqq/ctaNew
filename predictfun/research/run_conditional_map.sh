#!/usr/bin/env bash
set -euo pipefail
MAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MAP_PY=/home/yuqing/pricer-sol/venv/bin/python3
MAP_WRAP=/home/yuqing/ctaNew/predictfun/research/run_light.sh
MAP_OUT=/home/yuqing/ctaNew/data/predict_fun/research/conditional_map_20261117_v1
mkdir -p "$MAP_OUT/logs"
for stage in selftest bank; do
  PF_SLOT=codex "$MAP_WRAP" "conditional_map_$stage" -- "$MAP_PY" "$MAP_DIR/pf_conditional_map_eval.py" "$stage" >"$MAP_OUT/logs/$stage.log" 2>&1
done
for day in 20261026 20261027; do
  PF_SLOT=codex "$MAP_WRAP" "conditional_map_actions_$day" -- "$MAP_PY" "$MAP_DIR/pf_conditional_map_eval.py" actions "$day" >"$MAP_OUT/logs/actions_$day.log" 2>&1
done
for stage in gated_pilot audit; do
  PF_SLOT=codex "$MAP_WRAP" "conditional_map_$stage" -- "$MAP_PY" "$MAP_DIR/pf_conditional_map_eval.py" "$stage" >"$MAP_OUT/logs/$stage.log" 2>&1
done
printf 'Conditional mapping experiment complete\n'
