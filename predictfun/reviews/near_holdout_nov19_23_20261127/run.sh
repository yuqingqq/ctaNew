#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
out="$root/data/predict_fun/research/near_holdout_nov19_23"
py=/home/yuqing/pricer-sol/venv/bin/python3
mkdir -p "$out/logs"
run_stage() {
  stage="$1"
  shift
  if test -s "$out/logs/$stage.done"; then return; fi
  echo "$(date -u +%FT%TZ) starting $stage"
  PF_SLOT=codex "$root/predictfun/research/run_light.sh" "near_nov_$stage" -- "$py" "$@" >"$out/logs/$stage.log" 2>&1 || {
    tail -16 "$out/logs/$stage.log"
    return 1
  }
  tail -1 "$out/logs/$stage.log" >"$out/logs/$stage.done"
}
test -s "$out/control_pass.json"
for day in 20261119 20261120 20261121 20261122 20261123; do
  for stage in pf_step1 pf_step1_timing pf_step1_booktime pf_txmap; do
    run_stage "${day}_$stage" "$here/prepare.py" "$day" "$stage"
  done
  run_stage "${day}_signals" "$here/signals.py" "$day"
  run_stage "${day}_replay" "$here/replay.py" "$day"
done
run_stage score "$here/score.py"
