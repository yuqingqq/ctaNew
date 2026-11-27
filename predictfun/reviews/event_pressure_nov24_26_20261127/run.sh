#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
out="$root/data/predict_fun/research/event_pressure_nov24_26"
mkdir -p "$out/logs"
exec 9>"$out/driver.lock"
flock -n 9
stage() {
  local tag="${2:-block}_$1"
  local log="$out/logs/$tag.log"
  if test -s "$log.done"; then return; fi
  echo "$(date -u +%FT%TZ) $tag started"
  if ! PF_SLOT=codex "$root/predictfun/research/run_light.sh" "event_nov_$tag" -- /home/yuqing/pricer-sol/venv/bin/python3 -u "$here/replay_block.py" "$@" > "$log" 2>&1; then
    tail -16 "$log"
    return 1
  fi
  tail -1 "$log" | tee "$log.done"
}
for day in 20261124 20261125 20261126; do
  if [[ "$day" == 20261126 ]]; then
    for step in pf_step1 pf_step1_timing pf_step1_booktime; do stage "$step" "$day"; done
  else
    stage cached_book "$day"
  fi
  for step in pf_txmap signals baseline event; do
    stage "$step" "$day"
  done
done
stage finish
