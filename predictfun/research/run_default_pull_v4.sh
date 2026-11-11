#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/yuqing/ctaNew
WORK=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/yuqing/pricer-sol/venv/bin/python3
OUT=$ROOT/data/predict_fun/research/default_pull_20261111_v4
HELPER=$WORK/predictfun/research/pf_default_pull_known_empty.py
export DEFAULT_PULL_DRIVER_PID=$$
mkdir -p "$OUT/logs"
exec 9>"$OUT/driver.lock"
flock -n 9 || { echo 'REFUSED: another v4 driver owns this loop'; exit 1; }
update() { "$PY" "$HELPER" progress "$@"; }
run() {
  local tag=$1; shift
  if [ -e "$OUT/logs/$tag.done" ]; then sha256sum -c "$OUT/logs/$tag.done"; return; fi
  if [ -e "$OUT/logs/$tag.log" ]; then update "$tag" failed 'Existing incomplete log; explicit recovery required.'; exit 1; fi
  while ! "$PY" "$HELPER" slot > "$OUT/slot.json"; do update "$tag" queued 'Waiting for the shared numeric slot.'; sleep 15; done
  update "$tag" running 'v4: observed empty levels are known, retaining all data and inventory controls.'
  set +e
  "$ROOT/predictfun/research/run_light.sh" "default_pull_v4_$tag" -- bash -c 'cd "$1" && shift && exec "$@"' _ "$WORK" "$PY" -m predictfun.research.pf_default_pull_known_empty "$@" > "$OUT/logs/$tag.log" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then update "$tag" failed "Exit $rc; logs/$tag.log retained, later stages stopped."; exit "$rc"; fi
  sha256sum "$OUT/logs/$tag.log" > "$OUT/logs/$tag.done"
  update "$tag" complete "Passed; logs/$tag.log plus artifact/feature checks retained."
}
trap 'update driver interrupted "Driver interrupted; completed stages retained."; exit 130' INT TERM
run controls controls
run smoke run 20261023 smoke
for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do run "build_$d" run "$d" build; done
run fit fit
for d in 20261026 20261027 20261031 20261101 20261102 20261103; do run "eval_$d" run "$d" eval; done
run primary_report report primary
for d in 20261026 20261027 20261031 20261101 20261102 20261103; do run "stress_$d" run "$d" stress; done
run final_report report final
run completion_audit audit
run comparison_figure plot
update driver complete 'All v4 tests complete. Canonical output default_pull_20261111_v4/comparison_final.json; completion artifact audit and comparison figure passed. No live promotion.'
