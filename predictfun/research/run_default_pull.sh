#!/usr/bin/env bash
# Durable bounded research loop. Numerical work is strictly serial and run_light-limited.
set -euo pipefail
ROOT=/home/yuqing/ctaNew
WORK=$(cd "$(dirname "$0")/../.." && pwd)
PY=/home/yuqing/pricer-sol/venv/bin/python3
OUT=$ROOT/data/predict_fun/research/default_pull_20261111
PROGRESS=$WORK/predictfun/research/pf_default_pull_progress.py
export DEFAULT_PULL_DRIVER_PID=$$
mkdir -p "$OUT/logs"
exec 9>"$OUT/driver.lock"
flock -n 9 || { echo 'REFUSED: another default-pull driver owns this loop'; exit 1; }
update() { "$PY" "$PROGRESS" "$@"; }
wait_slot() {
  while ! "$PY" "$PROGRESS" slot > "$OUT/slot.json"; do
    update "$1" queued 'Waiting for the shared numeric slot; no job started.'
    sleep 15
  done
}
run() {
  local tag=$1; shift
  if [ -e "$OUT/logs/$tag.done" ]; then sha256sum -c "$OUT/logs/$tag.done"; echo "already complete: $tag"; return; fi
  if [ -e "$OUT/logs/$tag.log" ]; then update "$tag" failed 'Existing incomplete log: refusing an automatic retry/overwrite.'; exit 1; fi
  wait_slot "$tag"
  update "$tag" running 'One numeric process on core8; output retained in per-stage log.'
  set +e
  "$ROOT/predictfun/research/run_light.sh" "default_pull_$tag" -- bash -c 'cd "$1" && shift && exec "$@"' _ "$WORK" "$PY" "$@" > "$OUT/logs/$tag.log" 2>&1
  local rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then update "$tag" failed "Exit $rc. Inspect logs/$tag.log; later stages were not launched."; exit "$rc"; fi
  sha256sum "$OUT/logs/$tag.log" > "$OUT/logs/$tag.done"
  update "$tag" complete "Passed; immutable checks/results and logs/$tag.log retained."
}
trap 'update driver interrupted "Driver received interruption; no automatic continuation."; exit 130' INT TERM
run controls -m unittest predictfun.research.tests.test_default_pull -v
run smoke -m predictfun.research.pf_default_pull_engine 20261023 smoke
for d in 20261023 20261024 20261025 20261026 20261027 20261031 20261101 20261102 20261103; do
  run "build_$d" -m predictfun.research.pf_default_pull_engine "$d" build
done
run fit -m predictfun.research.pf_default_pull_fit
for d in 20261026 20261027 20261031 20261101 20261102 20261103; do
  run "eval_$d" -m predictfun.research.pf_default_pull_engine "$d" eval
done
run primary_report -m predictfun.research.pf_default_pull_report primary
for d in 20261026 20261027 20261031 20261101 20261102 20261103; do
  run "stress_$d" -m predictfun.research.pf_default_pull_engine "$d" stress
done
run final_report -m predictfun.research.pf_default_pull_report final
update driver complete 'All predeclared tests complete. Read comparison_final.json and handoffs/20261111_default_pull_final.md. No live promotion.'
