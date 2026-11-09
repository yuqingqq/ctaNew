#!/usr/bin/env bash
set -euo pipefail
CHECKOUT=$(cd -- "$(dirname -- "$0")/../.." && pwd)
ROOT=/home/yuqing/ctaNew
OUT=$ROOT/data/predict_fun/research/fill_clock_20261109
run() {
  local tag=$1; shift
  while ps -eo args= | rg -q '^(/[^ ]*/)?python(3([.][0-9]+)?)? .*predictfun[./]research[./]|^bash [^ ]*run_live_samewindow.sh'; do
    echo "QUEUED $tag: shared research slot occupied"; sleep 15
  done
  echo "START $tag"
  if ! "$ROOT/predictfun/research/run_light.sh" "$tag" -- bash -c 'cd "$1"; shift; exec python3 -m "$@"' _ "$CHECKOUT" "$@" > "$OUT/logs/$tag.log" 2>&1; then
    tail -25 "$OUT/logs/$tag.log"; exit 1
  fi
  tail -2 "$OUT/logs/$tag.log"
}
run fill_clock_controls unittest predictfun.research.tests.test_fill_clock
for phase in extract anchors report; do
  run fill_clock_$phase predictfun.research.pf_fill_clock "$phase"
done
