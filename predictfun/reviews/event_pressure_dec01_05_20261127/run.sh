#!/usr/bin/env bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
stage() {
  PF_SLOT=codex "$root/predictfun/research/run_light.sh" "pressure_dec_$1_${2:-block}" -- /home/yuqing/pricer-sol/venv/bin/python3 "$here/forward_pair.py" "$@"
}
for day in 20261201 20261202 20261203 20261204 20261205; do
  for step in pf_step1 pf_step1_timing pf_step1_booktime pf_txmap signals baseline event; do
    stage "$step" "$day"
  done
done
stage finish
