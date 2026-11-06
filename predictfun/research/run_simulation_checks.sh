#!/usr/bin/env bash
# Guarded, source-only regression/causality gate. No historical or forward market data is read.
set -euo pipefail
cd /home/yuqing/ctaNew
TAG=${1:-simulation_checks}
exec predictfun/research/run_light.sh "$TAG" -- \
  python3 -m pytest -q predictfun/research/tests/test_simulation_causality.py --tb=short
