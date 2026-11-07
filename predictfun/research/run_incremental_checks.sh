#!/usr/bin/env bash
# Source-only checks, using the existing one-process resource guard.
set -euo pipefail
cd /home/yuqing/ctaNew
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec predictfun/research/run_light.sh "${1:-incremental_size_checks}" -- \
  python3 -m predictfun.research.pf_incremental_checks
