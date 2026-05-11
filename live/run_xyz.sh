#!/bin/bash
# Wrapper for xyz cron jobs: cd to repo root, source .env (auto-export),
# then exec the venv python with whatever args are passed.
#
# Usage from cron:
#   /home/yuqing/ctaNew/live/run_xyz.sh -m live.xyz_paper_bot
#   /home/yuqing/ctaNew/live/run_xyz.sh -m live.xyz_hourly_monitor
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x .venv/bin/python3 ]; then
    PYTHON_BIN=".venv/bin/python3"
  elif [ -x venv/bin/python3 ]; then
    PYTHON_BIN="venv/bin/python3"
  else
    PYTHON_BIN="python3"
  fi
fi
exec "$PYTHON_BIN" "$@"
