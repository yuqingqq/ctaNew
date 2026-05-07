#!/bin/bash
# Wrapper for cron: cd to repo root, source .env (auto-exporting all keys),
# then exec the venv python with whatever args are passed.
#
# Usage from cron:
#   /home/yuqing/ctaNew/live/run_with_env.sh -m live.paper_bot --source hl
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
exec /usr/bin/python3 "$@"
