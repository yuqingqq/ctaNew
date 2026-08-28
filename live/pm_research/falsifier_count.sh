#!/usr/bin/env bash
# Print ONLY the verified falsifier count for a module.
#
# NEVER falls back to running a module bare. The previous fallback ran
# `python3 <module>` with no flag when --selftest produced no PASS lines, and
# for a runner that means main() — the HEAVY data path, launched from the
# session shell, which the resource rule forbids. A helper for counting tests
# must not be able to start a research run.
set -uo pipefail
m="${1:?module}"
n=$(timeout 120 python3 "$m" --selftest 2>/dev/null | grep -c "^  PASS " || true)
printf '%s\n' "${n:-0}"
