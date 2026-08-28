#!/usr/bin/env bash
# Print ONLY the verified falsifier count for a module, on stdout, nothing else.
# Use its output in a commit message; do not type a count from memory.
# The first version let the module's own stdout leak when the selftest path was
# taken via the fallback, which put a stray count line into a commit message.
set -uo pipefail
m="${1:?module}"
n=$(python3 "$m" --selftest 2>/dev/null | grep -c "^  PASS " || true)
if [ "${n:-0}" -eq 0 ]; then
  n=$(python3 "$m" 2>/dev/null | grep -c "^  PASS " || true)
fi
printf '%s\n' "${n:-0}"
