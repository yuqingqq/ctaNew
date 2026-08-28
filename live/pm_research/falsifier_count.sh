#!/usr/bin/env bash
# Print the verified falsifier count for a module. Use its OUTPUT in a commit
# message; do not type a count from memory. Twice in consecutive commits I wrote
# a number before running it, and proofreading is not the fix — generating is.
set -euo pipefail
python3 "${1:?module}" --selftest 2>/dev/null | grep -c "  PASS " || \
  python3 "${1}" 2>/dev/null | grep -c "  PASS "
