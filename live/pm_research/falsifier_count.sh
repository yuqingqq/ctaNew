#!/usr/bin/env bash
# Print ONLY the verified falsifier count for a module, on STDOUT.
#
# FAILS LOUDLY ON STDOUT, not just stderr. A caller running this with
# `2>/dev/null` silenced the ${1:?module} error, its own || fallback filled the
# gap with a source-text grep, and the wrong number entered a register as
# "the script's output" (R-250). A tool whose only failure signal is stderr can
# be muted by its caller; this one puts the failure where the number goes, so a
# consumer reading stdout gets an unusable token instead of a plausible count.
#
# NEVER runs a module bare: the removed fallback could invoke main(), i.e. the
# HEAVY data path from the session shell, which R-148(3) forbids.
set -uo pipefail

die() { printf 'ERROR_NOT_A_COUNT: %s\n' "$1"; exit 2; }

[ $# -ge 1 ] || die "usage: falsifier_count.sh <module.py> [--ref]"
m="$1"
[ -f "$m" ] || die "no such module: $m"

n=$(timeout 180 python3 "$m" --selftest 2>/dev/null | grep -c "^  PASS " || true)
[ -n "${n:-}" ] || die "no count produced for $m"
[ "$n" -gt 0 ] 2>/dev/null || die "module $m produced ZERO PASS lines; a count \
of 0 is far more likely a broken invocation than a suite with no falsifiers"

# A count without a commit ref is not a measurement — both seats violated that
# comparing 126 against 134 across different trees (R-250).
if [ "${2:-}" = "--ref" ]; then
  printf '%s@%s\n' "$n" "$(git -C "$(dirname "$m")" rev-parse --short HEAD 2>/dev/null || echo nogit)"
else
  printf '%s\n' "$n"
fi
