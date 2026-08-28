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
#
# THE COUNT IS ONLY VALID IF THE SUITE COMPLETED. The previous line 21 ended in
# `|| true`, which discarded the interpreter's exit status: a suite that printed
# some PASS lines and THEN crashed yielded a plausible count with wrapper_rc=0
# (probe: one PASS then exit 1 -> "1@<ref>"). A partial count is worse than an
# error, because it is indistinguishable from a real one. `|| true` was there to
# absorb grep's exit-1-on-zero-matches -- already covered by the >0 check below,
# so it bought nothing and hid everything. Statuses now propagate, and timeout
# is reported distinctly from a crash.
set -uo pipefail

FC_TIMEOUT="${FC_TIMEOUT:-180}"

die() { printf 'ERROR_NOT_A_COUNT: %s\n' "$1"; exit 2; }

# NOT a command substitution: `die` does `exit 2`, and inside $(...) that exits
# only the SUBSHELL -- the refusal text would be captured as the count and the
# script would sail on to the generic zero-check, reporting the WRONG reason for
# every refusal. (This script's own selftest caught exactly that, on the first
# run of this fix: three known-bads reported "ZERO PASS lines" instead of
# crashed/timed-out/missing.) The count comes back in a global instead.
COUNT=""
count_of() {  # module -> sets COUNT, or dies in the CALLER'''s shell
  local m="$1" out rc err tail_
  [ -f "$m" ] || die "no such module: $m"
  err="$(mktemp)"
  out="$(timeout "$FC_TIMEOUT" python3 "$m" --selftest 2>"$err")"; rc=$?
  tail_="$(tr '\n' ' ' <"$err" | tail -c 300)"; rm -f "$err"
  if [ "$rc" -ne 0 ]; then
    [ "$rc" -eq 124 ] && die "TIMED OUT after ${FC_TIMEOUT}s: $m --selftest"
    die "$m --selftest exited $rc, so the suite did NOT complete; any PASS \
lines it managed to print are a PARTIAL count and must not be reported as one. \
stderr tail: ${tail_:-<empty>}"
  fi
  COUNT="$(printf '%s\n' "$out" | grep -c "^  PASS ")"
}

# ---- self-verification (rule 15: a positive control AND a known-bad) --------
if [ "${1:-}" = "--selftest" ]; then
  d="$(mktemp -d)"; fails=0
  chk() { # label expected_regex actual
    if printf '%s' "$3" | grep -qE "$2"; then printf '  PASS %s\n' "$1"
    else printf '  FAIL %s -- got: %s\n' "$1" "$3"; fails=1; fi; }

  printf 'print("  PASS a");print("  PASS b");print("  PASS c")\n' >"$d/good.py"
  printf 'print("  PASS a")\nimport sys;sys.exit(1)\n'             >"$d/partial.py"
  printf 'print("nothing here")\n'                                 >"$d/zero.py"
  printf 'import time;time.sleep(30)\n'                            >"$d/hang.py"

  chk "positive control: a complete suite yields its count" \
      '^3$'                  "$("$0" "$d/good.py" 2>&1)"
  chk "known-bad: a suite that CRASHED after printing PASS is refused" \
      'ERROR_NOT_A_COUNT.*exited 1' "$("$0" "$d/partial.py" 2>&1)"
  chk "known-bad: zero PASS lines is refused, never reported as 0" \
      'ERROR_NOT_A_COUNT.*ZERO'     "$("$0" "$d/zero.py" 2>&1)"
  chk "known-bad: a hung suite is refused as TIMED OUT, not counted" \
      'ERROR_NOT_A_COUNT.*TIMED OUT' "$(FC_TIMEOUT=2 "$0" "$d/hang.py" 2>&1)"
  chk "known-bad: a missing module is refused" \
      'ERROR_NOT_A_COUNT.*no such'  "$("$0" "$d/nope.py" 2>&1)"
  rm -rf "$d"
  [ "$fails" -eq 0 ] || { printf 'FALSIFIER_COUNT SELFTEST FAILED\n'; exit 1; }
  printf 'falsifier_count.sh selftest: all checks passed\n'; exit 0
fi

[ $# -ge 1 ] || die "usage: falsifier_count.sh <module.py> [--ref] | --selftest"
m="$1"
count_of "$m"; n="$COUNT"
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
