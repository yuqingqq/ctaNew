#!/bin/bash
# SEAM 22: does the await-gate wrapper PROPAGATE a checker's refusal?
#
# RED-FIRST EVIDENCE (2026-08-27, user audit #4): it did not. The wrapper ran
# each checker, echoed `GATE_EXIT=$?` into its log, and then exited with the
# status of that final echo -- zero. So a REFUSAL was recorded in the log and
# reported to systemd as `Result=success`. The failure mode looked exactly like
# the success mode, which is the defect class this seat has spent the day
# filing against other people's instruments.
#
# It never gated a real artifact -- this morning's drill invoked the checker
# directly, which is why its exit 1 was real -- but it was armed to gate tape5,
# and tape5 would have been refused.
#
#   ./da_wrapper_seam_test.sh          # asserts the wrapper propagates
set -u
W=/home/yuqing/ctaNew/live/pm_research/da_await_gate.sh
# ITS OWN LOG. This test drives the wrapper with STUB exit codes, and it used
# to inherit the wrapper's DEFAULT log -- the production one. Six stub runs
# therefore landed underneath a LIVE "armed" header while the real gate was
# still polling for its tape, and the first person to read it (me) concluded
# the armed gate had run and exited 3. A log that cannot distinguish "the gate
# ran" from "a test of the gate ran" is not evidence, and this log exists to
# carry a refusal.
SEAM_LOG="$(mktemp -t da_seam_log.XXXXXX)"
trap 'rm -f "$SEAM_LOG"' EXIT
pass=0; fail=0
check() {  # check <label> <expected_rc> <gate_rc> <count_rc>
  local label="$1" want="$2" g="$3" c="$4" got
  DA_GATE_STUB=1 DA_STUB_GATE_RC="$g" DA_STUB_COUNT_RC="$c" \
    DA_LOG="$SEAM_LOG" \
    DA_TAPE=/dev/null DA_SKIP_WAIT=1 "$W" >/dev/null 2>&1
  got=$?
  if [ "$got" = "$want" ]; then
    pass=$((pass+1)); echo "  OK   $label (rc=$got)"
  else
    fail=$((fail+1)); echo "  FAIL $label: expected $want got $got"
  fi
}
echo "seam 22 -- wrapper exit-code propagation:"
check "both checkers pass  -> 0"                0 0 0
check "GATE refuses        -> nonzero"          1 1 0
check "COUNT refuses       -> nonzero"          1 0 1
check "both refuse         -> nonzero"          1 1 1
check "gate rc 2 preserved -> WORST wins"       2 2 1
check "count rc 3 preserved-> WORST wins"       3 1 3
echo
echo "seam22: $pass passed, $fail failed"
[ "$fail" = 0 ]
