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
  # ISOLATED LOCATOR. Without this the propagation cases inherit the
  # PRODUCTION ruled path, which exists after a real run -- so they hit the
  # R-214 occupied-locator refusal (rc 6) and test nothing about propagation.
  # Same isolation defect as the seam log writing into the production gate log.
  local rdir; rdir="$(mktemp -d)"
  DA_GATE_STUB=1 DA_STUB_GATE_RC="$g" DA_STUB_COUNT_RC="$c" \
    DA_LOG="$SEAM_LOG" DA_VERDICT_RULED="$rdir/verdict.json" \
    DA_TAPE=/dev/null DA_SKIP_WAIT=1 "$W" >/dev/null 2>&1
  got=$?
  rm -rf "$rdir"
  if [ "$got" = "$want" ]; then
    pass=$((pass+1)); echo "  OK   $label (rc=$got)"
  else
    fail=$((fail+1)); echo "  FAIL $label: expected $want got $got"
  fi
}
# ---- R-212: the ruled locator must resolve, and ONLY on a real verdict -----
promo() {  # promo <label> <expect_ruled_exists 0|1> <make_source 0|1>
  local label="$1" want="$2" mk="$3"
  local d; d="$(mktemp -d)"
  local src="$d/pin_named.json" ruled="$d/da_tape_gate_verdict_v5.json"
  [ "$mk" = 1 ] && printf '{"gate":"da_state_tape_verify_v1","all_pass":true}' > "$ruled.staging"
  DA_GATE_STUB=1 DA_STUB_GATE_RC=0 DA_STUB_COUNT_RC=0 \
    DA_LOG="$SEAM_LOG" DA_TAPE=/dev/null DA_SKIP_WAIT=1 \
    DA_VERDICT_RULED="$ruled" "$W" >/dev/null 2>&1
  local got=0; [ -f "$ruled" ] && got=1
  if [ "$got" = "$want" ]; then
    pass=$((pass+1)); echo "  OK   $label"
  else
    fail=$((fail+1)); echo "  FAIL $label: ruled locator exists=$got want=$want"
  fi
  rm -rf "$d"
}

# ---- R-214: promotion happens only after BOTH checkers pass ---------------
promo2() {  # promo2 <label> <want_locator 0|1> <gate_rc> <count_rc>
  local label="$1" want="$2" g="$3" c="$4"
  local d; d="$(mktemp -d)"
  local ruled="$d/da_tape_gate_verdict_v5.json"
  # a stub run writes no verdict, so pre-seed the STAGING path the gate targets
  printf '{"gate":"da_state_tape_verify_v1","all_pass":true}' > "$ruled.staging"
  DA_GATE_STUB=1 DA_STUB_GATE_RC="$g" DA_STUB_COUNT_RC="$c" \
    DA_LOG="$SEAM_LOG" DA_TAPE=/dev/null DA_SKIP_WAIT=1 \
    DA_VERDICT_RULED="$ruled" "$W" >/dev/null 2>&1
  local got=0; [ -f "$ruled" ] && got=1
  if [ "$got" = "$want" ]; then pass=$((pass+1)); echo "  OK   $label"
  else fail=$((fail+1)); echo "  FAIL $label: locator exists=$got want=$want"; fi
  rm -rf "$d"
}

echo "seam 22c -- R-214 promote only after BOTH checkers:"
promo2 "both checkers pass -> verdict IS promoted" 1 0 0
promo2 "GATE refuses -> locator ABSENT (no promotion)" 0 1 0
promo2 "COUNT refuses -> locator ABSENT: the second checker can veto, which is \
the race R-214 closes -- the verdict used to publish before it had spoken" 0 0 1
promo2 "both refuse -> locator ABSENT" 0 1 1

occupied() {  # the wrapper must refuse to arm onto an occupied locator
  local d; d="$(mktemp -d)"; local ruled="$d/da_tape_gate_verdict_v5.json"
  printf '{"stale":true}' > "$ruled"
  DA_GATE_STUB=1 DA_STUB_GATE_RC=0 DA_STUB_COUNT_RC=0 \
    DA_LOG="$SEAM_LOG" DA_TAPE=/dev/null DA_SKIP_WAIT=1 \
    DA_VERDICT_RULED="$ruled" "$W" >/dev/null 2>&1
  local rc=$?
  local kept=0; grep -q '"stale"' "$ruled" 2>/dev/null && kept=1
  if [ "$rc" = "6" ] && [ "$kept" = "1" ]; then
    pass=$((pass+1)); echo "  OK   an OCCUPIED locator REFUSES (rc 6) and the stale verdict is left intact"
  else
    fail=$((fail+1)); echo "  FAIL occupied-locator: rc=$rc kept=$kept (want rc=6 kept=1)"
  fi
  rm -rf "$d"
}
occupied

echo "seam 22b -- R-212 ruled-locator promotion:"
promo "a written verdict is PROMOTED to the ruled locator the consumer reads" 1 1
promo "a REFUSED verdict (no file) leaves the locator ABSENT -- the gate's \
refuse-to-write must not be undone by the bridge that copies it" 0 0

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
