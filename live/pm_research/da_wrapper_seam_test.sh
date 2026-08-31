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
  local sv=""
  [ "$mk" = 1 ] && sv='{"gate":"da_state_tape_verify_v1","all_pass":true}'
  DA_GATE_STUB=1 DA_STUB_GATE_RC=0 DA_STUB_COUNT_RC=0 DA_STUB_VERDICT="$sv" \
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
  # R-228: the stub WRITES, exactly as the real gate does. Pre-seeding the
  # staging path is indistinguishable from the fail-open it would mask -- that
  # is how this suite passed 13/13 while a stale file was being promoted.
  DA_GATE_STUB=1 DA_STUB_GATE_RC="$g" DA_STUB_COUNT_RC="$c" \
    DA_STUB_VERDICT='{"gate":"da_state_tape_verify_v1","all_pass":true}' \
    DA_LOG="$SEAM_LOG" DA_TAPE=/dev/null DA_SKIP_WAIT=1 \
    DA_VERDICT_RULED="$ruled" "$W" >/dev/null 2>&1
  local got=0; [ -f "$ruled" ] && got=1
  if [ "$got" = "$want" ]; then pass=$((pass+1)); echo "  OK   $label"
  else fail=$((fail+1)); echo "  FAIL $label: locator exists=$got want=$want"; fi
  rm -rf "$d"
}

echo "seam 22d -- R-228 fail-open (the user's known-bad, verbatim):"
r228() {
  local d; d="$(mktemp -d)"; local ruled="$d/da_tape_gate_verdict_v5.json"
  # a STALE staging file, and checkers that exit 0 having written NOTHING
  printf '{"stale":true}' > "$ruled.staging"
  DA_GATE_STUB=1 DA_STUB_GATE_RC=0 DA_STUB_COUNT_RC=0 \
    DA_LOG="$SEAM_LOG" DA_TAPE=/dev/null DA_SKIP_WAIT=1 \
    DA_VERDICT_RULED="$ruled" "$W" >/dev/null 2>&1
  if [ -f "$ruled" ]; then
    fail=$((fail+1)); echo "  FAIL a STALE staging file was PROMOTED by checkers that wrote nothing: $(cat "$ruled")"
  else
    pass=$((pass+1)); echo "  OK   a stale staging file is NEVER promoted -- the verdict must be a file THIS RUN created"
  fi
  # and the orphan must be gone, with the cause logged
  if [ -e "$ruled.staging" ]; then
    fail=$((fail+1)); echo "  FAIL the orphan staging file survived"
  else
    pass=$((pass+1)); echo "  OK   the orphan staging file is removed, with its cause logged"
  fi
  rm -rf "$d"
}
r228

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

# ==========================================================================
# da_midnight_verify.sh: rc GOVERNS the install, it is not merely logged.
# The install decision rested on the artifact's SHAPE alone while `rc` was
# captured and only printed -- so a verifier exiting non-zero that still left
# a well-formed file would have been installed under a log line reading
# "verdict artifact written" beside "exit=4". A gate never seen to fail is not
# evidence, so each case drives a STUB verifier that writes a good artifact
# and exits with the code under test.
# ==========================================================================
echo "seam 23 -- midnight rc gate (stub verifier writes a GOOD artifact):"
M="/home/yuqing/ctaNew/live/pm_research/da_midnight_verify.sh"
rcgate() {  # rcgate <label> <stub_rc> <want_installed 0|1>
  local label="$1" srb="$2" want="$3"
  local d; d="$(mktemp -d)"
  cat > "$d/stub.py" <<STUB
import json, sys
a = sys.argv
out = a[a.index("--out") + 1]
day = a[a.index("--day") + 1]
json.dump({"day_token": day, "predicates": [{"predicate": "x", "pass": True}],
           "all_pass": True, "day_closed_calendar": True,
           "as_of_utc": "2026-08-31T00:00:00+00:00"}, open(out, "w"))
sys.exit($srb)
STUB
  DA_MIDNIGHT_LOG="$d/log" DA_MIDNIGHT_OUTDIR="$d/out" \
    DA_MIDNIGHT_VERIFY_BIN="$d/stub.py" bash "$M" >/dev/null 2>&1
  local got=0; ls "$d/out"/da_dayverdict_*.json >/dev/null 2>&1 && got=1
  if [ "$got" = "$want" ]; then pass=$((pass+1)); echo "  OK   $label"
  else fail=$((fail+1)); echo "  FAIL $label: installed=$got want=$want"; fi
  rm -rf "$d"
}
rcgate "rc=4 (instrument failure) is NOT installed even though the artifact is well-formed" 4 0
rcgate "rc=3 (an exit code that is not a verdict outcome) is NOT installed" 3 0
rcgate "POSITIVE CONTROL: rc=0 installs" 0 1
rcgate "POSITIVE CONTROL: rc=1 (day FAILED its bars) is a verdict and installs" 1 1

echo "seam 24 -- a stub verifier may not run half-isolated:"
d="$(mktemp -d)"; : > "$d/stub.py"
if DA_MIDNIGHT_VERIFY_BIN="$d/stub.py" DA_MIDNIGHT_LOG="$d/log" bash "$M" >/dev/null 2>&1; then
  fail=$((fail+1)); echo "  FAIL a stub verifier ran without DA_MIDNIGHT_OUTDIR"
else
  pass=$((pass+1)); echo "  OK   a stub verifier without BOTH overrides REFUSES -- it would write production verdicts"
fi
rm -rf "$d"

# THE RUNNER MUST BE ABLE TO FAIL. This file had NO exit statement: its status
# was whatever the last command returned -- an `echo`, then an `rm` -- so it
# reported "N failed" in text and exited 0 regardless. Every green run of it,
# including as a gate in an all-gates sweep, was evidence of nothing. A gate
# that cannot fail is not a gate.
echo "seam total: $pass passed, $fail failed"
if [ "$fail" -ne 0 ]; then
  echo "SEAM TEST FAILED ($fail case(s))" >&2
  exit 1
fi
exit 0
