#!/usr/bin/env bash
# DA round 52: does the midnight unit's rc still turn RED on a real failure?
#
# THE DEFECT THIS GUARDS. `da-midnight-verify.service` exited 4 EVERY night --
# 2026-09-05T00:06:29Z and 2026-09-06T00:06:41Z, measured in its own log --
# because `days_needing_verdict` always carries the day that just began, whose
# mask the frozen detector correctly refuses six minutes after midnight. On
# both nights the CLOSED day was perfect. An alarm that fires every night is an
# alarm that gets turned off, so the open-day refusal now carries rc 2
# (OPEN_DAY_MASK_DEFERRED) and the unit declares SuccessExitStatus=2.
#
# THE RISK THAT CREATES, and the only reason this file exists: a fix that makes
# a red light green is one mistake away from making EVERY light green. So this
# plants a REAL mask failure on a CLOSED day and requires FAILURE, in the same
# run that requires DEFERRED for the open day. Both directions, every time.
#
# It drives `classify_mask_failure` from the production script by sourcing its
# definition -- the SAME text that runs at 00:06Z, not a copy of it. A copy
# would drift and this would pass while the unit misclassified.
#
#   bash live/pm_research/da_midnight_rc_falsifier.sh
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/da_midnight_verify.sh"
[ -r "$SRC" ] || { echo "REFUSED: no $SRC"; exit 2; }

# Extract the function from the production script by name and eval it. If the
# function is renamed or deleted this REFUSES rather than silently testing
# nothing -- the empty-set trap on a falsifier.
FN="$(awk '/^classify_mask_failure\(\) \{/,/^\}/' "$SRC")"
case "$FN" in
  *classify_mask_failure*) : ;;
  *) echo "REFUSED: classify_mask_failure not found in $SRC. A falsifier that"\
          "cannot find its subject must not report that the subject passed."
     exit 2 ;;
esac
eval "$FN"

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
fails=0
say() { if [ "$1" = "$2" ]; then echo "ok   $3"; else echo "FAIL $3 (got $2, want $1)"; fails=$((fails+1)); fi; }

# The refusal logs, in the STRUCTURED form the builder now emits: one token
# on a line of its own, written only by `da_blackout_mask.main`.
printf '%s\n' "MASK_STATUS=CONTENT_LIVENESS_UNJUDGEABLE" > "$TMP/unjudgeable"
printf '%s\n' "MASK_STATUS=CONTENT_LIVENESS_UNRESOLVED" > "$TMP/unresolved"
printf '%s\n' "Traceback (most recent call last):" "OSError: [Errno 28] No space left on device" > "$TMP/realfail"
printf '%s\n' "" > "$TMP/empty"
# THE CASE THE REVIEWER DROVE AND THIS BATTERY DID NOT (C-1): a log carrying
# BOTH the liveness token AND a real failure. Under the old prose match this
# returned DEFERRED -- a disk-full night reported as success.
printf '%s\n' "MASK_STATUS=CONTENT_LIVENESS_UNJUDGEABLE" \
  "Traceback (most recent call last):" \
  "RuntimeError: while handling CONTENT_LIVENESS_UNJUDGEABLE the writer died" \
  "OSError: [Errno 28] No space left on device" > "$TMP/mixed"
# The same failure WITHOUT the structured line, only the token in prose --
# which is exactly what the old matcher accepted.
printf '%s\n' "Traceback (most recent call last):" \
  "RuntimeError: while handling CONTENT_LIVENESS_UNJUDGEABLE the writer died" \
  > "$TMP/prose_only"
# A token quoted mid-line must not satisfy the line-anchored match.
printf '%s\n' "note: MASK_STATUS=CONTENT_LIVENESS_UNJUDGEABLE was seen" \
  > "$TMP/inline"

echo "== the nightly case: an OPEN day refusing for want of windows =="
say DEFERRED "$(classify_mask_failure 0 "$TMP/unjudgeable" 1)" \
  "OPEN + UNJUDGEABLE -> DEFERRED (tonight, 2026-09-06T00:06:41Z)"
say DEFERRED "$(classify_mask_failure 0 "$TMP/unresolved" 1)" \
  "OPEN + UNRESOLVED  -> DEFERRED (last night, 2026-09-05T00:06:29Z)"

echo "== THE FALSIFIER: a REAL mask failure on a CLOSED day must still be rc 4 =="
say FAILURE "$(classify_mask_failure 1 "$TMP/unjudgeable" 1)" \
  "CLOSED + UNJUDGEABLE -> FAILURE: the planted case. A closed day whose mask \
cannot be built is UNSCOREABLE and stays red, whatever the reason text says"
say FAILURE "$(classify_mask_failure 1 "$TMP/unresolved" 1)" \
  "CLOSED + UNRESOLVED  -> FAILURE"
say FAILURE "$(classify_mask_failure 1 "$TMP/realfail" 1)" \
  "CLOSED + a disk error -> FAILURE"

echo "== and an OPEN day failing for any OTHER reason is still rc 4 =="
say FAILURE "$(classify_mask_failure 0 "$TMP/realfail" 1)" \
  "OPEN + a disk error -> FAILURE: the two conjuncts are ANDed, so the \
deferral cannot swallow a real failure that lands on an open day"
say FAILURE "$(classify_mask_failure 0 "$TMP/empty" 1)" \
  "OPEN + NO reason text at all -> FAILURE: silence is not a deferral"

echo "== an unreadable verdict is not an open day =="
say FAILURE "$(classify_mask_failure '?' "$TMP/unjudgeable" 1)" \
  "UNKNOWN closed-ness -> FAILURE: '?' is not '0', so a verdict that cannot \
be read falls to the safe side"

echo "== C-1: the mixed log, in both directions =="
say FAILURE "$(classify_mask_failure 0 "$TMP/mixed" 1)" \
  "OPEN + structured refusal AND a real disk-full failure -> FAILURE. This is \
the exact log the reviewer drove; under the prose match it returned DEFERRED \
and systemctl reported success on a broken night"
say FAILURE "$(classify_mask_failure 0 "$TMP/prose_only" 1)" \
  "OPEN + the token ONLY IN PROSE, no structured line -> FAILURE: a message \
that mentions the token cannot forge one the builder never emitted"
say FAILURE "$(classify_mask_failure 0 "$TMP/inline" 1)" \
  "OPEN + the token quoted MID-LINE -> FAILURE: the match is anchored to the \
line start, so a mention inside other text does not satisfy it"
say DEFERRED "$(classify_mask_failure 0 "$TMP/unjudgeable" 1)" \
  "OPEN + a PURE structured refusal, no failure marker -> DEFERRED: the \
tightening did not close the path it exists to allow"

echo "== A-1: the SIGKILLed builder, in both directions =="
# The reviewer's demonstration: a builder that PRINTED the token and was then
# SIGKILLed leaves a log with the token ALONE -- `Killed` is written by the
# PARENT SHELL to its own stderr and never enters the redirected child log.
# So the log is INDISTINGUISHABLE from an honest refusal and only the exit
# code separates them.
say DEFERRED "$(classify_mask_failure 0 "$TMP/unjudgeable" 1)" \
  "OPEN + token + rc 1 -> DEFERRED: rc 1 is the builder's OWN refusal return, \
so the path the deferral exists for is still open"
say FAILURE "$(classify_mask_failure 0 "$TMP/unjudgeable" 137)" \
  "OPEN + token + rc 137 (SIGKILL) -> FAILURE. The log is IDENTICAL to the \
honest refusal above -- byte for byte, same file -- and only the exit code \
tells them apart. This is the reviewer's A-1, and under the previous \
classifier it DEFERRED"
say FAILURE "$(classify_mask_failure 0 "$TMP/unjudgeable" 139)" \
  "OPEN + token + rc 139 (SIGSEGV) -> FAILURE"
say FAILURE "$(classify_mask_failure 0 "$TMP/unjudgeable" 143)" \
  "OPEN + token + rc 143 (SIGTERM) -> FAILURE"
say FAILURE "$(classify_mask_failure 0 "$TMP/unjudgeable" 2)" \
  "OPEN + token + rc 2 -> FAILURE: the test is rc EXACTLY 1, not merely \
non-zero-and-small, because only 1 is the refusal return"
say FAILURE "$(classify_mask_failure 0 "$TMP/unjudgeable" 0)" \
  "OPEN + token + rc 0 -> FAILURE: a builder that SUCCEEDED did not refuse, \
so a token beside rc 0 is incoherent and is not a deferral"

echo "== the guard moves with its input, in both directions =="
say DEFERRED "$(classify_mask_failure 0 "$TMP/unjudgeable" 1)" \
  "same input, same answer -- deterministic"
say FAILURE "$(classify_mask_failure 0 1 "$TMP/nonexistent-file")" \
  "a missing mask log -> FAILURE, never a deferral"

# ROUND 54: THE SUMMARY IS COMPUTED, NOT ASSERTED. It used to echo
# "rc 2 is reachable ONLY for an open day refusing for want of windows" --
# a conclusion printed beside a passing test set, which is rule 10 in a shell
# echo, and the coordinator quoted it as verification in R-542(A). The
# reviewer then reached rc 2 by another route, so the sentence was false while
# every case passed. What the battery can honestly say is a COUNT over its own
# outcomes: how many inputs reached DEFERRED, and which.
echo
echo "== the summary, COMPUTED over this battery's own outcomes =="
n_def=0; n_fail=0; deferring=""
for f in unjudgeable unresolved realfail empty mixed prose_only inline; do
  for c in 0 1 '?'; do
    for rc in 1 0 2 137 139 143; do
      r="$(classify_mask_failure "$c" "$TMP/$f" "$rc")"
      if [ "$r" = "DEFERRED" ]; then
        n_def=$((n_def+1)); deferring="$deferring closed=$c/rc=$rc/$f"
      else
        n_fail=$((n_fail+1))
      fi
    done
  done
done
echo "   inputs driven: $((n_def+n_fail))   DEFERRED: $n_def   FAILURE: $n_fail"
echo "   every input that reached DEFERRED:$deferring"
want=" closed=0/rc=1/unjudgeable closed=0/rc=1/unresolved"
say "$want" "$deferring" \
  "COMPUTED PREDICATE: across every (closed-state x exit-code x log) triple \
this battery drives, the ONLY ones reaching DEFERRED are the two pure \
structured open-day refusals AT rc 1 -- enumerated, not asserted. Every \
signal death carrying the same token reaches FAILURE."
echo
if [ "$fails" -eq 0 ]; then
  echo "da_midnight_rc_falsifier: $((n_def+n_fail)) drives + 21 named checks \
PASSED"
  exit 0
fi
echo "da_midnight_rc_falsifier: $fails FAILURE(S)"; exit 1
