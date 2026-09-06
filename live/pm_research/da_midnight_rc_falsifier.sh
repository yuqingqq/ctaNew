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

# The two real refusal texts, copied from the unit's own log.
printf '%s\n' "MaskRefused: REFUSED: the frozen detector reports CONTENT_LIVENESS_UNJUDGEABLE for 20260906 (no coin had enough windows for a median)." > "$TMP/unjudgeable"
printf '%s\n' "MaskRefused: REFUSED: the frozen detector reports CONTENT_LIVENESS_UNRESOLVED for 20260905 (20260905 has a raw directory but NO window files)." > "$TMP/unresolved"
printf '%s\n' "Traceback (most recent call last):" "OSError: [Errno 28] No space left on device" > "$TMP/realfail"
printf '%s\n' "" > "$TMP/empty"

echo "== the nightly case: an OPEN day refusing for want of windows =="
say DEFERRED "$(classify_mask_failure 0 "$TMP/unjudgeable")" \
  "OPEN + UNJUDGEABLE -> DEFERRED (tonight, 2026-09-06T00:06:41Z)"
say DEFERRED "$(classify_mask_failure 0 "$TMP/unresolved")" \
  "OPEN + UNRESOLVED  -> DEFERRED (last night, 2026-09-05T00:06:29Z)"

echo "== THE FALSIFIER: a REAL mask failure on a CLOSED day must still be rc 4 =="
say FAILURE "$(classify_mask_failure 1 "$TMP/unjudgeable")" \
  "CLOSED + UNJUDGEABLE -> FAILURE: the planted case. A closed day whose mask \
cannot be built is UNSCOREABLE and stays red, whatever the reason text says"
say FAILURE "$(classify_mask_failure 1 "$TMP/unresolved")" \
  "CLOSED + UNRESOLVED  -> FAILURE"
say FAILURE "$(classify_mask_failure 1 "$TMP/realfail")" \
  "CLOSED + a disk error -> FAILURE"

echo "== and an OPEN day failing for any OTHER reason is still rc 4 =="
say FAILURE "$(classify_mask_failure 0 "$TMP/realfail")" \
  "OPEN + a disk error -> FAILURE: the two conjuncts are ANDed, so the \
deferral cannot swallow a real failure that lands on an open day"
say FAILURE "$(classify_mask_failure 0 "$TMP/empty")" \
  "OPEN + NO reason text at all -> FAILURE: silence is not a deferral"

echo "== an unreadable verdict is not an open day =="
say FAILURE "$(classify_mask_failure '?' "$TMP/unjudgeable")" \
  "UNKNOWN closed-ness -> FAILURE: '?' is not '0', so a verdict that cannot \
be read falls to the safe side"

echo "== the guard moves with its input, in both directions =="
say DEFERRED "$(classify_mask_failure 0 "$TMP/unjudgeable")" \
  "same input, same answer -- deterministic"
say FAILURE "$(classify_mask_failure 0 "$TMP/nonexistent-file")" \
  "a missing mask log -> FAILURE, never a deferral"

echo
if [ "$fails" -eq 0 ]; then
  echo "da_midnight_rc_falsifier: 10 checks PASSED -- rc 2 is reachable ONLY \
for an open day refusing for want of windows, and rc 4 still fires on \
everything else"
  exit 0
fi
echo "da_midnight_rc_falsifier: $fails FAILURE(S)"; exit 1
