#!/bin/bash
# DA: wait for a tape, run the pre-registered checks, and PROPAGATE the verdict.
#
# SURFACE AUTHORISATION (R-126): R-199 item 1 (wrapper exit propagation),
# item 3 (verdict artifact). Supersedes da_await_tape5_gate.sh, whose defect is
# committed red at b19af5c and asserted by da_wrapper_seam_test.sh.
#
# THE DEFECT THIS REPLACES. The old wrapper echoed `GATE_EXIT=$?` into its log
# and then exited with the status of that echo -- zero. A refusal was written
# down and reported to systemd as success. Here every checker's status is
# tracked and the script exits with the WORST of them, so a refusal cannot be
# swallowed by the thing carrying it.
#
# A UNIT, NOT A SESSION JOB: every session-scoped watch armed in this programme
# was killed mid-poll. Size-stability is required because a build writing its
# final path directly means presence is not completeness.
#
#   DA_TAPE=<path> DA_EXPECT_COUNT=289 DA_EXPECT_PROV=<ref> \
#   DA_GAPPED_SLUGS=133 DA_VERDICT_OUT=<path> ./da_await_gate.sh
set -u
D=/home/yuqing/ctaNew/live/pm_research
PY=/home/yuqing/pricer-sol/venv/bin/python3
T="${DA_TAPE:?DA_TAPE is required; refusing to guess a tape}"
LOG="${DA_LOG:-/home/yuqing/ctaNew/data/pm_5min/derived/.da_tape_gate.log}"
EXPECT_COUNT="${DA_EXPECT_COUNT:-}"
EXPECT_PROV="${DA_EXPECT_PROV:-}"
GAPPED="${DA_GAPPED_SLUGS:-133}"
# R-213(4): BOTH checkers must read the SAME pinned ledger the builder used.
# A gate reading the live ledger while the tape was built from a snapshot is
# comparing against a different gap population.
LEDGER="${DA_LEDGER:-}"
LEDGER_SHA="${DA_EXPECT_LEDGER_SHA:-}"
# R-212(a/b): THE CONSUMER'S LOCATOR IS THE PRIMARY OUTPUT, not a pin-named
# file someone must know to look for. `phase2_arms.DA_VERDICT` (:346) resolves
# exactly this path and REFUSES when it is absent -- so a verdict written under
# a different name is a verdict that does not exist, which is the name-drift
# class in its third instance (after tape_non_empty/dataset_non_empty and the
# LOAD_BEARING six-vs-seven). Verified at the consumer, not taken from a name.
#
# Written to a temp and PROMOTED only if the gate actually produced a verdict:
# the gate REFUSES to write when a load-bearing predicate was not asserted, and
# a refusal must leave the locator ABSENT so the consumer refuses too. A
# half-written file at the ruled name would authorise fitting by accident.
RULED_VERDICT="${DA_VERDICT_RULED:-/home/yuqing/ctaNew/data/pm_5min/derived/da_tape_gate_verdict_v5.json}"
VERDICT="${DA_VERDICT_OUT:-$RULED_VERDICT}"
ARCHIVE="${DA_VERDICT_ARCHIVE:-}"
# R-214: the gate writes HERE first. The verdict reaches the ruled locator only
# after BOTH checkers have spoken, so a refusal by EITHER leaves the locator
# absent. Previously promotion happened at the end of run_gate, and
# assert_gate_passed ACCEPTED while the independent counter was still running --
# fitting was permitted on the gate alone. An exit code nobody consumes is not
# a gate.
# R-228 FAIL-OPEN, found by the user: this was a FIXED name, never removed and
# never refused at startup. A leftover staging file from any prior run -- or a
# hand-placed one -- was promoted verbatim by checkers that exited 0 WITHOUT
# WRITING ANYTHING. Verified: {"stale":true} reached the ruled locator.
# The verdict must be a file THIS RUN created, so the name is unique per run
# and removed on exit. A stale file can no longer be mistaken for this run's
# output because this run's output did not exist until mktemp made it.
STAGING="$(mktemp "${RULED_VERDICT}.staging.XXXXXX")"
rm -f "$STAGING"          # mktemp reserves the NAME; the gate must create the file
trap 'rm -f "$STAGING"' EXIT
# Orphans from crashed runs cannot be promoted (they are not this run's path),
# but they accumulate and hide, so they are removed with the cause LOGGED --
# never silently, and the ruled locator itself is never touched.
for _orphan in "${RULED_VERDICT}".staging*; do
  [ -e "$_orphan" ] || continue
  [ "$_orphan" = "$STAGING" ] && continue
  echo "REMOVED ORPHAN STAGING $_orphan ($(stat -c%s "$_orphan" 2>/dev/null) bytes) -- a leftover from a run that did not finish; it is NOT this run's output and was never promotable" >> "$LOG"
  rm -f "$_orphan"
done
DEADLINE=$(( $(date +%s) + ${DA_DEADLINE_S:-14400} ))
# A STALE verdict at the ruled locator defeats the entire absence contract: a
# refusal would leave the PREVIOUS run's verdict resolving, and only the
# consumer's sha check would stand between it and a wrong authorisation. DA had
# to clear one by hand before tape6e. Refuse instead, and make the operator
# archive it deliberately.
if [ -e "$RULED_VERDICT" ] && [ "${DA_ALLOW_OCCUPIED_LOCATOR:-0}" != "1" ]; then
  echo "REFUSED: $RULED_VERDICT already exists. A stale verdict there means a" >&2
  echo "  refusal would still leave a resolvable verdict. Archive or rename it" >&2
  echo "  first (deliberately), then re-arm." >&2
  exit 6
fi

worst=0
track() {  # track <rc>
  [ "$1" -gt "$worst" ] && worst="$1"
  return 0
}

run_gate() {
  if [ "${DA_GATE_STUB:-0}" = "1" ]; then
    # A stub that WRITES writes to THIS RUN's staging path, exactly as the real
    # gate does. Tests must never pre-seed the staging file: doing so is
    # indistinguishable from the R-228 defect, which is why the suite passed
    # 13/13 while the defect was live.
    [ -n "${DA_STUB_VERDICT:-}" ] && printf '%s' "$DA_STUB_VERDICT" > "$STAGING"
    return "${DA_STUB_GATE_RC:-0}"
  fi
  local args=(verify --tape "$T" --gapped-slugs "$GAPPED")
  [ -n "$LEDGER" ] && args+=(--ledger "$LEDGER")
  [ -n "$LEDGER_SHA" ] && args+=(--expect-ledger-sha "$LEDGER_SHA")
  [ -n "$EXPECT_COUNT" ] && args+=(--expect-gap-count "$EXPECT_COUNT")
  [ -n "$EXPECT_PROV" ] && args+=(--expect-provenance "$EXPECT_PROV")
  args+=(--verdict-out "$STAGING")
  "$PY" "$D/da_state_tape_verify.py" "${args[@]}" >> "$LOG" 2>&1
}

run_count() {
  if [ "${DA_GATE_STUB:-0}" = "1" ]; then return "${DA_STUB_COUNT_RC:-0}"; fi
  local cargs=(count --tape "$T")
  [ -n "$LEDGER" ] && cargs+=(--ledger "$LEDGER")
  "$PY" "$D/da_gap_at_cutoff_count.py" "${cargs[@]}" >> "$LOG" 2>&1
}

if [ "${DA_SKIP_WAIT:-0}" != "1" ]; then
  { echo; echo "======== armed $(date -u +%FT%TZ) for $T ========"; } >> "$LOG"
  while [ ! -f "$T" ]; do
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
      echo "TIMEOUT: tape never appeared" >> "$LOG"; exit 2
    fi
    sleep 20
  done
  echo "appeared $(date -u +%FT%TZ); awaiting size stability" >> "$LOG"
  prev=-1; stable=0
  while [ "$stable" -lt 3 ]; do
    sleep 15
    cur=$(stat -c%s "$T" 2>/dev/null || echo -1)
    if [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ]; then stable=$((stable+1)); else stable=0; fi
    prev=$cur
    if [ "$(date +%s)" -gt "$DEADLINE" ]; then
      echo "TIMEOUT awaiting stability" >> "$LOG"; exit 2
    fi
  done
  echo "stable at $prev bytes $(date -u +%FT%TZ)" >> "$LOG"
fi

run_gate;  rc=$?; track "$rc"; echo "GATE_EXIT=$rc"  >> "$LOG"
run_count; rc=$?; track "$rc"; echo "COUNT_EXIT=$rc" >> "$LOG"
echo "WORST_EXIT=$worst  ($(date -u +%FT%TZ))" >> "$LOG"

# R-214 PROMOTION, after BOTH checkers. Three outcomes, all logged:
#  worst != 0            -> staging DISCARDED, locator absent, fit blocked
#  worst == 0, no verdict -> the gate refused to write; locator absent
#  worst == 0, verdict    -> promoted atomically, archive copy optional
if [ "$worst" -ne 0 ]; then
  rm -f "$STAGING"
  echo "NOT PROMOTED: worst rc $worst -- a refusal by EITHER checker leaves the ruled locator ABSENT" >> "$LOG"
elif [ ! -s "$STAGING" ]; then
  echo "NOT PROMOTED: both checkers passed but the gate wrote NO verdict (refuse-to-write); locator stays absent" >> "$LOG"
else
  vtmp="$(mktemp "${RULED_VERDICT}.XXXXXX")"
  if cp "$STAGING" "$vtmp" && mv -f "$vtmp" "$RULED_VERDICT"; then
    rm -f "$STAGING"
    echo "PROMOTED -> $RULED_VERDICT (both checkers rc 0; content unchanged)" >> "$LOG"
    if [ -n "$ARCHIVE" ]; then
      cp "$RULED_VERDICT" "$ARCHIVE" && echo "archived -> $ARCHIVE" >> "$LOG"
    fi
  else
    rm -f "$vtmp"; track 5
    echo "PROMOTION FAILED -- the ruled locator does NOT resolve; fitting must not start" >> "$LOG"
  fi
fi
exit "$worst"
