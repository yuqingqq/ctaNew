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
DEADLINE=$(( $(date +%s) + ${DA_DEADLINE_S:-14400} ))

worst=0
track() {  # track <rc>
  [ "$1" -gt "$worst" ] && worst="$1"
  return 0
}

run_gate() {
  if [ "${DA_GATE_STUB:-0}" = "1" ]; then return "${DA_STUB_GATE_RC:-0}"; fi
  local args=(verify --tape "$T" --gapped-slugs "$GAPPED")
  [ -n "$EXPECT_COUNT" ] && args+=(--expect-gap-count "$EXPECT_COUNT")
  [ -n "$EXPECT_PROV" ] && args+=(--expect-provenance "$EXPECT_PROV")
  [ -n "$VERDICT" ] && args+=(--verdict-out "$VERDICT")
  "$PY" "$D/da_state_tape_verify.py" "${args[@]}" >> "$LOG" 2>&1
}

run_count() {
  if [ "${DA_GATE_STUB:-0}" = "1" ]; then return "${DA_STUB_COUNT_RC:-0}"; fi
  "$PY" "$D/da_gap_at_cutoff_count.py" count --tape "$T" >> "$LOG" 2>&1
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
# Promote to the ruled locator. Only a verdict the gate actually WROTE is
# promoted; a refusal leaves the locator absent by design.
if [ -n "$VERDICT" ] && [ "$VERDICT" != "$RULED_VERDICT" ] && [ -s "$VERDICT" ]; then
  vtmp="$(mktemp "${RULED_VERDICT}.XXXXXX")"
  if cp "$VERDICT" "$vtmp" && mv -f "$vtmp" "$RULED_VERDICT"; then
    echo "PROMOTED verdict -> $RULED_VERDICT (content unchanged; $VERDICT kept as archive)" >> "$LOG"
  else
    rm -f "$vtmp"; track 5
    echo "PROMOTION FAILED -- the ruled locator does NOT resolve; fitting must not start" >> "$LOG"
  fi
elif [ -n "$ARCHIVE" ] && [ -s "$RULED_VERDICT" ]; then
  cp "$RULED_VERDICT" "$ARCHIVE" && echo "archived -> $ARCHIVE" >> "$LOG"
fi
run_count; rc=$?; track "$rc"; echo "COUNT_EXIT=$rc" >> "$LOG"
echo "WORST_EXIT=$worst  ($(date -u +%FT%TZ))" >> "$LOG"
exit "$worst"
