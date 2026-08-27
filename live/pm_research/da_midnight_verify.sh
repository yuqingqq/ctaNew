#!/bin/bash
# DA verify-first, fired by a systemd timer at the UTC day boundary.
#
# Runs as a TIMER-launched unit, not a session job: every session-scoped watch
# DA armed on 2026-08-26 was killed before it fired. The duty is nightly and
# must not depend on a session being awake (R-153(2) made it a hard
# precondition, and the one time it ran late the day had already been scored).
#
# FIRES AT 00:06, NOT 00:00:30. The first run verdicted a day whose tape was
# still settling: the collector records each window until start+WINDOW_S+GRACE_S,
# so a day's LAST window (23:55) is still recording until 00:01:30 and is gzipped
# after that. At 00:00:30 the 08-26 counts read 277/278; minutes later they read
# 278/279. Cosmetic for an already-inadmissible day; for DAY ONE it would
# undercount a complete day and fail complete_tape on tape that exists.
#
# It COMPUTES and LOGS. It does not exclude and does not file -- a day that
# fails is excluded with a stated reason by the coordinator (rule 14).
set -u
# Overridable ONLY so this script can be rehearsed without writing into the
# nightly record. The seam test that drove the tape wrapper wrote its stub runs
# into the PRODUCTION gate log and I misread them as a real refusal -- a log
# that cannot tell a rehearsal from the real thing is not evidence.
LOG="${DA_MIDNIGHT_LOG:-/home/yuqing/ctaNew/data/pm_5min/derived/.da_midnight_verify.log}"
V=/home/yuqing/ctaNew/live/pm_research/da_forward_day_verify.py
PY=/home/yuqing/pricer-sol/venv/bin/python3
cd /home/yuqing/ctaNew/live/pm_research || exit 3
{
  echo
  echo "======== fired $(date -u +%FT%TZ) ========"
} >> "$LOG"
# The day that just CLOSED, and the day that just OPENED. Both are logged
# because which one is "day one" was ambiguous at scheduling time and a
# verifier that silently picks one would be guessing on the record.
CLOSED=$(date -u -d "yesterday" +%Y%m%d)
OPENED=$(date -u +%Y%m%d)
# A FAILING DAY AND A BROKEN INSTRUMENT ARE DIFFERENT OUTCOMES.
#   0 = verified, all pass      1 = verified, day FAILS (a real result)
#   4 = INSTRUMENT FAILURE, nothing verified
# The unit must go red only for the third: a day failing verification is the
# instrument working, and marking that as a unit failure would train everyone
# to ignore it. But this script previously exited with the status of its last
# `echo` -- always 0 -- so a verifier that crashed reported SUCCESS to systemd
# and day one's hard precondition would have silently not happened. That is
# the exact defect fixed in the tape wrapper (R-199 item 1); it was still here.
# THE EXIT CODE ALONE CANNOT CARRY THIS, and my first version of this fix
# believed it could. `return 4` only fires for an exception INSIDE main(); a
# module that dies on import, or python itself failing, exits 1 -- identical to
# a legitimately failing day. My own falsifier caught it: a deliberately broken
# verifier still let this wrapper exit 0.
# So the test is POSITIVE EVIDENCE that a verdict was computed: the run must
# leave a parseable artifact naming the day it claims to have verified. Absence
# is never success, the same rule the tape gate learned about skip counters.
# Written to a temp path and PROMOTED only after it validates -- nothing
# pre-existing is deleted, and a stale file cannot masquerade as tonight's run.
OUTDIR=/home/yuqing/ctaNew/data/pm_5min/derived
broke=0
for d in "$CLOSED" "$OPENED"; do
  echo "---- day $d ----" >> "$LOG"
  tmp="$(mktemp "$OUTDIR/.da_dayverdict_$d.XXXXXX.json")"
  "$PY" "$V" verify --day "$d" --out "$tmp" >> "$LOG" 2>&1
  rc=$?
  if "$PY" -c 'import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get("day_token")==sys.argv[2] and d.get("predicates") else 1)'        "$tmp" "$d" 2>/dev/null; then
    mv -f "$tmp" "$OUTDIR/da_dayverdict_$d.json"
    echo "exit=$rc for $d (verdict artifact written)" >> "$LOG"
  else
    rm -f "$tmp"
    broke=4
    echo "exit=$rc for $d  <-- INSTRUMENT FAILURE: NO PARSEABLE VERDICT NAMING $d. NOTHING WAS VERIFIED." >> "$LOG"
  fi
done
echo "done $(date -u +%FT%TZ); worst_instrument_rc=$broke" >> "$LOG"
exit "$broke"
