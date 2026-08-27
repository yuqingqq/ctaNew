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
LOG=/home/yuqing/ctaNew/data/pm_5min/derived/.da_midnight_verify.log
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
for d in "$CLOSED" "$OPENED"; do
  echo "---- day $d ----" >> "$LOG"
  "$PY" "$V" verify --day "$d" >> "$LOG" 2>&1
  echo "exit=$? for $d" >> "$LOG"
done
echo "done $(date -u +%FT%TZ)" >> "$LOG"
