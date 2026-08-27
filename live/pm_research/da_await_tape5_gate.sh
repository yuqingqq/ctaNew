#!/bin/bash
# DA: wait for tape5, then run ALL THREE pre-registered checks.
#
# A SYSTEMD UNIT, not a session job: every session-scoped watch armed in this
# programme was killed mid-poll (R-147(2) lesson, re-learned twice).
# Size-stability is required because a build writing its final path directly
# means presence is not completeness (Q-DA-77(6)).
#
# THE THREE ARE PRE-REGISTERED AND FIXED BEFORE THE ARTIFACT EXISTS:
#   count      289
#   edge       the 493 at-g1 rows PRESENT and UNFLAGGED
#   provenance 057b1b7   (anchored prefix; ABSENT = FAIL)
# Each covers a blind spot of the others: the count alone cannot distinguish
# half-open-landed from gaps-never-arrived; the edge assertion alone cannot
# catch a wrong population; provenance alone catches neither, but without it
# neither of the first two attributes to any particular code.
set -u
T=/home/yuqing/ctaNew/data/pm_5min/derived/phase2_state_tape_v5.json
LOG=/home/yuqing/ctaNew/data/pm_5min/derived/.da_tape5_gate.log
PY=/home/yuqing/pricer-sol/venv/bin/python3
D=/home/yuqing/ctaNew/live/pm_research
DEADLINE=$(( $(date +%s) + 14400 ))
{ echo; echo "======== armed $(date -u +%FT%TZ) ========"; } >> "$LOG"
while [ ! -f "$T" ]; do
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT: tape5 never appeared" >> "$LOG"; exit 2; }
  sleep 20
done
echo "appeared $(date -u +%FT%TZ); awaiting size stability" >> "$LOG"
prev=-1; stable=0
while [ "$stable" -lt 3 ]; do
  sleep 15
  cur=$(stat -c%s "$T" 2>/dev/null || echo -1)
  if [ "$cur" = "$prev" ] && [ "$cur" -gt 0 ]; then stable=$((stable+1)); else stable=0; fi
  prev=$cur
  [ "$(date +%s)" -gt "$DEADLINE" ] && { echo "TIMEOUT awaiting stability" >> "$LOG"; exit 2; }
done
echo "stable at $prev bytes $(date -u +%FT%TZ)" >> "$LOG"
cd "$D" || exit 3
echo "---- GATE (count 289 + provenance 057b1b7) ----" >> "$LOG"
"$PY" "$D/da_state_tape_verify.py" verify --tape "$T" --gapped-slugs 133 \
      --expect-gap-count 289 --expect-provenance 057b1b7 >> "$LOG" 2>&1
echo "GATE_EXIT=$?" >> "$LOG"
echo "---- COUNT + EDGE PROBE ----" >> "$LOG"
"$PY" "$D/da_gap_at_cutoff_count.py" count --tape "$T" >> "$LOG" 2>&1
echo "COUNT_EXIT=$?" >> "$LOG"
echo "done $(date -u +%FT%TZ)" >> "$LOG"
