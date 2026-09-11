#!/bin/bash
# LAUNCH A VALUATION THE MOMENT A NAMED PRODUCER RELEASES THE HEAVY LOCK.
#
# KEYED TO THE PRODUCER, NEVER TO A CLOCK. An emit waiter armed with a 3 h
# budget once expired ten minutes before its result; a waiter keyed to one
# offer unit would have refused on the next lock conflict. The producer is
# named on the command line and the wait ends when THAT unit stops running.
#
# Usage:
#   value_after_producer.sh --producer <unit> --day <YYYY-MM-DD> \
#       --book <pkl> --receipt <json> --cert <json> --out <dir> [--draws N]
#   value_after_producer.sh --falsify
#
# Refusals (named, never a bare non-zero):
#   PRODUCER_UNIT_DOES_NOT_EXIST        nothing to wait for
#   PRODUCER_ENDED_WITHOUT_RELEASING_THE_LOCK
#   LAUNCHER_BYTES_NOT_COMMITTED        the script is not the landed bytes
#   VALUATION_INPUT_ABSENT:<file>
set -uo pipefail
TREE=/home/yuqing/ctaNew-wt-deval
LOCK=/home/yuqing/ctaNew/data/.heavy_run.lock
PY=/home/yuqing/pricer-sol/venv/bin/python3
DRAWS=0

say() { echo "$(date -u +%H:%M:%SZ) $*"; }
refuse() { echo "REFUSED $1" >&2; exit "${2:-9}"; }

if [ "${1:-}" = "--falsify" ]; then
  # (i) a producer that does not exist, (ii) a lock held by a producer that
  # has already ended, (iii) the clean path -- all three DRIVEN, not read.
  n=0; ok=0
  ck() { n=$((n+1)); if [ "$2" = "$3" ]; then ok=$((ok+1)); echo "  [PASS] $1  $3"; else echo "  [FAIL] $1  got $3 want $2"; fi; }
  out=$("$0" --producer no-such-unit-$$.service --day 2026-01-01 \
        --book /dev/null --receipt /dev/null --cert /dev/null \
        --out /tmp 2>&1); rc=$?
  ck "a producer that does not exist REFUSES by name" \
     "1x11" "$(echo "$out" | grep -c PRODUCER_UNIT_DOES_NOT_EXIST)x$rc"
  d=$(mktemp -d); : > "$d/l"
  flock -n "$d/l" sleep 30 &
  holder=$!
  out=$(LOCKOVERRIDE="$d/l" "$0" --producer dbus.service --day 2026-01-01 \
        --book /dev/null --receipt /dev/null --cert /dev/null --out /tmp \
        --lock "$d/l" --no-wait 2>&1); rc=$?
  kill $holder 2>/dev/null
  ck "a lock still held after the producer ends REFUSES by name" \
     "1x12" "$(echo "$out" | grep -c PRODUCER_ENDED_WITHOUT_RELEASING_THE_LOCK)x$rc"
  out=$("$0" --producer dbus.service --day 2026-01-01 \
        --book /tmp/no-such-book-$$.pkl --receipt /dev/null --cert /dev/null \
        --out /tmp --lock "$d/l" --no-wait 2>&1); rc=$?
  rm -rf "$d"
  ck "an absent book REFUSES by name, not by traceback" \
     "1x13" "$(echo "$out" | grep -c VALUATION_INPUT_ABSENT)x$rc"
  echo; echo "$ok/$n cells pass"; [ "$ok" = "$n" ] && exit 0 || exit 1
fi

WAIT=1
while [ $# -gt 0 ]; do
  case "$1" in
    --producer) PRODUCER="$2"; shift 2;;
    --day) DAY="$2"; shift 2;;
    --book) BOOK="$2"; shift 2;;
    --receipt) RECEIPT="$2"; shift 2;;
    --cert) CERT="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --draws) DRAWS="$2"; shift 2;;
    --lock) LOCK="$2"; shift 2;;
    --no-wait) WAIT=0; shift;;
    *) refuse "UNKNOWN_ARGUMENT:$1" 10;;
  esac
done

if ! systemctl --user list-unit-files --no-legend "$PRODUCER" >/dev/null 2>&1 \
   || ! systemctl --user show "$PRODUCER" -p LoadState 2>/dev/null \
        | grep -q "LoadState=loaded"; then
  refuse "PRODUCER_UNIT_DOES_NOT_EXIST:$PRODUCER" 11
fi

if [ "$WAIT" = 1 ]; then
  say "waiting on $PRODUCER (not on a clock)"
  while [ "$(systemctl --user show "$PRODUCER" -p SubState --value 2>/dev/null)" = "running" ]; do
    sleep 10
  done
  say "$PRODUCER is $(systemctl --user show "$PRODUCER" -p SubState --value 2>/dev/null)"
fi

for f in "$BOOK" "$RECEIPT" "$CERT"; do
  [ -e "$f" ] || refuse "VALUATION_INPUT_ABSENT:$f" 13
done
if ! flock -n "$LOCK" true; then
  refuse "PRODUCER_ENDED_WITHOUT_RELEASING_THE_LOCK:$LOCK" 12
fi

cd "$TREE" || refuse "TREE_ABSENT:$TREE" 14
H=$(git -C "$TREE" rev-parse HEAD)
say "tree $TREE head ${H:0:12}"
PARAMS=$("$PY" -c "
import sys; sys.path.insert(0,'$TREE/live/pm_research')
import be_score_neutrality as B
from pathlib import Path
print(Path('$TREE/live/pm_research/declarations')/Path(
    B.resolve_frozen_params_pin(Path('$TREE/live/pm_research/declarations'))['pin']['path']).name)")
[ -f "$PARAMS" ] || refuse "PARAMS_NOT_RESOLVED_FROM_THE_CHAIN" 15
say "params from the chain: $(basename "$PARAMS")"

export BE_WORKTREE="$TREE" DE_VALUATION_EXPECTED_TREE="$TREE"
"$TREE/live/pm_research/be_heavy_run.sh" --inner --lock "$LOCK" \
  "$PY" "$TREE/live/pm_research/de_forward_value_day.py" \
  --day "$DAY" --book "$BOOK" --book-receipt "$RECEIPT" \
  --score-certification "$CERT" --params "$PARAMS" --out-dir "$OUT" \
  --n-draws "$DRAWS" --seed 0 --days-scored "$DAY" --n-declared 7 \
  --derived /home/yuqing/ctaNew/data/pm_5min/derived
rc=$?
say "valuation rc=$rc"
exit $rc
