#!/usr/bin/env bash
# research_slot.sh -- refuse to launch a research process when the box is full.
#
# The cap is on the BOX, not per agent. Six agents each honouring a per-process
# limit is still six processes. This script is the checkable form of that rule.
#
# Usage:   research_slot.sh <core> -- <command...>
# Example: research_slot.sh 4 -- python3 de_my_probe.py --day 20260910
#
# Env:     RESEARCH_SLOTS  (default 2)  total concurrent research processes
#          RESEARCH_MEM_GIB(default 4)  per-process address-space cap
#          RESEARCH_MIN_FREE_GIB (default 6) refuse if less free than this
set -u

SLOTS="${RESEARCH_SLOTS:-2}"
MEM_GIB="${RESEARCH_MEM_GIB:-4}"
MIN_FREE="${RESEARCH_MIN_FREE_GIB:-6}"

# Heavy research processes: our own runners/probes, plus the forward race.
# Matched on the python3 argv head so wrapper shells are not counted.
running_list() {
  ps -eo pid,etimes,rss,args \
    | grep -E '^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+(/[^[:space:]]*/)?python3(\.[0-9]+)? ' \
    | grep -E '(de_|be_|da_|db_|cb_)[a-z0-9_]*\.py|forward_race_day_runner' \
    | grep -v 'research_slot'
}
# MINE = lines this session owns. The other seat's forward race shares the box
# but is not ours to schedule, so it is REPORTED and counted against memory,
# never against our slot cap.
mine_list()  { running_list | grep -v 'forward_race_day_runner'; }
other_list() { running_list | grep    'forward_race_day_runner'; }
n_mine()  { mine_list  | grep -c . ; }
n_other() { other_list | grep -c . ; }

N="$(n_mine)"
NO="$(n_other)"
FREE_GIB="$(free -g | awk '/^Mem:/{print $7}')"

if [ "$N" -ge "$SLOTS" ]; then
  echo "REFUSED: $N of our research process(es) already running, cap is $SLOTS." >&2
  echo "(plus $NO forward-race process(es) belonging to the other seat)" >&2
  echo "Running now:" >&2
  running_list | sed 's/^/  /' | cut -c1-120 >&2
  echo "Wait for a slot, or raise RESEARCH_SLOTS deliberately and say why." >&2
  exit 1
fi

if [ -n "$FREE_GIB" ] && [ "$FREE_GIB" -lt "$MIN_FREE" ]; then
  echo "REFUSED: only ${FREE_GIB} GiB available, need ${MIN_FREE}. This host has NO SWAP;" >&2
  echo "an overrun hangs the box rather than degrading." >&2
  exit 1
fi

if [ "${1:-}" = "--check" ]; then
  echo "OK: $N/$SLOTS of our slots used, $NO other-seat race proc(s), ${FREE_GIB} GiB free."
  exit 0
fi

CORE="${1:?usage: research_slot.sh <core> -- <command...>}"; shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: research_slot.sh <core> -- <command...>" >&2; exit 2; }

echo "SLOT GRANTED: ${N}/${SLOTS} ours used before this one, ${NO} other-seat, ${FREE_GIB} GiB free, core ${CORE}, ${MEM_GIB} GiB cap." >&2
exec taskset -c "$CORE" prlimit --as=$((MEM_GIB * 1024 * 1024 * 1024)) -- "$@"
