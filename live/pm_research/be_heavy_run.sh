#!/bin/bash
# THE ONE LAUNCH FORM FOR BE'S HEAVY PRODUCERS (R-628; REV 63 S4).
#
# WHY THIS EXISTS. Every BE heavy run through 09-05 -- be55book, be58frag,
# be58tape, be59book, be61frag, be61tape -- was launched as a transient
# `--scope`. A scope registers processes the CALLER forked, so the payload
# lives in the launching shell's process tree and dies with it. Those runs
# survived because nothing signalled the launcher; survival is not the
# property. A transient SERVICE is owned by systemd: the launcher returns
# immediately, and killing it cannot touch the run.
#
# THE LOCK GOES INSIDE THE UNIT. `flock -n <lock> systemd-run --scope ...`
# put the lock in the launching shell, so the lock died with the shell too.
# Here `flock` is the unit's own main process.
#
# THE JOURNAL IS THE LOG, and the unit is polled BY NAME:
#   systemctl --user show <unit> -p ActiveState -p ExecMainStatus
#   journalctl --user -u <unit>
# never by a child PID -- there is no child of this shell to poll.
#
# A HELD LOCK IS A REFUSAL, NOT A FAILURE: `flock -E 75` makes the conflict
# exit code distinct from anything the command itself can return, so
# "another run holds the lock" can never be misread as "the build failed".
#
# usage: be_heavy_run.sh <unit> <module.py> [args...]
set -u
# BE_HEAVY_LOCK exists so the launcher's own falsifiers can drive a
# SCRATCH lock without touching the real one. It cannot weaken a real
# build: the producers verify the REAL lock by inode through DE's
# `wrapper_observed`, so a run launched against a scratch lock refuses
# inside the builder before it does any work.
LOCK="${BE_HEAVY_LOCK:-/home/yuqing/ctaNew/data/.heavy_run.lock}"
PY=/home/yuqing/pricer-sol/venv/bin/python3
WT="${BE_WORKTREE:-/home/yuqing/ctaNew-wt-be}"
REPO=/home/yuqing/ctaNew
LOCK_CONFLICT_RC=75

if [ "${1:-}" = "--inner" ]; then
  # Runs AS the unit's main process. Not called directly.
  shift
  flock -n -E "$LOCK_CONFLICT_RC" "$LOCK" "$@"
  rc=$?
  if [ "$rc" -eq "$LOCK_CONFLICT_RC" ]; then
    echo "REFUSED: the heavy-run lock $LOCK is held by another run." \
         "This unit did NO work and wrote nothing (rule 20)." >&2
  fi
  exit "$rc"
fi

if [ "${1:-}" = "--falsify" ]; then
  # RULE 15: this launcher ships its own falsifier. Two cells, both driven
  # against real systemd units, because the property being claimed is about
  # process ownership and nothing short of a real unit can show it.
  D=$(mktemp -d); RC=0
  u1=be_hr_falsify_alive_$$; u2=be_hr_falsify_lock_$$
  cat > "$D/sleeper.py" <<'PYEOF'
import time; print("payload up", flush=True); time.sleep(25)
PYEOF
  # CELL 1: a TERM to the LAUNCHING shell's process group must leave the
  # unit ALIVE. Under the old `--scope` form the payload sat in that group
  # and died with it; six BE heavy runs survived only because nothing
  # signalled the launcher.
  : > "$D/l1"
  setsid bash -c "echo \$\$ > $D/pg; BE_HEAVY_LOCK=$D/l1 $0 $u1 $D/sleeper.py >/dev/null 2>&1; sleep 20" &
  for _ in $(seq 1 100); do
    [ "$(systemctl --user show $u1 -p ActiveState --value)" = "active" ] && break
  done
  PG=$(cat "$D/pg" 2>/dev/null); MP1=$(systemctl --user show $u1 -p MainPID --value)
  kill -TERM -"$PG" 2>/dev/null
  for _ in $(seq 1 50); do ps -p "$PG" >/dev/null 2>&1 || break; done
  ST=$(systemctl --user show $u1 -p ActiveState --value)
  MP2=$(systemctl --user show $u1 -p MainPID --value)
  PPID_OF_PAYLOAD=$(ps -o ppid= -p "$MP2" 2>/dev/null | tr -d ' ')
  if [ "$ST" = "active" ] && [ "$MP1" = "$MP2" ] && ! ps -p "$PG" >/dev/null 2>&1; then
    echo "PASS cell 1: the launching shell's process group was TERMed and is gone; unit $u1 is still $ST on the same MainPID $MP2, whose parent is pid $PPID_OF_PAYLOAD (systemd --user). A --scope payload would have died with that group."
  else
    echo "FAIL cell 1: unit=$ST mainpid $MP1->$MP2 launcher_alive=$(ps -p "$PG" >/dev/null 2>&1 && echo yes || echo no)"; RC=1
  fi
  systemctl --user stop $u1 >/dev/null 2>&1; systemctl --user reset-failed $u1 >/dev/null 2>&1
  # CELL 2: a HELD lock must refuse with the distinct conflict code, do no
  # work, and say so in the journal.
  : > "$D/l2"; flock -n "$D/l2" sleep 12 &
  HOLDER=$!
  for _ in $(seq 1 50); do grep -q "$(stat -c %i "$D/l2")" /proc/locks && break; done
  BE_HEAVY_LOCK="$D/l2" "$0" $u2 "$D/sleeper.py" >/dev/null 2>&1
  for _ in $(seq 1 100); do
    [ "$(systemctl --user show $u2 -p ActiveState --value)" = "active" ] || break
  done
  RC2=$(systemctl --user show $u2 -p ExecMainStatus --value)
  J=$(journalctl --user -u $u2 --no-pager -o cat 2>/dev/null | grep -c "REFUSED: the heavy-run lock")
  if [ "$RC2" = "75" ] && [ "$J" -ge 1 ]; then
    echo "PASS cell 2: a held lock refused with ExecMainStatus=75 (the distinct conflict code, never the command's own 1) and the journal carries the named refusal; the unit did no work."
  else
    echo "FAIL cell 2: ExecMainStatus=$RC2 journal_refusal_lines=$J"; RC=1
  fi
  kill "$HOLDER" 2>/dev/null; wait "$HOLDER" 2>/dev/null
  systemctl --user reset-failed $u2 >/dev/null 2>&1; rm -rf "$D"
  exit $RC
fi

UNIT="${1:?usage: be_heavy_run.sh <unit> <module.py> [args...]}"; shift
MOD="${1:?usage: be_heavy_run.sh <unit> <module.py> [args...]}"; shift
SELF="$(readlink -f "$0")"
case "$MOD" in /*) TARGET="$MOD";; *) TARGET="live/pm_research/$MOD";; esac

exec systemd-run --user --unit="$UNIT" --slice=research.slice \
  -p MemoryMax=8G -p CPUQuota=100% \
  --setenv=PM_DATA_ROOT="$REPO" \
  --setenv=BE_HEAVY_LOCK="$LOCK" \
  --working-directory="$WT" \
  -- "$SELF" --inner "$PY" "$TARGET" "$@"
