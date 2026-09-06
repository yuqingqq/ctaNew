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
# THE JOURNAL IS NOT THE RECORD (R-641, rule 20 as amended). It rotates
# within hours -- DE 84's Started line was gone four hours later -- so this
# launcher writes a LAUNCH RECORD under data/pm_5min/derived/ at launch and
# at exit: the unit, the command, the tip, the lock, the clock at each step,
# and the outcome. The journal is quoted beside it, filtered on the run's
# InvocationID, never depended on. The unit is polled BY NAME, never by a
# child PID -- there is no child of this shell to poll.
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
DECLDIR="$(dirname "$(readlink -f "$0")")/declarations"
# THE CHAIN HEAD, never a filename. Pinning `_v1.json` is exactly what
# the declaration's own chain_head_rule forbids -- REV 69 S4 measured a
# runner pinned to v1 at 14:00Z while v2 existed, satisfying rule 20
# guard 1's words without its property. Measured here too: this line
# said _v1.json and the head is v3.
DECL=$(python3 - "$DECLDIR" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1]); loaded = {}
for q in sorted(d.glob("heavy_run_form_v*.json")):
    b = q.read_bytes()
    loaded[q.name] = (q, hashlib.sha256(b).hexdigest(), json.loads(b))
sup = set()
for name, (q, sha, doc) in loaded.items():
    s = doc.get("supersedes")
    if isinstance(s, dict) and s.get("path"):
        prev = Path(s["path"]).name
        if prev in loaded and loaded[prev][1] != s.get("sha256"):
            sys.exit(1)
        sup.add(prev)
heads = [n for n in loaded if n not in sup]
print(str(loaded[heads[0]][0]) if len(heads) == 1 else "", end="")
PYEOF
)
# R-646: ONE declaration, read by the launcher AND by be_rule22. The
# literal 75 lived here and in Python, and the checker returned the
# PYTHON one -- so a launcher refusing with 76 published as 75.
if [ ! -r "$DECL" ]; then
  echo "REFUSED: the heavy-run form declaration $DECL is absent." \
       "A check that depends on a declaration FAILS when it is gone;" \
       "it never skips (R-649)." >&2
  exit 78
fi
# the key is the DECLARATION's own: `lock_conflict_rc` (heavy_run_form v1-v3)
LOCK_CONFLICT_RC=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["lock_conflict_rc"])' "$DECL" 2>/dev/null)
if ! [ "$LOCK_CONFLICT_RC" -eq "$LOCK_CONFLICT_RC" ] 2>/dev/null; then
  echo "REFUSED: could not read lock_conflict_rc from $DECL." \
       "An empty value would reach flock as `-E ''` and turn a refusal" \
       "into a usage error -- measured, 14:41Z." >&2
  exit 78
fi

if [ "${1:-}" = "--inner" ]; then
  # Runs AS the unit's main process. Not called directly.
  # THE LOCK ARRIVES AS AN ARGUMENT, not through the environment: measured
  # 14:44Z, a unit launched with `--setenv=BE_HEAVY_LOCK=<scratch>` locked
  # the REAL lock instead, because the variable did not reach the unit --
  # and a falsifier that silently drives the wrong lock is worse than none.
  shift
  if [ "${1:-}" = "--lock" ]; then LOCK="$2"; shift 2; fi
  flock -n -E "$LOCK_CONFLICT_RC" "$LOCK" "$@"
  rc=$?
  if [ "$rc" -eq "$LOCK_CONFLICT_RC" ]; then
    MSG="REFUSED: the heavy-run lock $LOCK is held by another run. This unit did NO work and wrote nothing (rule 20)."
    echo "$MSG" >&2
    # captured AT THE MOMENT OF REFUSAL, so no reader depends on journald
    [ -n "${BE_REFUSAL_FILE:-}" ] && echo "$MSG" >> "$BE_REFUSAL_FILE"
  fi
  [ -n "${BE_RECORD:-}" ] && printf '{"event":"exit","utc":"%s","rc":%s}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >> "$BE_RECORD"
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
  # REV 65 S1.1: this used `$0`. Invoked without a slash, the inner
  # `bash -c` resolves it through $PATH, `.` is not on $PATH, the
  # launch FAILS -- and with its output at /dev/null the cell reported
  # `unit=inactive`, which reads as THE PROPERTY IS FALSE. A control
  # that cannot tell "false" from "I could not start it" reports the
  # wrong direction. Qualified path, and stderr kept.
  ME="$(readlink -f "$0")"
  setsid bash -c "echo \$\$ > $D/pg; BE_HEAVY_LOCK=$D/l1 \"$ME\" $u1 $D/sleeper.py >$D/inner1.out 2>&1; sleep 20" &
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
    if [ "$ST" = "inactive" ] && [ "$MP1" = "0" ]; then
      echo "FAIL cell 1 -- THE UNIT NEVER STARTED, so this says NOTHING about the property: $(tail -2 "$D/inner1.out" 2>/dev/null | tr '\n' ' ')"
    else
      echo "FAIL cell 1: unit=$ST mainpid $MP1->$MP2 launcher_alive=$(ps -p "$PG" >/dev/null 2>&1 && echo yes || echo no)"
    fi; RC=1
  fi
  systemctl --user stop $u1 >/dev/null 2>&1; systemctl --user reset-failed $u1 >/dev/null 2>&1
  # CELL 2: a HELD lock must refuse with the distinct conflict code, do no
  # work, and say so in the journal.
  : > "$D/l2"; flock -n "$D/l2" sleep 12 &
  HOLDER=$!
  for _ in $(seq 1 50); do grep -q "$(stat -c %i "$D/l2")" /proc/locks && break; done
  ME2="$(readlink -f "$0")"
  BE_REFUSAL_FILE="$D/refusal.txt" BE_HEAVY_LOCK="$D/l2" "$ME2" $u2 "$D/sleeper.py" >/dev/null 2>&1
  for _ in $(seq 1 100); do
    [ "$(systemctl --user show $u2 -p ActiveState --value)" = "active" ] || break
  done
  RC2=$(systemctl --user show $u2 -p ExecMainStatus --value)
  # THE JOURNAL IS NOT THE RECORD (R-641). This cell used to grep journald
  # after the fact, so its verdict depended on retention -- and the journal
  # rotates within hours. The refusal is now written by the wrapper to a
  # file AT THE MOMENT IT REFUSES, and the cell reads the unit's own result
  # plus that file. The journal is quoted beside them, never depended on.
  F=$(grep -c "REFUSED: the heavy-run lock" "$D/refusal.txt" 2>/dev/null || echo 0)
  J=$(journalctl --user -u $u2 --no-pager -o cat 2>/dev/null | grep -c "REFUSED: the heavy-run lock")
  if [ "$RC2" = "$LOCK_CONFLICT_RC" ] && [ "$F" -ge 1 ]; then
    echo "PASS cell 2: a held lock refused with ExecMainStatus=$RC2 (the declared conflict code, never the command's own 1); the refusal was captured to the unit's own file at the moment it refused ($F line(s)); the unit did no work. Journal lines for the same unit: $J -- quoted, not depended on."
  else
    echo "FAIL cell 2: ExecMainStatus=$RC2 (declared $LOCK_CONFLICT_RC) refusal_file_lines=$F journal_lines=$J"; RC=1
  fi
  kill "$HOLDER" 2>/dev/null; wait "$HOLDER" 2>/dev/null
  systemctl --user reset-failed $u2 >/dev/null 2>&1; rm -rf "$D"
  exit $RC
fi

if [ "${1:-}" = "--poll" ]; then
  # DECLARED POLL (R-653). Every attempt records the FIVE declared outcome
  # fields AND the InvocationID it observed, because a unit NAME names every
  # run ever launched under it and a failed unit's properties persist until
  # `reset-failed` -- so a REPEATED id is the SAME refusal, not a new one,
  # and a poll that does not record the id cannot tell them apart.
  shift
  PUNIT="${1:?usage: --poll <unit> <module.py> [args...]}"; shift
  SELFP="$(readlink -f "$0")"
  CEIL="${BE_POLL_CEILING:-150}"; N=0; PREV_ID=""
  REC="$REPO/data/pm_5min/derived/be_heavy_run_record_${PUNIT}.jsonl"
  while [ "$N" -lt "$CEIL" ]; do
    N=$((N+1))
    # RESET BEFORE EVERY ATTEMPT: without it systemd refuses the name and
    # the next reading would be the PREVIOUS attempt's, unchanged.
    systemctl --user reset-failed "$PUNIT.service" >/dev/null 2>&1
    REC="$REPO/data/pm_5min/derived/be_heavy_run_record_${PUNIT}.jsonl"
    MARK=$(wc -l < "$REC" 2>/dev/null || echo 0)
    "$SELFP" "$PUNIT" "$@" >/dev/null 2>&1
    # SETTLE ON THE RECORD, NOT ON ActiveState. Measured 15:03:14Z on the
    # real lock: the poll read `active/running` and called it LOCK TAKEN in
    # the SAME SECOND the record showed the payload exiting 75. The unit IS
    # running at that instant -- the wrapper is -- and `flock` has not yet
    # decided; `RemainAfterExit` then keeps it `active` after it does. So
    # the discriminator is the launcher's own exit event: if one appears
    # within the settle window the run is over, and if none does it is
    # genuinely running (a real heavy run holds for tens of minutes).
    SETTLE=0
    for _ in $(seq 1 25); do
      NOW=$(wc -l < "$REC" 2>/dev/null || echo 0)
      if [ "$NOW" -gt "$MARK" ] && tail -n +$((MARK+1)) "$REC" 2>/dev/null | grep -q '"event":"exit"'; then
        SETTLE=1; break
      fi
      sleep 0.4
    done
    LS=$(systemctl --user show "$PUNIT.service" -p LoadState --value)
    AS=$(systemctl --user show "$PUNIT.service" -p ActiveState --value)
    SS=$(systemctl --user show "$PUNIT.service" -p SubState --value)
    RS=$(systemctl --user show "$PUNIT.service" -p Result --value)
    MS=$(systemctl --user show "$PUNIT.service" -p ExecMainStatus --value)
    ID=$(systemctl --user show "$PUNIT.service" -p InvocationID --value)
    SAME=false; [ -n "$ID" ] && [ "$ID" = "$PREV_ID" ] && SAME=true
    printf '{"event":"poll","utc":"%s","attempt":%s,"LoadState":"%s","ActiveState":"%s","SubState":"%s","Result":"%s","ExecMainStatus":"%s","InvocationID":"%s","same_invocation_as_previous":%s}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$N" "$LS" "$AS" "$SS" "$RS" "$MS" "$ID" "$SAME" >> "$REC"
    if [ "$LS" = "not-found" ]; then
      echo "VOID attempt $N: the unit is not loaded -- a collected unit and one that never ran are indistinguishable (R-653). Not a verdict."
    elif [ "$SETTLE" = "1" ] && [ "$MS" = "$LOCK_CONFLICT_RC" ]; then
      if [ "$SAME" = true ]; then
        echo "REFUSAL #$N $(date -u +%Y-%m-%dT%H:%M:%SZ) -- SAME InvocationID $ID as the previous read: this is the SAME refusal, not a new one"
      else
        echo "REFUSAL #$N $(date -u +%Y-%m-%dT%H:%M:%SZ) rc=$MS id=$ID (did no work)"
      fi
    elif [ "$SETTLE" = "0" ] && [ "$AS" = "active" ] && [ "$SS" = "running" ]; then
      # SUBSTATE DECIDES, NOT ActiveState. With RemainAfterExit=yes a unit
      # that has ALREADY EXITED stays `active` -- measured at 14:42:23Z,
      # where a poll read `active` and called it LOCK TAKEN while the
      # record showed the payload had exited 75 in the same second.
      echo "LOCK TAKEN $(date -u +%Y-%m-%dT%H:%M:%SZ) attempt $N after $((N-1)) refusals; InvocationID $ID"
      exit 0
    elif [ "$SETTLE" = "1" ] && [ "$MS" = "0" ] && [ "$RS" = "success" ]; then
      # A RUN THAT FINISHES INSIDE THE SETTLE WINDOW. Measured at 16:34:29Z:
      # the structure verification took the lock and completed in 4.3 s, and
      # this poll called it UNEXPECTED because it had no branch for success
      # -- only for refusal and for still-running. The run was unaffected;
      # the classification was wrong, which is its own defect.
      echo "TOOK THE LOCK AND FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ) attempt $N after $((N-1)) refusals; Result=$RS ExecMainStatus=$MS InvocationID=$ID"
      exit 0
    else
      echo "UNEXPECTED attempt $N: LoadState=$LS ActiveState=$AS SubState=$SS Result=$RS ExecMainStatus=$MS id=$ID"
      exit 2
    fi
    PREV_ID="$ID"
    sleep 60
  done
  echo "CEILING $CEIL reached after $N attempts -- not taken"
  exit 1
fi

UNIT="${1:?usage: be_heavy_run.sh <unit> <module.py> [args...]}"; shift
MOD="${1:?usage: be_heavy_run.sh <unit> <module.py> [args...]}"; shift
SELF="$(readlink -f "$0")"
case "$MOD" in /*) TARGET="$MOD";; *) TARGET="live/pm_research/$MOD";; esac

REC="$REPO/data/pm_5min/derived/be_heavy_run_record_${UNIT}.jsonl"
TIP=$(git -C "$WT" rev-parse HEAD 2>/dev/null || echo UNRESOLVED)
printf '{"event":"launch","utc":"%s","unit":"%s","payload":"%s","args":"%s","tip":"%s","worktree":"%s","lock":"%s","conflict_rc":%s,"declaration":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$UNIT" "$TARGET" "$*" "$TIP" "$WT" "$LOCK" \
  "$LOCK_CONFLICT_RC" "$DECL" >> "$REC"

exec systemd-run --user --unit="$UNIT" --slice=research.slice \
  -p MemoryMax=8G -p CPUQuota=100% -p RemainAfterExit=yes \
  --setenv=BE_RECORD="$REC" \
  --setenv=BE_REFUSAL_FILE="${BE_REFUSAL_FILE:-$REPO/data/pm_5min/derived/be_heavy_run_refusal_${UNIT}.txt}" \
  --setenv=PM_DATA_ROOT="$REPO" \
  --setenv=BE_HEAVY_LOCK="$LOCK" \
  --working-directory="$WT" \
  -- "$SELF" --inner --lock "$LOCK" "$PY" "$TARGET" "$@"
