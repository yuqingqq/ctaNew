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
  systemctl --user reset-failed $u2 >/dev/null 2>&1

  # CELL 3 (BE 87 (a)): THE PEAK OF RECORD IS SAMPLED WHILE THE RUN IS
  # ALIVE, and the capture of a FINISHED run names the released leaf
  # instead of refusing. A fixture that holds 1 GiB is the control: a
  # sampler that reported a number smaller than what the payload provably
  # allocated would be measuring nothing.
  u3=be_hr_falsify_peak_$$
  cat > "$D/holder.py" <<'PYPEAK'
import time
x = bytearray(1024 * 1024 * 1024)      # 1 GiB, zero-filled = pages touched
print("holding 1 GiB", flush=True)
time.sleep(20)
print("done", flush=True)
PYPEAK
  BE_HEAVY_LOCK="$D/lock3" "$ME" "$u3" "$D/holder.py" >/dev/null 2>&1
  sleep 3
  BE_SAMPLE_CEILING=60 "$ME" --sample "$u3" 2 >/dev/null 2>&1
  R3="$REPO/data/pm_5min/derived/be_heavy_run_record_${u3}.jsonl"
  P3=$(grep '"event":"leaf_peak"' "$R3" 2>/dev/null | tail -1 | sed -n 's/.*"peak_of_record_bytes":\([0-9]*\).*/\1/p')
  N3=$(grep '"event":"leaf_peak"' "$R3" 2>/dev/null | tail -1 | sed -n 's/.*"n_samples":\([0-9]*\).*/\1/p')
  CAP3=$("$ME" --capture "$u3" 2>&1); CRC3=$?
  if [ "${P3:-0}" -ge 1073741824 ] && [ "${N3:-0}" -ge 1 ] \
     && [ "$CRC3" = "0" ] && printf '%s' "$CAP3" | grep -q "LEAF_RELEASED" \
     && printf '%s' "$CAP3" | grep -q "peak_of_record=$P3"; then
    echo "PASS cell 3: a unit that held 1 GiB was SAMPLED WHILE ALIVE -- peak of record $P3 bytes over $N3 sample(s), which is >= the 1073741824 the payload provably allocated; and the capture AFTER it exited returned 0, named the released leaf (LEAF_RELEASED) and carried that same peak forward instead of refusing the whole capture (BE 72 finding 1)."
  else
    echo "FAIL cell 3: peak=$P3 samples=$N3 capture_rc=$CRC3 capture=$CAP3"; RC=1
  fi
  systemctl --user reset-failed $u3 >/dev/null 2>&1
  systemctl --user stop $u3 >/dev/null 2>&1

  # CELLS 4 AND 5 (BE 87 (b) and (c)): a run that finishes BEFORE the poll
  # reads it is a SUCCESS, not an anomaly; and its stdout is in a FILE.
  u4=be_hr_falsify_done_$$
  cat > "$D/quick.py" <<'PYQUICK'
print("QUICK_PAYLOAD_RECORD_LINE", flush=True)
PYQUICK
  BE_HEAVY_LOCK="$D/lock4" "$ME" "$u4" "$D/quick.py" >/dev/null 2>&1
  sleep 4
  POUT=$(BE_HEAVY_LOCK="$D/lock4" "$ME" --poll "$u4" "$D/quick.py" 2>&1); PRC=$?
  if [ "$PRC" = "0" ] && printf '%s' "$POUT" | grep -q "ALREADY FINISHED" \
     && ! printf '%s' "$POUT" | grep -q "UNEXPECTED"; then
    echo "PASS cell 4: a unit that had already exited SUCCESSFULLY (active/exited, success, 0) is polled as a SUCCESS and exits 0 -- $(printf '%s' "$POUT" | head -1). It used to fall through to UNEXPECTED and exit 2, because every branch wanted a settle and a finished unit cannot produce one (measured on be72struct at 01:30:18Z)."
  else
    echo "FAIL cell 4: rc=$PRC out=$POUT"; RC=1
  fi
  OUT4="$REPO/data/pm_5min/derived/be_heavy_run_stdout_${u4}.log"
  if [ -r "$OUT4" ] && grep -q "QUICK_PAYLOAD_RECORD_LINE" "$OUT4"; then
    echo "PASS cell 5: the payload's stdout is in the FILE $OUT4 ($(wc -l < "$OUT4") line(s)) and the assertion above reads THE FILE, never the journal -- the record no longer lives only in something that rotates (rule 20, BE 72 finding 3)."
  else
    echo "FAIL cell 5: $OUT4 absent or does not carry the payload's line"; RC=1
  fi
  systemctl --user reset-failed $u4 >/dev/null 2>&1
  systemctl --user stop $u4 >/dev/null 2>&1
  rm -rf "$D"
  exit $RC
fi

if [ "${1:-}" = "--sample" ]; then
  # (a) THE PEAK OF RECORD IS SAMPLED WHILE THE RUN IS ALIVE (BE 72 finding
  # 1). `--capture` could never read it for a run that had finished: when
  # the payload exits systemd releases the cgroup leaf, `ControlGroup=`
  # goes empty, and the leaf's memory.peak -- the peak OF RECORD -- is
  # gone. Refusing was right (systemd's property is not that number, BE
  # 74); refusing FOREVER was the defect, because every completed run has
  # already exited by the time anyone captures it.
  #
  # memory.peak is MONOTONIC, so the sampler needs no state: the largest
  # value it ever reads IS the high-water mark up to its last read, and the
  # time of that read is recorded so nobody mistakes a bound for the peak.
  shift
  SUNIT="${1:?usage: --sample <unit> [interval_s]}"; shift
  IVL="${1:-5}"; SCEIL="${BE_SAMPLE_CEILING:-8640}"
  SREC="$REPO/data/pm_5min/derived/be_heavy_run_record_${SUNIT}.jsonl"
  BEST=0; BEST_T=""; NS=0; SS=""; LSS=""
  printf '{"event":"leaf_sampler_started","utc":"%s","unit":"%s","interval_s":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SUNIT" "$IVL" >> "$SREC"
  for _ in $(seq 1 "$SCEIL"); do
    LSS=$(systemctl --user show "$SUNIT.service" -p LoadState --value)
    SS=$(systemctl --user show "$SUNIT.service" -p SubState --value)
    SCG=$(systemctl --user show "$SUNIT.service" -p ControlGroup --value)
    if [ -n "$SCG" ] && [ -r "/sys/fs/cgroup$SCG/memory.peak" ]; then
      SP=$(cat "/sys/fs/cgroup$SCG/memory.peak" 2>/dev/null || echo 0)
      NS=$((NS+1))
      if [ "${SP:-0}" -gt "$BEST" ] 2>/dev/null; then
        BEST=$SP; BEST_T=$(date -u +%Y-%m-%dT%H:%M:%SZ)
      fi
    fi
    [ "$LSS" = "not-found" ] && break
    case "$SS" in exited|failed) break;; esac
    sleep "$IVL"
  done
  printf '{"event":"leaf_peak","utc":"%s","unit":"%s","peak_of_record_bytes":%s,"last_increase_utc":"%s","n_samples":%s,"interval_s":%s,"final_substate":"%s","sampled":"WHILE ALIVE -- the leaf is released at exit and cannot be read afterwards","if_n_samples_is_0":"the run finished before the first read; the peak is ABSENT, never zero"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SUNIT" "$BEST" "$BEST_T" "$NS" "$IVL" \
    "$SS" >> "$SREC"
  echo "sampled $SUNIT: peak_of_record=$BEST bytes at $BEST_T over $NS sample(s) every ${IVL}s; final SubState=$SS"
  exit 0
fi

if [ "${1:-}" = "--capture" ]; then
  # THE OUTCOME INTO THE RECORD (REV 78 / step 7). The launcher wrote only
  # `launch` and `exit` rows: for be70race the five fields and the
  # InvocationID were read by the SEAT and lived in its report, not in the
  # record -- a fact in prose, which is the class this programme keeps
  # closing. This appends them, then STOPS the unit, then copies the
  # journal -- in that order, because the Stopped/Consumed lines are
  # written BY the stop and a copy taken before it cannot contain them
  # (DE 106).
  shift
  CUNIT="${1:?usage: --capture <unit>}"; shift
  CREC="$REPO/data/pm_5min/derived/be_heavy_run_record_${CUNIT}.jsonl"
  LS=$(systemctl --user show "$CUNIT.service" -p LoadState --value)
  if [ "$LS" = "not-found" ]; then
    echo "REFUSED: $CUNIT is not loaded (LoadState=not-found). Its five" \
         "fields and InvocationID are unobtainable -- a collected unit and" \
         "one that never ran are indistinguishable (R-653). Capture BEFORE" \
         "the stop." >&2
    exit 76
  fi
  AS=$(systemctl --user show "$CUNIT.service" -p ActiveState --value)
  SS=$(systemctl --user show "$CUNIT.service" -p SubState --value)
  RS=$(systemctl --user show "$CUNIT.service" -p Result --value)
  MS=$(systemctl --user show "$CUNIT.service" -p ExecMainStatus --value)
  ID=$(systemctl --user show "$CUNIT.service" -p InvocationID --value)
  MP=$(systemctl --user show "$CUNIT.service" -p MemoryPeak --value)
  # THE PEAK'S SOURCE (BE 74's finding, REV 80 ruling on the reading).
  # systemd's MemoryPeak PROPERTY is not the cgroup's memory.peak for an
  # exited unit: measured on be74struct04b, the property read 847,671,296
  # while the leaf's own file read 2,578,067,456 and the process measured
  # 2.405 GB. Read WHILE RUNNING (be74probe) the two agreed exactly. So the
  # record now carries BOTH, each labelled with its source, and the peak OF
  # RECORD is the leaf's file. No check changes; this is a measurement.
  CG=$(systemctl --user show "$CUNIT.service" -p ControlGroup --value)
  # A RUN THAT FINISHED NORMALLY MUST NOT LOSE ITS CAPTURE. The leaf is
  # released at exit, so for EVERY completed run this exited 77 and wrote
  # NOTHING -- no five fields, no stop, no journal copy. The state is now
  # NAMED (`LEAF_RELEASED`) and the capture proceeds; the peak of record
  # comes from the `--sample` row taken while the run was alive, and
  # systemd's property rides beside it, never in its place (BE 74).
  CGB=""; LEAFPEAK=""; LEAFCUR=""; LEAF_STATUS="READ_FROM_THE_LEAF"
  if [ -z "$CG" ]; then
    LEAF_STATUS="LEAF_RELEASED"
  else
    CGB="/sys/fs/cgroup${CG}"
    LEAFPEAK=$(cat "$CGB/memory.peak" 2>/dev/null || echo "")
    LEAFCUR=$(cat "$CGB/memory.current" 2>/dev/null || echo "")
    [ -z "$LEAFPEAK" ] && LEAF_STATUS="LEAF_PRESENT_BUT_UNREADABLE"
  fi
  # THE PEAK OF RECORD: the live leaf if it is still there, else the
  # sampler's row, else ABSENT -- and WHICH ONE IT IS is said, never
  # inferred from a bare number.
  SLINE=$(grep '"event":"leaf_peak"' "$CREC" 2>/dev/null | tail -1)
  SPEAK=$(printf '%s' "$SLINE" | sed -n 's/.*"peak_of_record_bytes":\([0-9]*\).*/\1/p')
  STIME=$(printf '%s' "$SLINE" | sed -n 's/.*"last_increase_utc":"\([^"]*\)".*/\1/p')
  SN=$(printf '%s' "$SLINE" | sed -n 's/.*"n_samples":\([0-9]*\).*/\1/p')
  [ "${SPEAK:-0}" = "0" ] && SPEAK=""
  if [ -n "$LEAFPEAK" ]; then
    POR="$LEAFPEAK"
    POR_SRC="the leaf's own memory.peak, read live at capture"
  elif [ -n "$SPEAK" ]; then
    POR="$SPEAK"
    POR_SRC="the --sample row: the leaf's memory.peak read WHILE THE RUN WAS ALIVE, last increase $STIME over $SN sample(s)"
  else
    POR=""
    POR_SRC="ABSENT -- the leaf was released before any read and this run was not sampled. NOT zero, and NOT systemd's property (BE 74)"
  fi
  OUTF="$REPO/data/pm_5min/derived/be_heavy_run_stdout_${CUNIT}.log"
  if [ -r "$OUTF" ]; then
    OUTSHA=$(sha256sum "$OUTF" | cut -d" " -f1); OUTL=$(wc -l < "$OUTF")
  else
    OUTSHA="ABSENT"; OUTL=0
  fi
  printf '{"event":"outcome","utc":"%s","read_while":"LOADED","LoadState":"%s","ActiveState":"%s","SubState":"%s","Result":"%s","ExecMainStatus":"%s","InvocationID":"%s","peak_of_record_bytes":"%s","peak_of_record_source":"%s","leaf_status":"%s","cgroup_leaf":"%s","leaf_memory_peak":"%s","leaf_memory_current":"%s","systemd_MemoryPeak_property":"%s","systemd_property_source":"systemctl show -p MemoryPeak, recorded verbatim; NOT the peak of record (BE 74: it read 847671296 where the leaf read 2578067456)","stdout_file":"%s","stdout_file_sha256":"%s","stdout_file_lines":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$LS" "$AS" "$SS" "$RS" "$MS" "$ID" \
    "$POR" "$POR_SRC" "$LEAF_STATUS" "$CG" "$LEAFPEAK" "$LEAFCUR" "$MP" \
    "$OUTF" "$OUTSHA" "$OUTL" >> "$CREC"
  systemctl --user stop "$CUNIT.service" >/dev/null 2>&1
  printf '{"event":"stopped","utc":"%s","unit":"%s","InvocationID":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$CUNIT" "$ID" >> "$CREC"
  # the journal AFTER the stop, filtered on the run's own id with BOTH
  # fields, with the retention state measured beside it (R-641)
  OLDEST=$(journalctl --user --no-pager -o short-iso 2>/dev/null | head -1 | cut -d' ' -f1)
  NP=$(journalctl --user _SYSTEMD_INVOCATION_ID="$ID" --no-pager -o cat 2>/dev/null | wc -l)
  NM=$(journalctl --user USER_INVOCATION_ID="$ID" --no-pager -o cat 2>/dev/null | wc -l)
  NS=$(journalctl --user USER_INVOCATION_ID="$ID" --no-pager -o cat 2>/dev/null | grep -c -e Stopped -e Consumed)
  printf '{"event":"journal_copy","utc":"%s","taken":"AFTER the stop","InvocationID":"%s","n_payload_lines":%s,"n_manager_lines":%s,"n_stopped_or_consumed_lines":%s,"retention_oldest_entry":"%s","why_after":"the Stopped/Consumed lines are written BY the stop; a copy taken before it cannot contain them (DE 106)"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$ID" "$NP" "$NM" "$NS" "$OLDEST" >> "$CREC"
  echo "captured $CUNIT: $LS/$AS/$SS/$RS/$MS id=$ID; peak_of_record=${POR:-ABSENT} ($POR_SRC); leaf_status=$LEAF_STATUS; systemd property=$MP; stdout $OUTF sha=$OUTSHA lines=$OUTL; stopped; journal by id payload=$NP manager=$NM stopped_or_consumed=$NS"
  exit 0
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
      # THE SAMPLER GOES UP WITH THE RUN (BE 87 (a)). It is detached with
      # setsid so it outlives this poll: the peak of record can only be
      # read while the payload is alive, and a sampler tied to the caller
      # would die with the shell -- the same defect the scope form had.
      setsid "$SELFP" --sample "$PUNIT" "${BE_SAMPLE_INTERVAL:-5}" \
        </dev/null >/dev/null 2>&1 &
      echo "LOCK TAKEN $(date -u +%Y-%m-%dT%H:%M:%SZ) attempt $N after $((N-1)) refusals; InvocationID $ID; leaf sampler started (every ${BE_SAMPLE_INTERVAL:-5}s)"
      exit 0
    elif [ "$SETTLE" = "1" ] && [ "$MS" = "0" ] && [ "$RS" = "success" ]; then
      # A RUN THAT FINISHES INSIDE THE SETTLE WINDOW. Measured at 16:34:29Z:
      # the structure verification took the lock and completed in 4.3 s, and
      # this poll called it UNEXPECTED because it had no branch for success
      # -- only for refusal and for still-running. The run was unaffected;
      # the classification was wrong, which is its own defect.
      echo "TOOK THE LOCK AND FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ) attempt $N after $((N-1)) refusals; Result=$RS ExecMainStatus=$MS InvocationID=$ID"
      exit 0
    elif [ "$SS" = "exited" ] && [ "$RS" = "success" ] && [ "$MS" = "0" ]; then
      # (b) A RUN THAT HAD ALREADY FINISHED BEFORE THIS POLL RAN. Measured
      # at 01:30:18Z on be72struct: the verification took the lock and was
      # done in 3.1 s, so by the first poll the unit sat `active/exited,
      # success, 0` -- and because the relaunch inside the attempt cannot
      # start a unit name that is already loaded, no NEW exit row appeared
      # and SETTLE stayed 0. Every branch above wants SETTLE, so the poll
      # fell through to UNEXPECTED and exited 2: a SUCCESS reported as an
      # anomaly. Under RemainAfterExit a finished run IS `active/exited`,
      # and the five fields already say it succeeded -- that is the whole
      # verdict, with or without a settle.
      echo "ALREADY FINISHED $(date -u +%Y-%m-%dT%H:%M:%SZ) attempt $N: the unit had exited before this poll ran; Result=$RS ExecMainStatus=$MS SubState=$SS InvocationID=$ID"
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
# BE 110: THE UNIT'S MEMORY ENVELOPE IS A DECLARED NUMBER WITH A REASON, not
# a constant someone edited. Default unchanged at 8G, so every existing form
# behaves exactly as before; `BE_MEMORY_MAX` overrides it and the value plus
# its basis are written into the run record, because a raised ceiling that
# nobody can trace back to a measurement is the thing rule 20 forbids.
# The basis (BE 104/110): the cgroup LEAF tracks TAPE ROWS at ~15,489 B/row,
# measured on the one un-thrashed build (09-03 at L=250: 8,430,505,984 B over
# 544,286 rows). The heaviest design day is 09-04 at 638,602 rows -> an
# implied 9,891,376,928 B, which is why an 8 GiB cap pinned it with 775
# reclaim events. research.slice allows 15,032,385,536 B, so any override
# must stay under that.
MEMMAX="${BE_MEMORY_MAX:-8G}"
MEMBASIS="${BE_MEMORY_MAX_BASIS:-the launcher default, unchanged since R-551}"
# BE 142, USER REVIEW: THE SCOPE CLAIM IS DERIVED, NOT TYPED. The EV21
# launches carried `BE_MEMORY_MAX_BASIS='R-837, EV20 queue only: ...'`
# because a human typed the caller's rationale and the revision moved
# under it -- the same class as a `_PARAMS_V20` protocol string that no
# longer matches what it labels. The RATIFICATION's scope is a constant
# (R-837 ratified the envelope for the EV20 queue only, expiring with it);
# the LAUNCH's revision is read from this invocation's own arguments; and
# whether they match is COMPUTED. A typed rationale may still be passed,
# and is recorded verbatim, but it is no longer the only scope claim.
REV_LAUNCHED=""
_prev=""
for _a in "$@"; do
  [ "$_prev" = "--artifact-revision" ] && REV_LAUNCHED="$_a"
  _prev="$_a"
done
[ -n "$REV_LAUNCHED" ] || REV_LAUNCHED="NONE_PASSED"
RATIFIED_FOR="EV20 queue only (R-837, ratified on its recorded basis and expiring with that queue)"
if [ "$REV_LAUNCHED" = "EV20" ]; then SCOPE_MATCH=true; else SCOPE_MATCH=false; fi
printf '{"event":"launch","utc":"%s","unit":"%s","payload":"%s","args":"%s","tip":"%s","worktree":"%s","lock":"%s","conflict_rc":%s,"declaration":"%s","stdout_file":"%s","stdout_note":"the payload'"'"'s stdout is appended to this file AS IT RUNS; the journal keeps stderr and the manager lines. A record that lives only in a rotating journal is not a record (rule 20)."}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$UNIT" "$TARGET" "$*" "$TIP" "$WT" "$LOCK" \
  "$LOCK_CONFLICT_RC" "$DECL" \
  "$REPO/data/pm_5min/derived/be_heavy_run_stdout_${UNIT}.log" >> "$REC"
printf '{"event":"memory_envelope","utc":"%s","unit":"%s","MemoryMax":"%s","basis":"%s","artifact_revision_launched":"%s","envelope_ratified_for":"%s","ratification_scope_matches_this_launch":%s,"scope_note":"the NUMBERS are unchanged and sufficient; this records that the ratification'"'"'s scope and this launch'"'"'s revision are read from different places -- the first a constant, the second THIS invocation'"'"'s own arguments -- so a typed rationale can no longer be the only scope claim (BE 142, user review)","slice_MemoryMax_bytes":%s,"caps_are_never_raised":"rule 8 / rule 20 / R-174 say a cap is never raised; this override was DISPATCHED by the coordinator at BE 104 and reaffirmed at BE 110 and wants a register amendment, which BE has flagged and not assumed."}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$UNIT" "$MEMMAX" "$MEMBASIS" \
  "$REV_LAUNCHED" "$RATIFIED_FOR" "$SCOPE_MATCH" \
  "$(systemctl --user show research.slice -p MemoryMax --value 2>/dev/null || echo null)" >> "$REC"

# (c) THE PAYLOAD'S OWN RECORD GOES TO A FILE, AT RUN TIME (BE 72 finding
# 3). `--verify-structure` prints its verification record to stdout and the
# launcher redirected nothing, so the only copy lived in the journal -- and
# the journal is not the record (rule 20, R-641): it rotates, and BE had to
# read the record back out of it by invocation id and write the file by
# hand. `StandardOutput=append:` puts every stdout byte in a file the
# moment it is written. (`tee:` is not a valid systemd value here --
# measured on systemd 255: "Invalid StandardOutput setting" -- so stderr
# and the manager lines remain the journal's and stdout is the file's.)
OUTF="$REPO/data/pm_5min/derived/be_heavy_run_stdout_${UNIT}.log"
exec systemd-run --user --unit="$UNIT" --slice=research.slice \
  -p MemoryMax="$MEMMAX" -p CPUQuota=100% -p RemainAfterExit=yes \
  -p StandardOutput=append:"$OUTF" \
  --setenv=BE_RECORD="$REC" \
  --setenv=BE_REFUSAL_FILE="${BE_REFUSAL_FILE:-$REPO/data/pm_5min/derived/be_heavy_run_refusal_${UNIT}.txt}" \
  --setenv=PM_DATA_ROOT="$REPO" \
  --setenv=BE_HEAVY_LOCK="$LOCK" \
  --working-directory="$WT" \
  -- "$SELF" --inner --lock "$LOCK" "$PY" "$TARGET" "$@"
