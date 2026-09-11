#!/bin/bash
# BE 148: launch one FWD stage, with be_build_preflight AS THE FIRST STAGE --
# it refuses BEFORE the lock is offered for, so a day is never half-built and
# never built twice. A NEW file: launch_stage.sh is being read right now by
# be147frag0909's offer loop and editing a running script is its own hazard.
set -u
# ---- REVIEW 164: this launcher's OWN entry-point falsifier. Every cell
# invokes THIS FILE as a subprocess, from a cwd that is NOT the tree root,
# exactly as a chain invokes it.
if [ "${1:-}" = "--falsify" ]; then
  ME="$(readlink -f "$0")"; RC=0; T=/tmp
  _note() { if [ "$2" = "1" ]; then echo "  PASS  $1"; else echo "  FAIL  $1"; RC=1; fi; }
  # (1) a stage whose INPUT is missing: exit 5, and NO UNIT created.
  U=be158falsify_$$
  # REVIEW 205 3: --dry-run. WITHOUT it these cells reach the wrapper and
  # LAUNCH A REAL BOOK BUILD the moment 20260910's tape exists -- their only
  # safety was the absence of a file the programme is actively creating. The
  # refusal path is byte-identical either way (--dry-run sits AFTER the
  # preflight and BEFORE the stage dispatch), so the assertion is unchanged
  # and the control can no longer cause the event it checks for.
  ( cd "$T" && bash "$ME" --dry-run book 20260910 "$U" >/dev/null 2>&1 ); rc=$?
  ls=$(systemctl --user show "$U" -p LoadState --value 2>/dev/null)
  _note "entry point refuses a stage whose input is missing with exit 5"         "$([ "$rc" = "5" ] && echo 1 || echo 0)"
  _note "and it creates NO UNIT -- the lock was never offered for"         "$([ "$ls" = "not-found" ] && echo 1 || echo 0)"
  # (2) the caller's BE_WORKTREE is IRRELEVANT: this launcher exports it
  #     itself, by design. The unset known-bad belongs to the PREFLIGHT
  #     entry point, which is where it is driven.
  U2=be158falsify2_$$
  ( cd "$T" && env -u BE_WORKTREE bash "$ME" --dry-run book 20260910 "$U2" >/dev/null 2>&1 ); rc2=$?
  _note "with BE_WORKTREE UNSET in the caller it still reaches the same refusal -- the launcher sets it (by design)"         "$([ "$rc2" = "5" ] && echo 1 || echo 0)"
  # (3) an unknown stage is refused before anything else.
  ( cd "$T" && bash "$ME" bogus 20260910 x >/dev/null 2>&1 ); rc3=$?
  _note "an unknown stage exits 2" "$([ "$rc3" = "2" ] && echo 1 || echo 0)"
  # (4) THE ADMIT DIRECTION, via --dry-run so a control cannot build a day.
  # BE 203: the fixture was the literal 20260910, a FOUND STATE of the disk --
  # and it went red the moment this programme built that fragment, exactly the
  # failure the preflight's own docstring records (REVIEW 205 finding 1). The
  # day is COMPUTED now: the first closed day whose book is absent and whose
  # inputs are present. If none exists the cell says so instead of passing.
  # The candidate is chosen by STAGE 0's OWN VERDICT (0 would-fail, 0 not-yet),
  # not by which files happen to be on disk: 20260903 has a tape, a spine and
  # no book, yet is unbuildable (41 missing interior windows). This is not
  # circular -- the selector asks the preflight, while the cells below assert
  # the LAUNCHER's behaviour AFTER stage 0: the guard, the echo, and no unit.
  CLEANDAY=""
  for _d in 20260903 20260904 20260905 20260906; do
    _j=$( cd /home/yuqing/ctaNew && BE_WORKTREE=/home/yuqing/ctaNew-wt-fwd \
            PM_DATA_ROOT=/home/yuqing/ctaNew python3 \
            live/pm_research/be_build_preflight.py --stage book "$_d" 2>/dev/null \
          | grep -o '"n_would_fail": *[0-9]*, *"n_not_yet": *[0-9]*' | tail -1 )
    case "$_j" in *'"n_would_fail": 0'*'"n_not_yet": 0'*) CLEANDAY="$_d"; break ;; esac
  done
  _note "a fixture day with satisfied preconditions exists to drive ADMIT with" \
        "$([ -n "$CLEANDAY" ] && echo 1 || echo 0)" 
  out=$( cd "$T" && bash "$ME" --dry-run book "${CLEANDAY:-00000000}" dryunit 2>&1 ); rc4=$?
  _note "entry point ADMITS a stage whose preconditions hold (rc 0)"         "$([ "$rc4" = "0" ] && echo 1 || echo 0)"
  _note "and says so rather than launching"         "$(printf '%s' "$out" | grep -q 'WOULD LAUNCH' && echo 1 || echo 0)"
  ls4=$(systemctl --user show dryunit -p LoadState --value 2>/dev/null)
  _note "the dry run creates no unit either"         "$([ "$ls4" = "not-found" ] && echo 1 || echo 0)"
  # (5) THE HAZARD ITSELF, driven on a day whose tape EXISTS: the cell that
  #     could have built a day must now only echo.
  U3=be163falsify3_$$
  out3=$( cd "$T" && bash "$ME" --dry-run book "${CLEANDAY:-00000000}" "$U3" 2>&1 ); rc5=$?
  ls3=$(systemctl --user show "$U3" -p LoadState --value 2>/dev/null)
  _note "on a day whose tape EXISTS the cell echoes instead of building"         "$([ "$rc5" = "0" ] && printf '%s' "$out3" | grep -q 'WOULD LAUNCH' && echo 1 || echo 0)"
  # BE 203: the NOT_YET gate, both ways, on the real entry point.
  out8=$( cd "$T" && bash "$ME" --dry-run frag 20260911 notyetunit 2>&1 ); rc8=$?
  _note "an OPEN day is REFUSED (exit 5) -- NOT_YET no longer reads as admitted"         "$([ "$rc8" = "5" ] && printf '%s' "$out8" | grep -q 'REFUSED PREFLIGHT_NOT_YET' && echo 1 || echo 0)"
  _note "and the NOT_YET gate does NOT claim a would-fail refusal as its own"         "$( cd "$T" && bash "$ME" --dry-run frag 20260902 nu2 2>&1 | grep -q 'REFUSED BY PREFLIGHT' && echo 1 || echo 0)"
  for u in notyetunit nu2; do systemctl --user reset-failed "$u" >/dev/null 2>&1; done
  _note "and still creates no unit -- the control cannot cause the event"         "$([ "$ls3" = "not-found" ] && echo 1 || echo 0)"
  # (6) rebuild-identity refuses when there is nothing to be identical TO,
  #     and its dry path moves nothing and launches nothing.
  ( cd "$T" && bash "$ME" rebuild-identity 20261231 nounit >/dev/null 2>&1 ); rc6=$?
  _note "rebuild-identity refuses with no incumbent (exit 6), builds nothing"         "$([ "$rc6" = "6" ] && echo 1 || echo 0)"
  # BE 188: this cell used to assert an ADMIT on 20260908, and went RED the
  # moment its own programme created that day's scratch rebuild -- the refusal
  # it hit IS rule 13 working. Both directions now, on the same real day.
  out7=$( cd "$T" && BE_REBUILD_SCRATCH="$T/be188scratch_$$" bash "$ME" --dry-run rebuild-identity 20260908 nounit2 2>&1 ); rc7=$?
  _note "rebuild-identity --dry-run echoes the incumbent sha and moves nothing"         "$([ "$rc7" = "0" ] && printf '%s' "$out7" | grep -q 'WOULD LAUNCH: rebuild-identity' && echo 1 || echo 0)"
  out7b=$( cd "$T" && bash "$ME" --dry-run rebuild-identity 20260908 nounit3 2>&1 ); rc7b=$?
  _note "and REFUSES (exit 6) rather than overwrite an existing rebuild -- rule 13"         "$([ "$rc7b" = "6" ] && printf '%s' "$out7b" | grep -q 'already exists (rule 13)' && echo 1 || echo 0)"
  [ -d "$T/be188scratch_$$" ] && rmdir "$T/be188scratch_$$" 2>/dev/null
  _note "and the dry run created no scratch directory either"         "$([ ! -e "$T/be188scratch_$$" ] && echo 1 || echo 0)"
  _note "and the incumbent 09-08 book is still in place after the dry run"         "$([ -f /home/yuqing/ctaNew/data/pm_5min/derived/be_daybook_20260908_btc__L250ms__FWD1.pkl ] && echo 1 || echo 0)"
  for u in "$U" "$U2" "$U3" dryunit; do systemctl --user reset-failed "$u" >/dev/null 2>&1; done
  echo "{\"falsifier\": \"launch_stage2\", \"n\": 17, \"failed\": $RC}"
  exit "$RC"
fi
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi

# ---- rebuild-identity <day> <unit>: THE BUILD-PATH FALSIFIER (BE 166).
# Rebuild a day that builds under BOTH the old and the new ruled set and
# require the book to be BIT-IDENTICAL. The rebuild must use the SAME
# revision, because `artifact_revision` is written into the pickled header
# (be_daybook_build.py:1655) -- a rebuild under any other revision differs by
# construction. So the incumbent is MOVED under rule 13, digests recorded on
# both sides, and restored if anything goes wrong before the build starts.
if [ "${1:-}" = "rebuild-identity" ]; then
  shift
  DAY="${1:?day}"; UNIT="${2:?unit}"
  WT=/home/yuqing/ctaNew-wt-fwd
  D=/home/yuqing/ctaNew/data/pm_5min/derived
  SCRATCH="${BE_REBUILD_SCRATCH:-/home/yuqing/ctaNew/data/pm_5min/derived/rebuild_identity}"
  # BE 188: override moves only the SCRATCH target; $B/$R below are the landed
  # pair and are NOT overridable -- a stray override cannot reach a landed book.
  B=$D/be_daybook_${DAY}_btc__L250ms__FWD1.pkl
  R=$D/be_daybook_receipt_${DAY}_btc__L250ms__FWD1.json
  export BE_WORKTREE="$WT" BE_MEMORY_MAX=11869652313 BE_POLL_CEILING="${BE_POLL_CEILING:-100000}"
  [ -f "$B" ] && [ -f "$R" ] || {
    echo "REFUSED REBUILD_IDENTITY_NO_INCUMBENT: $B and $R must both exist -- there is nothing to be identical TO"; exit 6; }
  SHA=$(sha256sum "$B" | cut -d' ' -f1)
  OB=$SCRATCH/be_daybook_${DAY}_btc__L250ms__FWD1.rebuild.pkl
  OR=$SCRATCH/be_daybook_receipt_${DAY}_btc__L250ms__FWD1.rebuild.json
  echo "incumbent sha256 $SHA ($(stat -c %s "$B") bytes); rebuild -> $OB"
  [ -e "$OB" ] && { echo "REFUSED: $OB already exists (rule 13)"; exit 6; }
  if [ "$DRY" = "1" ]; then
    echo "WOULD LAUNCH: rebuild-identity $DAY as $UNIT -> $OB   [--dry-run: the landed book is NOT touched, nothing launched]"
    exit 0
  fi
  mkdir -p "$SCRATCH"
  # the payload is an ABSOLUTE path into the shared tree (an instrument), and
  # the unit's cwd is wt-fwd, from which the driver imports the PINNED builder
  # and proves it did.
  bash /home/yuqing/ctaNew/live/pm_research/be_heavy_run.sh --poll "$UNIT" \
       /home/yuqing/ctaNew/live/pm_research/be_rebuild_identity.py \
       --build "$DAY" "$WT" "$OB" "$OR"
  _f() { systemctl --user show "$1" -p "$2" | sed "s/^$2=//"; }
  for _i in $(seq 1 400); do [ "$(_f "$UNIT" SubState)" != "running" ] && break; sleep 20; done
  echo "$UNIT -> LoadState=$(_f "$UNIT" LoadState) SubState=$(_f "$UNIT" SubState) ExecMainStatus=$(_f "$UNIT" ExecMainStatus) InvocationID=$(_f "$UNIT" InvocationID)"
  if [ "$(_f "$UNIT" LoadState)" != "loaded" ] || [ "$(_f "$UNIT" ExecMainStatus)" != "0" ]; then
    echo "REFUSED: the rebuild did not exit 0 -- the landed book was never touched"; exit 1
  fi
  NEW=$(sha256sum "$OB" | cut -d' ' -f1)
  echo "incumbent $SHA"
  echo "rebuilt   $NEW"
  cd /home/yuqing/ctaNew && PM_DATA_ROOT=/home/yuqing/ctaNew \
    python3 live/pm_research/be_rebuild_identity.py "$DAY" "$SHA" "$R" --rebuilt "$OB" --rebuilt-receipt "$OR"
  rc=$?
  [ "$rc" -eq 0 ] && echo "REBUILD IDENTICAL -- the re-pin admitted the days and moved nothing else" \
                  || echo "REFUSED REBUILT_BOOK_NOT_IDENTICAL -- the landed book stands; the re-pin did more than extend a set"
  exit "$rc"
fi
# BE 203: the gap-windows emit, as a STAGE rather than a seat remembering.
# 09-10's artifact was never published because no dispatch named it, and DE's
# stage 0 blocked for 17 minutes with every other row green. The fragment arm
# below calls this; it is also directly drivable, which is how it is falsified.
if [ "${1:-}" = "gap-windows" ]; then
  shift; DAY="${1:?day}"
  echo "--- gap-windows emit for $DAY ---"
  systemctl --user reset-failed "gapwin${DAY}" >/dev/null 2>&1
  systemd-run --user --unit="gapwin${DAY}" --slice=research.slice \
    -p MemoryMax=8G -p CPUQuota=100% -p RemainAfterExit=yes \
    -p WorkingDirectory=/home/yuqing/ctaNew \
    --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew \
    -p StandardOutput=append:/home/yuqing/ctaNew/data/pm_5min/derived/be202gapwin.log \
    -p StandardError=append:/home/yuqing/ctaNew/data/pm_5min/derived/be202gapwin.log \
    -- /home/yuqing/pricer-sol/venv/bin/python3 \
       live/pm_research/be_gap_windows.py "$DAY" >/dev/null 2>&1 || {
         echo "REFUSED GAP_WINDOWS_LAUNCH_FAILED"; exit 7; }
  _g() { systemctl --user show "gapwin${DAY}" -p "$1" | sed "s/^$1=//"; }
  for _i in $(seq 1 120); do [ "$(_g SubState)" != "running" ] && break; sleep 5; done
  GWF=/home/yuqing/ctaNew/data/pm_5min/derived/be137_gap_windows_${DAY}.json
  ST=$(_g ExecMainStatus)
  if [ "$(_g LoadState)" = "loaded" ] && [ "$ST" = "0" ]; then
    echo "gap-windows emitted: $(stat -c %s "$GWF") B sha $(sha256sum "$GWF" | cut -c1-16)"
    exit 0
  fi
  # exit 3 is the producer's rule-13 refusal to overwrite. On a re-run of a day
  # whose spine is already published that is the CORRECT outcome, not a
  # failure -- the stage's post-condition is THE ARTIFACT IS PRESENT, not
  # THE ARTIFACT WAS WRITTEN BY THIS INVOCATION. Distinguishing them matters:
  # asserting the sha is unchanged after a refusal proves nothing at all.
  if [ "$ST" = "3" ] && [ -f "$GWF" ]; then
    echo "gap-windows already published, NOT overwritten (rule 13): $(stat -c %s "$GWF") B sha $(sha256sum "$GWF" | cut -c1-16)"
    exit 0
  fi
  echo "REFUSED GAP_WINDOWS_DID_NOT_EXIT_0 (LoadState=$(_g LoadState) status=$ST)"
  exit 7
fi
STAGE="${1:?stage: frag|tape|book}"; DAY="${2:?day}"; UNIT="${3:?unit}"
WT=/home/yuqing/ctaNew-wt-fwd
PIN=7ed5a9015f75de64feeeeaad21d97e4eecc2b15c
D=/home/yuqing/ctaNew/data/pm_5min/derived
export BE_WORKTREE="$WT"          # BEFORE the preflight, which checks it
export BE_MEMORY_MAX=11869652313
export BE_MEMORY_MAX_BASIS="the envelope the coordinator used for p003fwd0907d, read from its own run record"
export BE_POLL_CEILING="${BE_POLL_CEILING:-250}"

case "$STAGE" in
  frag) MOD=be_gate1_fragment.py;   PFSTAGE=fragment
        OUT=$D/harmful_exposure_rows_v3_gate1_${DAY}_btc.json; ARGS="--day $DAY";;
  tape) MOD=be_gate1_state_tape.py; PFSTAGE=tape
        OUT=$D/phase2_state_tape_gate1_${DAY}_btc.json;        ARGS="--day $DAY";;
  book) MOD=be_daybook_build.py;    PFSTAGE=book
        OUT=$D/be_daybook_${DAY}_btc__L250ms__FWD1.pkl
        ARGS="--day $DAY --placement-latency-ms 250 --artifact-revision FWD1";;
  *) echo "REFUSED UNKNOWN_STAGE $STAGE"; exit 2;;
esac

# ---- STAGE 0: THE PREFLIGHT. Refuses before the lock, by name.
echo "--- preflight: $DAY $PFSTAGE ---"
PFOUT=$(mktemp); PFRC=0
( cd /home/yuqing/ctaNew && PM_DATA_ROOT=/home/yuqing/ctaNew \
    python3 live/pm_research/be_build_preflight.py --stage "$PFSTAGE" "$DAY" ) \
  > "$PFOUT" 2>&1 || PFRC=$?
cat "$PFOUT"
if [ "$PFRC" != "0" ]; then
  rm -f "$PFOUT"
  echo "REFUSED BY PREFLIGHT -- the lock was never offered for. Nothing launched."
  exit 5
fi
# BE 203: NOT_YET IS ALSO A REFUSAL HERE. The preflight returns `1 if fails
# else 0`, so a row that is NOT_YET -- notably "day is closed (+markout)" --
# leaves rc 0 and this launcher proceeded. Measured 20:29Z: a --dry-run of
# 20260911's fragment printed NOT_YET:day closes 2026-09-12T00:00:00Z and then
# WOULD LAUNCH, with rc 0. A real launch would have fragmented an OPEN day.
# The preflight's own semantics are left alone (other callers read that rc);
# the refusal belongs to the thing that takes the lock.
NOTYET=$(grep -o '"n_not_yet": *[0-9]*' "$PFOUT" | tail -1 | grep -o '[0-9]*$')
rm -f "$PFOUT"
if [ -n "${NOTYET:-}" ] && [ "$NOTYET" -gt 0 ]; then
  echo "REFUSED PREFLIGHT_NOT_YET ($NOTYET row(s)) -- a precondition has not"
  echo "arrived yet (an open day, an absent mask). The lock was never offered"
  echo "for. Nothing launched."
  exit 5
fi
# ---- the launcher's own guards, kept: they are cheap and they are the last
#      word between the preflight and the emit.
H=$(git -C "$WT" rev-parse HEAD)
# BE 188: was `[ "$H" = "$PIN" ]`. That equality is a LITERAL THAT MUST TRACK A
# MOVING THING: BE 163 ruled the tree rule to be DESCENDANT-OF-PIN + the five
# pinned digests, the tree has legitimately advanced four times since 7ed5a90,
# and the equality silently refused a 09-09 book the preflight admitted 25/25.
# The guard is one file with its own falsifier (8/8, both directions).
bash /home/yuqing/ctaNew/live/pm_research/be_tree_pin_guard.sh "$WT" "$PIN" || exit 4
[ -e "$OUT" ] && { echo "REFUSED OUTPUT_EXISTS: $OUT"; exit 4; }
if [ "$DRY" = "1" ]; then
  echo "WOULD LAUNCH: $UNIT ($STAGE $DAY) -> $MOD $ARGS   [--dry-run: nothing launched]"
  exit 0
fi
echo "preflight clean -- offering for the lock: $UNIT ($STAGE $DAY)"
bash /home/yuqing/ctaNew/live/pm_research/be_heavy_run.sh --poll "$UNIT" "$MOD" $ARGS
rc=$?

# ---- THE STAGE AFTER EVERY BOOK: the per-window generation census (BE 152).
# `--poll` returns when the LOCK IS TAKEN, not when the build finishes, so the
# census must wait for the UNIT -- a seat's own driver once `systemctl stop`ped
# a live build 15 s in by confusing the two. Only the book stage waits; frag
# and tape return as before.
# BE 203: the FRAGMENT arm now waits too, and publishes the gap-windows
# artifact itself. Before this it returned at LOCK TAKEN and the artifact was
# a separate act -- which is exactly how 09-10's went missing.
if [ "$STAGE" = "frag" ] && [ "$rc" -eq 0 ]; then
  _f() { systemctl --user show "$1" -p "$2" | sed "s/^$2=//"; }
  for _i in $(seq 1 400); do
    [ "$(_f "$UNIT" SubState)" != "running" ] && break
    sleep 20
  done
  if [ "$(_f "$UNIT" LoadState)" = "loaded" ] && [ "$(_f "$UNIT" ExecMainStatus)" = "0" ]; then
    bash "$0" gap-windows "$DAY"
  else
    echo "fragment unit did not exit 0 -- gap-windows NOT emitted (a spine off a failed fragment certifies nothing)"
  fi
fi
if [ "$STAGE" = "book" ] && [ "$rc" -eq 0 ]; then
  _f() { systemctl --user show "$1" -p "$2" | sed "s/^$2=//"; }
  for _i in $(seq 1 400); do
    [ "$(_f "$UNIT" SubState)" != "running" ] && break
    sleep 20
  done
  if [ "$(_f "$UNIT" LoadState)" = "loaded" ] && [ "$(_f "$UNIT" ExecMainStatus)" = "0" ]; then
    echo "--- book landed; per-window generation census for $DAY ---"
    # NOT under the heavy lock: measured 3.89 GiB leaf against a 14.00 GiB
    # slice, so it coexists with a valuation under the 12 GB rule. It runs
    # INSIDE research.slice so it stays accounted and capped.
    systemctl --user reset-failed "census${DAY}" >/dev/null 2>&1
    systemd-run --user --unit="census${DAY}" --slice=research.slice       -p MemoryMax=6G -p CPUQuota=100% -p RemainAfterExit=yes       -p WorkingDirectory=/home/yuqing/ctaNew       --setenv=PM_DATA_ROOT=/home/yuqing/ctaNew       -p StandardOutput=append:/home/yuqing/ctaNew/data/pm_5min/derived/be152census.log       -- /home/yuqing/pricer-sol/venv/bin/python3          live/pm_research/be_book_window_census.py "$DAY" >/dev/null 2>&1
  else
    echo "book unit did not exit 0 -- census NOT run (a census of a failed build certifies nothing)"
  fi
fi
exit $rc
