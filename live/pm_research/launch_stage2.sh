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
  out=$( cd "$T" && bash "$ME" --dry-run frag 20260910 dryunit 2>&1 ); rc4=$?
  _note "entry point ADMITS a stage whose preconditions hold (rc 0)"         "$([ "$rc4" = "0" ] && echo 1 || echo 0)"
  _note "and says so rather than launching"         "$(printf '%s' "$out" | grep -q 'WOULD LAUNCH' && echo 1 || echo 0)"
  ls4=$(systemctl --user show dryunit -p LoadState --value 2>/dev/null)
  _note "the dry run creates no unit either"         "$([ "$ls4" = "not-found" ] && echo 1 || echo 0)"
  # (5) THE HAZARD ITSELF, driven on a day whose tape EXISTS: the cell that
  #     could have built a day must now only echo.
  U3=be163falsify3_$$
  out3=$( cd "$T" && bash "$ME" --dry-run book 20260909 "$U3" 2>&1 ); rc5=$?
  ls3=$(systemctl --user show "$U3" -p LoadState --value 2>/dev/null)
  _note "on a day whose tape EXISTS the cell echoes instead of building"         "$([ "$rc5" = "0" ] && printf '%s' "$out3" | grep -q 'WOULD LAUNCH' && echo 1 || echo 0)"
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
  echo "{\"falsifier\": \"launch_stage2\", \"n\": 14, \"failed\": $RC}"
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
if ! ( cd /home/yuqing/ctaNew && PM_DATA_ROOT=/home/yuqing/ctaNew \
         python3 live/pm_research/be_build_preflight.py --stage "$PFSTAGE" "$DAY" ); then
  echo "REFUSED BY PREFLIGHT -- the lock was never offered for. Nothing launched."
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
