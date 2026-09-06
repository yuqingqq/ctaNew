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
# Overridable ONLY inside a FULLY isolated rehearsal (see the pair guard
# below, which this joins). A stub verifier is the only way to exercise what
# happens when the verifier exits non-zero while still leaving a well-formed
# artifact -- the case the artifact-shape check cannot distinguish, and the
# reason `rc` had to stop being decorative.
# RR12-1 -- THE VERIFIER FOLLOWS THIS SCRIPT, AND THE UNIT PINS IT.
# The default was the canonical absolute path, so running THIS script from a
# worktree still executed the MAIN tree's verifier: the record said one tree
# and the code came from another. The default is now script-relative, so a
# worktree run uses the worktree's verifier; the unit pins the canonical path
# EXPLICITLY (Environment=DA_MIDNIGHT_VERIFY_BIN) so the production path is
# named rather than inherited from wherever the script happens to sit.
SELFDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V_DEFAULT="$SELFDIR/da_forward_day_verify.py"
V="${DA_MIDNIGHT_VERIFY_BIN:-$V_DEFAULT}"
M_DEFAULT="$SELFDIR/da_blackout_mask.py"
M="${DA_MIDNIGHT_MASK_BIN:-$M_DEFAULT}"
PY=/home/yuqing/pricer-sol/venv/bin/python3
cd /home/yuqing/ctaNew/live/pm_research || exit 3
# THE PAIR GUARD RUNS BEFORE ANY WRITE. Codex batch-2 §7: with only OUTDIR
# overridden this refused with rc=5 *after* appending its header to the
# PRODUCTION log -- 46 bytes, measured. The guard written this morning
# against "an isolation that only covers the visible half" was itself
# half-isolated: it refused the run and mutated production on the way out.
# A guard that writes before it refuses has already done the thing it
# refuses.
#
# A REHEARSAL MUST NOT WRITE PRODUCTION ARTIFACTS. DA_MIDNIGHT_LOG was
# overridable and OUTDIR was not, so a rehearsal sent its output to a scratch
# log while overwriting the real verdicts -- which is how the 00:06Z artifacts
# for 08-27 and 08-28 were silently replaced at 09:14Z. Overriding ONE of the
# two is now a REFUSAL rather than a half-isolated run: an isolation that only
# covers the visible half is worse than none, because it reads as isolated.
OUTDIR="${DA_MIDNIGHT_OUTDIR:-/home/yuqing/ctaNew/data/pm_5min/derived}"
if { [ -n "${DA_MIDNIGHT_LOG:-}" ] && [ -z "${DA_MIDNIGHT_OUTDIR:-}" ]; } || \
   { [ -z "${DA_MIDNIGHT_LOG:-}" ] && [ -n "${DA_MIDNIGHT_OUTDIR:-}" ]; }; then
  echo "REFUSED: DA_MIDNIGHT_LOG and DA_MIDNIGHT_OUTDIR must be overridden" \
       "TOGETHER or not at all. Overriding only one gives a rehearsal that" \
       "writes PRODUCTION verdicts while logging elsewhere." >&2
  exit 5
fi
# RR9-3(a): A CANONICAL WRITE MUST BE NAMED. The pair guard above catches an
# override of ONE variable; it cannot catch an override of NEITHER, which is
# what a MIS-NAMED override produces (`OUTDIR=`/`LOG=` instead of
# `DA_MIDNIGHT_OUTDIR=`/`DA_MIDNIGHT_LOG=`). That reads as an ordinary
# production run and writes the canonical directory -- exactly what happened
# at 10:16Z on 2026-09-02, replacing two 00:06Z verdicts. Mis-naming is the
# likelier operator error and it was the one shape the guard could not see.
#
# TWO INDEPENDENT WAYS TO BE ALLOWED, deliberately, because ONE new required
# variable on a nightly governed path is itself an outage risk: if the unit
# lost it, every future 00:06Z verdict would refuse and nothing would run to
# say so.
#   (1) IDENTITY -- the cgroup says this process really is
#       da-midnight-verify.service. An identity, not an assertion, and the
#       same test `write_reason` already uses.
#   (2) DECLARATION -- DA_MIDNIGHT_MODE=production, for a deliberate hand run.
# The unit sets (2) as well, so a run has to lose BOTH to be refused.
# A run that is neither identifies itself as nothing and REFUSES BEFORE ANY
# WRITE -- including before the log header, per this script's own lesson that
# a guard which writes before it refuses has already done the thing it
# refuses.
# R-420: WHICH LEG WOULD ADMIT THIS RUN IS COMPUTED ALWAYS, and logged.
# Two legs exist so a lost Environment= cannot cause a nightly outage -- but a
# SILENT fallback to the cgroup leg would equally hide that the Environment=
# had been lost, and the redundancy would quietly become a single point again.
# Naming the leg every time is what makes it visible. Computed outside the
# canonical branch so a REHEARSAL can exercise it too, without any canonical
# write.
case ":$(cat /proc/self/cgroup 2>/dev/null):" in
  */da-midnight-verify.service/*|*/da-midnight-verify.service:*) _cgroup=1 ;;
  *) _cgroup=0 ;;
esac
if [ "${DA_MIDNIGHT_MODE:-}" = "production" ] && [ "$_cgroup" -eq 1 ]; then
  _leg="BOTH (cgroup identity AND DA_MIDNIGHT_MODE=production)"
elif [ "${DA_MIDNIGHT_MODE:-}" = "production" ]; then
  _leg="DECLARED_ONLY (DA_MIDNIGHT_MODE=production; cgroup did NOT match)"
elif [ "$_cgroup" -eq 1 ]; then
  _leg="IDENTITY_ONLY (cgroup matched; DA_MIDNIGHT_MODE UNSET -- if the unit "
  _leg="$_leg is meant to set it, it has been lost)"
else
  _leg="NONE (neither leg) -- a canonical write would be REFUSED"
fi
_named=0
[ "$_cgroup" -eq 1 ] && _named=1
[ "${DA_MIDNIGHT_MODE:-}" = "production" ] && _named=1
if [ -z "${DA_MIDNIGHT_OUTDIR:-}" ] && [ -z "${DA_MIDNIGHT_LOG:-}" ]; then
  _admission="CANONICAL via $_leg"
  if [ "$_named" -ne 1 ]; then
    echo "REFUSED: this run would write CANONICAL verdicts into $OUTDIR but" \
         "identifies itself as neither the scheduled unit (by cgroup) nor an" \
         "explicit DA_MIDNIGHT_MODE=production hand run. A mis-named" \
         "override (OUTDIR=/LOG= instead of DA_MIDNIGHT_OUTDIR=/" \
         "DA_MIDNIGHT_LOG=) sets NEITHER variable and is indistinguishable" \
         "from a production run to the pair guard above -- which is how two" \
         "00:06Z verdicts were replaced at 10:16Z. For a rehearsal set" \
         "DA_MIDNIGHT_OUTDIR and DA_MIDNIGHT_LOG together; for a deliberate" \
         "canonical run set DA_MIDNIGHT_MODE=production." >&2
    exit 6
  fi
fi
# A PIN IS NOT A SUBSTITUTION. The unit now names the verifier explicitly, so
# DA_MIDNIGHT_VERIFY_BIN alone can no longer mean "a stub is being injected".
# The distinction is by CONTENT, not by trust: pointing at the SAME file the
# script-relative default resolves to is a pin and is admitted; pointing at a
# DIFFERENT file is a substitution and still demands full isolation.
if [ -n "${DA_MIDNIGHT_VERIFY_BIN:-}" ] && \
   { [ -z "${DA_MIDNIGHT_LOG:-}" ] || [ -z "${DA_MIDNIGHT_OUTDIR:-}" ]; } && \
   [ "$(readlink -f "$V" 2>/dev/null)" != "$(readlink -f "$V_DEFAULT" 2>/dev/null)" ]; then
  echo "REFUSED: DA_MIDNIGHT_VERIFY_BIN points at a DIFFERENT verifier than" \
       "this script's own ($V vs $V_DEFAULT) outside a fully isolated" \
       "rehearsal. A pin to the same file is fine; substituting a stub while" \
       "writing production verdicts is worse than no isolation at all." >&2
  exit 5
fi
# ============================ THE DEPLOY GATE =============================
# R-549(E) item 4, filed twice by DA and bitten once: `ExecStart` is an
# ABSOLUTE PATH INTO /home/yuqing/ctaNew, a tree NO SEAT OWNS. Every seat
# commits there and nothing checked it, so at 00:06:00Z this unit ran whatever
# that tree happened to hold -- and because `SuccessExitStatus=2` maps the
# expected open-day deferral to success, a night on stale or half-landed code
# reported GREEN and wrote a governed verdict that fed race accrual.
# MEASURED, not hypothetical: at 2026-09-06T03:56Z the deploy tree stood at
# 1d66589 while origin/mm-research stood at e25c69d -- one commit behind, and
# nothing would have said so.
#
# `deployed` is now a RECORD written ONLY by live/pm_research/
# da_deploy_midnight.sh, and this gate recomputes its digests BEFORE any
# write. Drift => rc 7, and NOTHING RUNS.
#
# 7, NOT THE 5 THE DISPATCH SUGGESTED: 5 is taken twice above (pair guard,
# verifier substitution) and 6 by the unnamed-canonical-write refusal.
# Reusing one would make two different refusals indistinguishable in
# `systemctl` -- the defect this programme keeps re-finding.
#
# TWO INSTRUMENTS, DELIBERATELY NOT THE SAME ONE. First `sha256sum -c`
# (coreutils) over a manifest of the REFUSE tier; the guard's OWN digest is in
# that manifest, so the guard is verified by something that is not the guard
# before the guard is asked anything. Then the guard itself, which adds the
# installed unit files, the REPORT tier, the interpreter and the git anchor.
#
# THE COST IS STATED: a refusal loses this night's verdict. Verified
# affordable rather than assumed -- `days_needing_verdict`
# (da_forward_day_verify.py:97) derives the day list FROM DISK, floored at the
# earliest existing verdict, and returns any day inside the range with no
# artifact or one written while it was open. The night is recovered by the
# next successful run, with a late `as_of` and a catch-up reason.
#
# LIKE EVERY GUARD ABOVE IT, THIS ONE REFUSES TO STDERR AND WRITES NOTHING.
# This file's own lesson -- "a guard that writes before it refuses has already
# done the thing it refuses" -- cost 46 measured bytes in the production log.
# So a drift refusal leaves NO line in .da_midnight_verify.log and the record
# of it is the journal: `journalctl --user -u da-midnight-verify.service`.
# That gap is a known cost of obeying the rule, stated rather than papered
# over.
DTREE="${DA_DEPLOY_TREE:-$(cd "$SELFDIR/../.." && pwd)}"
_drec="${DA_DEPLOY_RECORD:-$SELFDIR/systemd/da_deploy_record.json}"
_dman="${_drec%.json}.sha256"
_canonical=0
if [ -z "${DA_MIDNIGHT_OUTDIR:-}" ] && [ -z "${DA_MIDNIGHT_LOG:-}" ]; then
  _canonical=1
fi
# A CANONICAL RUN MAY NOT BE POINTED AT A RECORD OF SOMEONE'S CHOOSING.
# The overrides exist so the falsifier can drive both directions on a fake
# tree; allowing them on the nightly path would let a forged record certify
# anything, which is worse than having no gate. Same shape as the
# pin-vs-substitution rule for DA_MIDNIGHT_VERIFY_BIN above.
if [ "$_canonical" -eq 1 ] && \
   { [ -n "${DA_DEPLOY_RECORD:-}" ] || [ -n "${DA_DEPLOY_TREE:-}" ]; }; then
  echo "REFUSED (rc 7): DA_DEPLOY_RECORD/DA_DEPLOY_TREE are set on a" \
       "CANONICAL run. A record chosen by the caller certifies nothing." >&2
  exit 7
fi
if [ "$_canonical" -eq 1 ] || [ -n "${DA_DEPLOY_RECORD:-}" ]; then
  if [ ! -f "$_drec" ] || [ ! -f "$_dman" ]; then
    echo "DEPLOY_DRIFT (RECORD_ABSENT), rc 7: no deploy record at $_drec" \
         "(or no manifest at $_dman). This unit has never been deployed by" \
         "the explicit act, so nothing says what it should be running." \
         "FIX: run live/pm_research/da_deploy_midnight.sh." >&2
    exit 7
  fi
  if ! ( cd "$DTREE" && sha256sum -c --status "$_dman" ); then
    echo "DEPLOY_DRIFT (REFUSE_TIER_DRIFT), rc 7: the bytes this unit would" \
         "execute are NOT the bytes it was deployed at. Nothing ran." >&2
    ( cd "$DTREE" && sha256sum -c "$_dman" 2>&1 | grep -v ': OK$' ) >&2
    echo "FIX: re-run live/pm_research/da_deploy_midnight.sh. This night is" \
         "recovered by days_needing_verdict as a catch-up day." >&2
    exit 7
  fi
  _dout="$("$PY" "$SELFDIR/da_deploy_guard.py" check \
             --record "$_drec" --tree "$DTREE" 2>&1)"
  _drc=$?
  if [ "$_drc" -ne 0 ]; then
    echo "$_dout" >&2
    exit 7
  fi
  _deploy_line="$_dout"
else
  _deploy_line="SKIPPED -- isolated rehearsal, and the gate protects the
CANONICAL write. Set DA_DEPLOY_RECORD (with DA_DEPLOY_TREE) to drive it."
fi
# ==========================================================================
mkdir -p "$OUTDIR"
{
  echo
  echo "======== fired $(date -u +%FT%TZ) ========"
  echo "admission: ${_admission:-REHEARSAL (isolated overrides); canonical admission would be: $_leg}"
  echo "deploy: $_deploy_line"
  echo "verifier: $V"
  echo "verifier_sha256: $(sha256sum "$V" 2>/dev/null | cut -d" " -f1)"
  echo "script_tree: $SELFDIR"
  echo "script_tree_commit: $(git -C "$SELFDIR" rev-parse HEAD 2>/dev/null || echo unknown)"
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
# THE CLASSIFICATION, AS A NAMED FUNCTION SO IT CAN BE DRIVEN DIRECTLY
# (round 52). Inline, this decision could only be exercised by reproducing a
# whole night; as a function a falsifier can plant a CLOSED day with a real
# mask failure and show rc 4 still fires. Two arguments, no globals: closed
# ("1"/"0"/"?") and a path to the mask builder's captured output.
#
# DEFERRED requires BOTH conjuncts. `?` (verdict unreadable) is NOT "0", so an
# unreadable verdict can never be deferred -- it falls through to FAILURE,
# which is the safe direction.
classify_mask_failure() {
  _c="$1"; _l="$2"; _rc="${3:-1}"
  # ROUND 54, C-1: THIS USED TO KEY ON PROSE AND THE REVIEWER BROKE IT.
  # The old test grepped the builder's TRACEBACK for the liveness token, so a
  # log carrying BOTH the token and a real failure --
  #     RuntimeError: while handling CONTENT_LIVENESS_UNJUDGEABLE the writer died
  #     OSError: [Errno 28] No space left on device
  # -- classified as DEFERRED, mapped to rc 2, and `systemctl` reported
  # SUCCESS on a disk-full night. A wrapper that handles MaskRefused and then
  # dies produces exactly that shape, so it is not exotic.
  #
  # TWO CONJUNCTS NOW, BOTH REQUIRED, AND THEY FAIL DIFFERENTLY:
  #   (a) the builder must EMIT `MASK_STATUS=<token>` on a line of its own.
  #       A message that merely MENTIONS the token cannot forge this, because
  #       the anchor is the line start and the only writer is the builder's
  #       own except-branch (`da_blackout_mask.main`).
  #   (b) the log must carry NO failure marker -- a traceback, an OSError, a
  #       MemoryError, a kill. So a run that refused CORRECTLY and then died
  #       anyway is a FAILURE, which is what it is.
  if [ "$_c" != "0" ]; then echo "FAILURE"; return; fi
  # ROUND 56, A-1: THE EXIT CODE IS THE CONJUNCT THE LOG CANNOT FORGE.
  # The reviewer demonstrated the hole: a builder that PRINTS the token and is
  # then SIGKILLed leaves a log containing the token ALONE. `Killed` is
  # written by the PARENT SHELL to ITS stderr when it reaps the signal -- it
  # never enters `> "$_mlog" 2>&1`, which captures only the child's own
  # streams. So the `Killed` alternative in the marker regex below is DEAD
  # CODE and cannot fire, and the mixed case deferred a killed builder.
  #
  # The builder's own refusal path returns EXACTLY 1 (`da_blackout_mask.main`
  # prints MASK_STATUS and `return 1`). A signal death returns 128+N -- 137
  # for SIGKILL, 139 for SIGSEGV, 143 for SIGTERM -- and no output the child
  # writes can change the code the kernel reports to the parent. So rc is
  # checked FIRST and exactly.
  if [ "$_rc" != "1" ]; then echo "FAILURE"; return; fi
  if ! grep -qE '^MASK_STATUS=(CONTENT_LIVENESS_UNRESOLVED|CONTENT_LIVENESS_UNJUDGEABLE)$' \
       "$_l" 2>/dev/null; then
    echo "FAILURE"; return
  fi
  # `Killed` and `Segmentation fault` are RETAINED here deliberately and they
  # CANNOT MATCH from a redirected child log -- kept so that a future caller
  # which pipes the parent shell's stderr in still gets the right answer, and
  # so the dead alternative is documented rather than silently removed.
  if grep -qE '^Traceback|^[A-Za-z_.]*(Error|Exception):|Killed|Segmentation fault|MemoryError' \
       "$_l" 2>/dev/null; then
    echo "FAILURE"; return
  fi
  echo "DEFERRED"
}

# THE UNIT'S EXIT CODE, DECLARED IN ONE PLACE (round 52). `broke` is the
# process exit status and it is a THREE-VALUED health signal, not a boolean:
#
#   0  every day needing a verdict got one, and every mask landed
#   2  OPEN_DAY_MASK_DEFERRED -- an EXPECTED status. The day that just began
#      has a correct open-day verdict and no mask yet, because the frozen
#      detector cannot judge a six-minute-old day. Nothing is wrong. The unit
#      declares `SuccessExitStatus=2` so `systemctl` reports success.
#   4  INSTRUMENT FAILURE -- a verdict that is not a verdict outcome, no
#      parseable verdict at all, or a CLOSED day whose mask did not land.
#
# `broke` is RAISED and never lowered, so a 4 anywhere in the night survives a
# 2 later in the same run. Before this existed the unit exited 4 EVERY NIGHT
# on the open day's correct refusal, which is why 2 had to become a distinct
# value rather than being folded into 0: folding it would have hidden the
# deferral instead of naming it, and rule 4 says exclusions are statuses.
broke=0
# R-255(4): WHICH DAYS. Persistent=true recovers a missed night, but the day
# list used to be exactly `date -d yesterday` and `date -d today` RELATIVE TO
# RUN TIME -- so a two-day outage lost the earlier day permanently: the
# catch-up fires once and its "yesterday" is the wrong day. The list is now
# DERIVED FROM DISK (days_needing_verdict), floored at the earliest existing
# verdict so it fills holes inside the verdicted range and can never mint a
# backlog behind it.
DAYS=(); KINDS=()
if DAYS_OUT="$("$PY" "$V" days --outdir "$OUTDIR" --closed "$CLOSED" \
                    --opened "$OPENED" 2>>"$LOG")" && [ -n "$DAYS_OUT" ]; then
  while IFS=$'\t' read -r _d _k; do
    case "$_d" in "") continue ;; \#*) echo "$_d $_k" >> "$LOG"; continue ;; esac
    DAYS+=("$_d"); KINDS+=("$_k")
  done <<< "$DAYS_OUT"
fi
if [ "${#DAYS[@]}" -eq 0 ]; then
  # FALL BACK, BUT GO RED. The duty still runs on the two days it always did,
  # so a broken derivation cannot cost a night's verdict -- and `broke=4` makes
  # the unit fail anyway, because a derivation that silently degraded to the
  # old behaviour is the failure this whole change exists to remove.
  echo "DERIVATION FAILURE: could not derive the day list; falling back to" \
       "$CLOSED $OPENED and FAILING the unit so it is looked at" >> "$LOG"
  DAYS=("$CLOSED" "$OPENED"); KINDS=("closed_today" "open_today")
  broke=4
fi
NCATCH=0
for _k in "${KINDS[@]}"; do case "$_k" in catchup*) NCATCH=$((NCATCH+1));; esac; done
echo "days to verdict: ${DAYS[*]} (catch-up: $NCATCH)" >> "$LOG"
for _i in "${!DAYS[@]}"; do
  d="${DAYS[$_i]}"; kind="${KINDS[$_i]}"
  echo "---- day $d ($kind) ----" >> "$LOG"
  tmp="$(mktemp "$OUTDIR/.da_dayverdict_$d.XXXXXX.json")"
  # (e) THE EPOCH IS STATED, NEVER DEFAULTED. The verifier no longer has a
  # default because the old one (2026-08-24T15:04Z) was 3.63 days stale against
  # the live freeze commit b3f7f9f (2026-08-28T06:09Z), whose receipt says the
  # clock STARTS AT THE FREEZE COMMIT -- so pre-freeze days passed
  # entirely_post_freeze and could count toward a clock that had not started.
  #
  # THIS VALUE IS DELIBERATELY THE OLD ONE, and it is NOT DA's to change:
  # switching to the freeze-commit epoch would make 08-28 fail
  # entirely_post_freeze (the freeze landed mid-day), which materially changes
  # tonight's verdict. Preserving tonight's behaviour exactly while making the
  # value VISIBLE is the safe half; choosing the governing epoch is a ruling.
  # ESCALATED: needs a ruling before the 08-29 verdict.
  # R-240 RULED: the FREEZE-COMMIT epoch governs. b3f7f9f = 2026-08-28T06:09:00Z
  # = 1787897340. The receipt's own clause is "clock_starts: at the freeze
  # commit", so a mid-day freeze means 08-28 is NOT entirely post-freeze and
  # must not count toward the btc candidate's five days.
  # This changes the ACCRUAL flag only: race_accrual_eligible is reported
  # separately from day_quality_pass, so a healthy-but-early day reads as a
  # good day that does not count, never as a bad day.
  FREEZE_EPOCH="${DA_FREEZE_EPOCH:-1787897340}"
  # WHO and WHY, carried into the artifact. A systemd run identifies itself
  # by INVOCATION_ID; a hand run of this launcher that states no reason is
  # recorded as UNATTRIBUTED rather than refused -- refusing here would put
  # the standing nightly duty at risk of an environment assumption, and an
  # unattributed write that SAYS SO is already the honest status.
  # IDENTITY, NOT PRESENCE. The first version tested `[ -n "$INVOCATION_ID" ]`
  # -- but INVOCATION_ID is INHERITED by every child of any systemd unit, and
  # this session's own shell has one, so a hand rehearsal stamped itself
  # "scheduled nightly timer". A presence test on an inherited variable is the
  # vocabulary-vs-identity mistake in an environment variable. The cgroup path
  # names the unit this process is ACTUALLY running inside and cannot be
  # inherited from a different one.
  case ":$(cat /proc/self/cgroup 2>/dev/null):" in
    */da-midnight-verify.service/*|*/da-midnight-verify.service:*)
      REASON="scheduled unit run, da-midnight-verify.service (INVOCATION_ID=${INVOCATION_ID:-?})" ;;
    *)
      REASON="${DA_WRITE_REASON:-UNATTRIBUTED hand run of da_midnight_verify.sh}" ;;
  esac
  # A CATCH-UP DAY SAYS SO IN ITS OWN ARTIFACT. Lateness stays self-documenting
  # (Q-DA-121): the late `as_of` records WHEN, and this records WHY, so a
  # verdict produced days after its day never reads as a timely one.
  case "$kind" in
    catchup*) REASON="$REASON [catch-up after outage, $NCATCH day(s) missed; this day recovered as $kind]" ;;
  esac
  # Name the artifact being REPLACED. This path is a cache of the current
  # verdict, not a receipt: a pre-guard verdict once stood here with nothing
  # in it naming its own correction, and a log line saying "written" cannot
  # be resolved by a reader.
  "$PY" "$V" verify --day "$d" --freeze-epoch "$FREEZE_EPOCH" \
        --supersedes "$OUTDIR/da_dayverdict_$d.json" \
        --write-reason "$REASON" --out "$tmp" >> "$LOG" 2>&1
  rc=$?
  # rc GOVERNS, it does not merely get printed. It was captured here and used
  # only in the log line, so the install decision rested entirely on the
  # artifact's SHAPE -- and a non-zero exit that still left a shaped file
  # would have been installed under a log line reading "verdict artifact
  # written" beside "exit=4". Only 0 (pass) and 1 (day failed) are verdict
  # outcomes; every other code is an instrument failure, whatever is on disk.
  if [ "$rc" -ne 0 ] && [ "$rc" -ne 1 ]; then
    rm -f "$tmp"
    broke=4
    echo "exit=$rc for $d  <-- INSTRUMENT FAILURE: exit code is not a verdict outcome (0=pass, 1=day failed). NOTHING WAS VERIFIED." >> "$LOG"
  elif "$PY" -c 'import json,sys
d=json.load(open(sys.argv[1]))
sys.exit(0 if d.get("day_token")==sys.argv[2] and d.get("predicates") else 1)'        "$tmp" "$d" 2>/dev/null; then
    mv -f "$tmp" "$OUTDIR/da_dayverdict_$d.json"
    # NAME THE ARTIFACT, not just the fact that one was written. The verdict
    # path is stable, so ANY later re-run -- a rehearsal, a manual verify --
    # overwrites it while this line still reads "written", and the log then
    # describes a file that is no longer the one it describes. Observed
    # 2026-08-28: the 00:06Z artifacts for 08-27 and 08-28 were replaced at
    # 09:14Z by a hand re-run, and nothing in the log said so. The digest and
    # the artifact's own as_of let a reader CHECK rather than assume.
    _sha="$(sha256sum "$OUTDIR/da_dayverdict_$d.json" | cut -c1-16)"
    _asof="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("as_of_utc"))' "$OUTDIR/da_dayverdict_$d.json" 2>/dev/null)"
    echo "exit=$rc for $d (verdict artifact written: sha256=$_sha as_of=$_asof reason=$REASON)" >> "$LOG"
    # THE MASK LANDS IN THE SAME RUN AS THE VERDICT (round 35).
    #
    # It was never written here at ALL -- the word did not appear in this
    # script -- so a day's mask arrived whenever someone ran the builder by
    # hand, and 09-01's and 09-03's arrived roughly TEN HOURS after their
    # verdicts. A governed day is UNSCOREABLE until the mask lands, so every
    # accruing night carried a ten-hour hole for no reason at all: nothing in
    # the mask's inputs needs the wait, it reads the same closed day the
    # verdict just read. Incidental, not inherent.
    #
    # WRITTEN VIA A TEMP DIR AND MOVED, like the verdict above: a builder
    # that fails partway must not leave a half-mask at the canonical path,
    # because the consumer refuses a malformed mask but a truncated-yet-valid
    # one is worse. A mask failure does NOT undo the verdict -- the verdict
    # is correct and installed -- but it IS an instrument failure, because a
    # day that cannot be scored is not a day that was verified. Silence here
    # would re-create the hole this closes.
    #
    # ROUND 52: AN OPEN DAY'S MASK REFUSAL IS AN EXPECTED STATUS, NOT AN
    # INSTRUMENT FAILURE (rule 4) -- and the reason is that this unit was
    # RED EVERY SINGLE NIGHT.
    #
    # `days_needing_verdict` always includes the day that just began. Its
    # verdict is written correctly as an open-day write (exit=1), and then the
    # mask builder is asked for a mask of a day that is ~6 minutes old. The
    # frozen detector refuses -- CONTENT_LIVENESS_UNJUDGEABLE ("no coin had
    # enough windows for a median") or CONTENT_LIVENESS_UNRESOLVED ("a raw
    # directory but NO window files") -- which is exactly right and must not
    # be weakened. The defect was HERE: that correct refusal was classified as
    # INSTRUMENT FAILURE and set worst_instrument_rc=4.
    #
    # Measured, both nights: 2026-09-05T00:06:29Z rc=4 with 20260905's mask
    # UNRESOLVED; 2026-09-06T00:06:41Z rc=4 with 20260906's mask UNJUDGEABLE.
    # On BOTH nights the CLOSED day was perfect -- 20260905 accrued at
    # exit=0 with its mask written, G=5. So `systemctl` has shown `failed`
    # every night while nothing was wrong, and a reader cannot tell that from
    # a night when something IS. An alarm that fires every night is one that
    # gets turned off: SEAT_PROTOCOL 16's shape on this unit's own status.
    #
    # THE FIX IS IN THE CLASSIFICATION AND NOWHERE ELSE. `MaskRefused` is
    # untouched; the detector still refuses; the mask is still not written.
    # What changes is that an OPEN day refusing FOR WANT OF WINDOWS carries
    # rc 2 (OPEN_DAY_MASK_DEFERRED) which never raises `broke` to 4. A CLOSED
    # day that cannot produce a mask is still rc 4, and so is an open day that
    # fails for ANY OTHER REASON -- the two conditions are ANDed, so this
    # cannot swallow a real failure that happens to land on an open day.
    _mtmp="$(mktemp -d)"
    _mlog="$(mktemp)"
    "$PY" "$M" --day "$d" --write --outdir "$_mtmp" > "$_mlog" 2>&1
    _mrc=$?
    cat "$_mlog" >> "$LOG"
    if [ "$_mrc" -eq 0 ] && [ -s "$_mtmp/da_blackout_mask_$d.json" ]; then
      mv -f "$_mtmp/da_blackout_mask_$d.json" \
            "$OUTDIR/da_blackout_mask_$d.json"
      _msha="$(sha256sum "$OUTDIR/da_blackout_mask_$d.json" | cut -c1-16)"
      echo "mask written for $d in the same run (sha256=$_msha)" >> "$LOG"
    else
      # CLOSED-ness is read from the verdict this run just installed, not
      # guessed from the date: the artifact is the authority on its own day.
      _closed="$("$PY" -c 'import json,sys
try:
    print("1" if json.load(open(sys.argv[1])).get("day_closed_calendar") is True else "0")
except Exception:
    print("?")' "$OUTDIR/da_dayverdict_$d.json" 2>/dev/null)"
      if [ "$(classify_mask_failure "$_closed" "$_mlog" "$_mrc")" = "DEFERRED" ]; then
        if [ "$broke" -lt 2 ]; then broke=2; fi
        echo "MASK DEFERRED for $d  <-- OPEN_DAY_MASK_DEFERRED (rc=2, EXPECTED STATUS, not an instrument failure): the day is not closed and the frozen detector cannot judge it yet for want of windows. The refusal is CORRECT and the mask will land at the next 00:06Z run, when this day is closed. Counted and reported, never silent (rule 4)." >> "$LOG"
      else
        broke=4
        echo "MASK NOT WRITTEN for $d  <-- INSTRUMENT FAILURE: the verdict is installed and the day is UNSCOREABLE until a mask lands. The pair must arrive together. (day_closed_calendar=$_closed, mask rc=$_mrc)" >> "$LOG"
      fi
    fi
    rm -rf "$_mtmp"
    rm -f "$_mlog"
  else
    rm -f "$tmp"
    broke=4
    echo "exit=$rc for $d  <-- INSTRUMENT FAILURE: NO PARSEABLE VERDICT NAMING $d. NOTHING WAS VERIFIED." >> "$LOG"
  fi
done
echo "done $(date -u +%FT%TZ); worst_instrument_rc=$broke" >> "$LOG"
exit "$broke"
