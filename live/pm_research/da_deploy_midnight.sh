#!/bin/bash
# DA: THE DEPLOY ACT for da-midnight-verify.service. The ONLY writer of the
# deploy record.
#
# WHY THIS EXISTS (R-549(E) item 4). The unit's ExecStart is an absolute path
# into /home/yuqing/ctaNew -- a tree no seat owns. "Deployed" had no meaning
# beyond "whatever is on disk at 00:06:00Z", so the unit could run stale or
# half-landed code and report GREEN. Deployment is now an ACT that leaves a
# RECORD, and the unit refuses (rc 7) if what is on disk is not that record.
#
# WHAT IT REFUSES, AND WHY EACH ONE. The nightly gate is DIGEST-ONLY -- it
# must not fire on a register append. This act is the opposite: it is
# deliberate, infrequent and cheap to satisfy, so it is STRICT. Strict at the
# deliberate act, precise at the automatic one.
#   * the guard's own selftest is not green   -- deploying an unproven gate
#   * live/pm_research is dirty               -- the recorded digests would
#                                                belong to no commit
#   * HEAD is BEHIND origin/mm-research       -- deploying code the branch has
#                                                already moved past
#   * HEAD is AHEAD of origin/mm-research     -- the recorded commit could not
#                                                be resolved by anyone else
#
# USAGE
#   live/pm_research/da_deploy_midnight.sh                 # deploy the real thing
#   live/pm_research/da_deploy_midnight.sh --tree T --no-install   # falsifier
set -u

TREE=/home/yuqing/ctaNew
INSTALL=1
BY="${DA_DEPLOY_BY:-DA seat (pm-da)}"
while [ $# -gt 0 ]; do
  case "$1" in
    --tree) TREE="$2"; shift 2 ;;
    --no-install) INSTALL=0; shift ;;
    --by) BY="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

PY=/home/yuqing/pricer-sol/venv/bin/python3
PR="$TREE/live/pm_research"
GUARD="$PR/da_deploy_guard.py"
REC="$PR/systemd/da_deploy_record.json"
MAN="$PR/systemd/da_deploy_record.sha256"
UNIT_SRC="$PR/systemd/da-midnight-verify.service"
TIMER_SRC="$PR/systemd/da-midnight-verify.timer"
UNIT_DST=~/.config/systemd/user/da-midnight-verify.service
TIMER_DST=~/.config/systemd/user/da-midnight-verify.timer

# THE CLOCK IS READ ONCE, HERE, AND CARRIED. Not re-derived per field: two
# fields stamped from two `date` calls describe two different instants and
# read as one (SEAT_PROTOCOL 12).
NOW="$(date -u +%FT%TZ)"

fail() { echo "DEPLOY REFUSED: $*" >&2; exit 1; }

[ -d "$TREE/.git" ] || [ -f "$TREE/.git" ] || fail "$TREE is not a git tree"
[ -f "$GUARD" ] || fail "no guard at $GUARD"
[ -f "$UNIT_SRC" ] || fail "no unit source at $UNIT_SRC"

# (1) THE GATE MUST BE GREEN BEFORE IT IS TRUSTED WITH THE NIGHT. A guard
# deployed without its own falsifiers passing is rule 15's zero from an
# instrument that never proved it can fire.
if ! "$PY" "$GUARD" --selftest > /tmp/da_deploy_guard_selftest.$$ 2>&1; then
  echo "--- guard selftest output ---" >&2
  cat /tmp/da_deploy_guard_selftest.$$ >&2
  rm -f /tmp/da_deploy_guard_selftest.$$
  fail "da_deploy_guard.py --selftest is NOT green"
fi
_ngreen="$(grep -c '^ok   ' /tmp/da_deploy_guard_selftest.$$)"
rm -f /tmp/da_deploy_guard_selftest.$$

# (2) DIRTY. Scoped to live/pm_research: an uncommitted register entry
# elsewhere in this shared tree is not a fact about the code this unit runs,
# and refusing on it would make the act unusable for the reason it exists.
_dirty="$(git -C "$TREE" status --porcelain -- live/pm_research)"
[ -z "$_dirty" ] || fail "live/pm_research is DIRTY -- the digests would
belong to no commit:
$_dirty"

# (3) BEHIND / AHEAD. Both directions, named separately, because they are
# different mistakes.
_behind="$(git -C "$TREE" rev-list --count HEAD..origin/mm-research 2>/dev/null)"
_ahead="$(git -C "$TREE" rev-list --count origin/mm-research..HEAD 2>/dev/null)"
[ -n "$_behind" ] || fail "cannot compare with origin/mm-research (fetch first)"
[ "$_behind" -eq 0 ] || fail "the tree is $_behind commit(s) BEHIND
origin/mm-research. Deploying here would pin code the branch has moved past.
FIX: git -C $TREE pull --ff-only"
[ "$_ahead" -eq 0 ] || fail "the tree is $_ahead commit(s) AHEAD of
origin/mm-research -- the recorded commit is unpushed and could not be
resolved by anyone else. FIX: push first."

HEAD="$(git -C "$TREE" rev-parse HEAD)"

# (4) INSTALL. After the refusals, never before: an act that installs and then
# refuses has already done the thing it refuses.
if [ "$INSTALL" -eq 1 ]; then
  install -Dm644 "$UNIT_SRC" "$UNIT_DST" || fail "could not install the unit"
  install -Dm644 "$TIMER_SRC" "$TIMER_DST" || fail "could not install the timer"
  systemctl --user daemon-reload || fail "daemon-reload failed"
  systemctl --user enable --now da-midnight-verify.timer >/dev/null 2>&1 \
    || fail "could not enable the timer"
  diff -q "$UNIT_SRC" "$UNIT_DST" >/dev/null \
    || fail "the installed unit differs from the repo copy after install"
fi

# (5) THE RECORD, WRITTEN LAST, so it digests the units that were just
# installed rather than the ones that were there before.
"$PY" "$GUARD" write-record --tree "$TREE" --out "$REC" --manifest "$MAN" \
      --by "$BY" --at-utc "$NOW" || fail "could not write the record"

# (6) CONFIRMED BY THE THING THAT WILL DO THE CHECKING, not by this script's
# own belief that it wrote a good file. Both instruments, in the order the
# unit runs them.
( cd "$TREE" && sha256sum -c --status "$MAN" ) \
  || fail "the manifest just written does not verify against the tree"
"$PY" "$GUARD" check --record "$REC" --tree "$TREE" || fail "the record just
written does not check out"

echo
echo "DEPLOYED $NOW"
echo "  tree:            $TREE"
echo "  commit:          $HEAD"
echo "  guard selftest:  $_ngreen checks green"
echo "  record:          $REC"
echo "  manifest:        $MAN"
if [ "$INSTALL" -eq 1 ]; then
  systemctl --user list-timers --all 2>/dev/null \
    | grep da-midnight-verify || true
fi
echo
echo "The record is UNCOMMITTED until you land it. The guard reports"
echo "RECORD_UNCOMMITTED until then and MATCHES_LAST_COMMIT afterwards;"
echo "a record hand-edited after its commit REFUSES."
