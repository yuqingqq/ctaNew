#!/bin/bash
# BE 189: READ THE HEAVY LOCK -- the one the units actually take.
#
# WHY THIS EXISTS. Tracing BE 188's launcher refusal, the unit records showed
# every heavy run taking /home/yuqing/ctaNew/data/.heavy_run.lock, while this
# seat's own "lock: FREE" lines came from `flock -n` on
# /home/yuqing/ctaNew/data/pm_5min/.heavy.lock -- a path referenced by NOTHING
# in the repository, which flock CREATED on first probe (mtime 15:51:15.427Z,
# the second of the first probe) and which therefore reads FREE forever.
# A reader that cannot say HELD is not an instrument (rule 15): it had never
# proved it could fire, and it reported on five occasions.
#
# THE DESIGN RULE THIS FILE OBEYS: do not retype the path. A second literal
# would be the same defect one file further on. The path is READ OUT OF
# be_heavy_run.sh, and if that read fails this REFUSES -- it never falls back
# to a literal, because a silent fallback is how the first one survived.
set -u
RUNNER=/home/yuqing/ctaNew/live/pm_research/be_heavy_run.sh

resolve_lock() {   # echo the path be_heavy_run.sh defaults to, or refuse
  local line p
  line=$(grep -m1 -E '^LOCK="\$\{BE_HEAVY_LOCK:-[^}]+\}"' "${1:-$RUNNER}" 2>/dev/null) || true
  [ -n "$line" ] || { echo "REFUSED LOCK_PATH_UNREADABLE: no LOCK=\${BE_HEAVY_LOCK:-...} line in ${1:-$RUNNER}" >&2; return 3; }
  p=$(printf '%s' "$line" | sed -E 's/^LOCK="\$\{BE_HEAVY_LOCK:-([^}]+)\}"/\1/')
  [ -n "$p" ] && [ "${p:0:1}" = "/" ] || { echo "REFUSED LOCK_PATH_NOT_ABSOLUTE: '$p'" >&2; return 3; }
  printf '%s\n' "$p"
}

state() {          # echo FREE|HELD for a lock path
  local L="$1"
  [ -e "$L" ] || { printf 'FREE (file absent -- nothing has ever taken it)\n'; return 0; }
  if flock -n -E 75 "$L" true; then printf 'FREE\n'; else printf 'HELD\n'; fi
}

if [ "${1:-}" = "--falsify" ]; then
  RC=0; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
  _note() { if [ "$2" = "1" ]; then echo "  PASS  $1"; else echo "  FAIL  $1"; RC=1; fi; }
  # (1) it resolves the SAME path the runner defaults to
  L=$(resolve_lock); r=$?
  _note "resolves the lock path out of be_heavy_run.sh (exit 0)"            "$([ "$r" = "0" ] && echo 1 || echo 0)"
  _note "and it is /home/yuqing/ctaNew/data/.heavy_run.lock"                "$([ "$L" = "/home/yuqing/ctaNew/data/.heavy_run.lock" ] && echo 1 || echo 0)"
  # (2) THE CELL THAT WOULD HAVE CAUGHT TODAY'S DEFECT: the resolved path is
  #     one the pipeline actually references; the path this seat probed is not.
  nref=$(grep -rl -- "$L" --include='*.sh' --include='*.py' /home/yuqing/ctaNew/live/pm_research 2>/dev/null | wc -l)
  # excluding THIS file, which names the bad path only to document the defect;
  # a mention in a refusal's own prose is not a use. The exclusion is asserted
  # below to be exactly this one file, so it cannot quietly hide a real user.
  badfiles=$(grep -rl -- 'pm_5min/\.heavy\.lock' --include='*.sh' --include='*.py' /home/yuqing/ctaNew/live/pm_research 2>/dev/null)
  nself=$(printf '%s\n' "$badfiles" | grep -c 'be_lock_state\.sh$' || true)
  nbad=$(printf '%s\n' "$badfiles" | grep -v 'be_lock_state\.sh$' | grep -c . || true)
  _note "the resolved lock is referenced by >1 pipeline instrument (n=$nref)" "$([ "$nref" -gt 1 ] && echo 1 || echo 0)"
  _note "the path this seat had been probing is used by NO pipeline instrument (n=$nbad)" "$([ "$nbad" -eq 0 ] && echo 1 || echo 0)"
  _note "  (and the only file naming it is this refusal itself, n=$nself)"      "$([ "$nself" -eq 1 ] && echo 1 || echo 0)"
  _note "  (this file never passes it to flock -- it appears in prose only)"    "$(grep -nE 'flock[^#]*pm_5min/\.heavy' /home/yuqing/ctaNew/live/pm_research/be_lock_state.sh >/dev/null 2>&1 && echo 0 || echo 1)"
  # (3) BOTH DIRECTIONS on a scratch lock: it must say HELD when something holds it.
  S="$T/scratch.lock"; : > "$S"
  _note "says FREE on a lock nobody holds"                                  "$([ "$(state "$S")" = "FREE" ] && echo 1 || echo 0)"
  flock -x "$S" -c 'sleep 6' & HOLDER=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do flock -n -E 75 "$S" true || break; sleep 0.3; done
  st=$(state "$S")
  _note "says HELD while another process holds it -- it CAN fire"           "$([ "$st" = "HELD" ] && echo 1 || echo 0)"
  wait "$HOLDER" 2>/dev/null
  _note "and says FREE again once the holder exits"                         "$([ "$(state "$S")" = "FREE" ] && echo 1 || echo 0)"
  # (4) the resolution REFUSES rather than falling back to a literal
  printf 'no lock line here\n' > "$T/fake_runner.sh"
  resolve_lock "$T/fake_runner.sh" >/dev/null 2>&1; r4=$?
  _note "REFUSES (exit 3) if the runner's LOCK line cannot be read -- no silent fallback" "$([ "$r4" = "3" ] && echo 1 || echo 0)"
  echo "{\"falsifier\": \"be_lock_state\", \"n\": 10, \"failed\": $RC}"
  exit "$RC"
fi

L=$(resolve_lock) || exit 3
S=$(state "$L")
printf '%s  heavy lock %s  %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$L" "$S"
[ "$S" = "HELD" ] && command -v fuser >/dev/null 2>&1 && fuser -v "$L" 2>&1 | sed 's/^/    /'
exit 0
