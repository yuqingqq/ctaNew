#!/bin/bash
# land_entry.sh <entry.md> <N> <commit-message-file> [--no-runbook]
# Register landing with the hold, the chained commit, the post-condition, and the revert on failure (R-661/R-662/R-667/R-686).
set -u
E="$1"; N="$2"; MSG="$3"; NORB="${4:-}"; cd /home/yuqing/ctaNew
LAND_TMP=$(mktemp -d); export LAND_TMP
B=orchestrator/PROGRAMS/P-2026-003-polymarket-5min/workspace; REG=$B/COORDINATION.md; RB=$B/COORDINATOR_RUNBOOK.md
git fetch -q origin; git merge -q --ff-only origin/mm-research 2>/dev/null
D=$(git status --short -- "$REG"); if [ -n "$D" ]; then echo "HELD: register dirty [$D]"; exit 3; fi
python3 - "$E" "$N" "$REG" <<'PY' || { echo "HELD: insertion refused"; exit 4; }
import os; import re,sys
from pathlib import Path
e=Path(sys.argv[1]).read_text().strip(); n=int(sys.argv[2]); reg=Path(sys.argv[3]); s=reg.read_text()
# R-690: a coordinator drive enters an entry ONLY as pasted output. A prose claim of a drive without a fenced block in the entry is refused.
claims=re.findall(r"(?i)(driven by (?:me|the coordinator)|re-?drove|the coordinator (?:drove|ran|re-ran)|coordinator's (?:own )?drive)", e)
if claims and "```" not in e:
    raise SystemExit(f"REFUSED (R-690): the entry claims a coordinator drive ({claims[:3]}) and carries no fenced output block")
assert f'### R-{n} ' not in s, 'already present'
last=max(int(x) for x in re.findall(r'\n### R-(\d+)', s)); assert last==n-1, f'last is R-{last}, expected R-{n-1}'
a=s.index(f'### R-{last}'); sec=s.find('\n## 6. Build-readiness'); nxt=s.find('\n### ', a+10); ins=sec if (nxt==-1 or (sec!=-1 and sec<nxt)) else nxt
s=s[:ins].rstrip('\n')+'\n\n'+e+'\n'+s[ins:].lstrip('\n'); reg.write_text(s); (Path(os.environ.get('LAND_TMP','/tmp'))/'expected_added.txt').write_text(str(e.count('\n')+1)); print(f'R-{n} inserted (register clean)')
PY
{ cat "$MSG"; printf '\nLanded-By: land_entry.sh %s\n' "$(sha256sum "$0" | cut -c1-64)"; } > "$MSG.landed"; git add -- "$REG" && git commit -q -F "$MSG.landed" -- "$REG" || { echo "commit failed"; git restore -q --staged --worktree -- "$REG"; exit 5; }
ADDED=$(git show --format= HEAD -- "$REG" | grep -c '^+[^+]'); EXP=$(cat ${LAND_TMP:-/tmp}/expected_added.txt); OTHER=$(git show --format= HEAD -- "$REG" | grep -E '^\+\| Q-' | wc -l); NPATHS=$(git show --stat --format= HEAD | grep -c '|')
echo "post-condition: paths $NPATHS added $ADDED expected $EXP foreign $OTHER"
if [ "$NPATHS" = "1" ] && [ "$OTHER" = "0" ] && [ "$ADDED" -le $((EXP+2)) ]; then echo "POST-CONDITION OK"; else echo "POST-CONDITION FAILED — reverting"; git revert --no-edit HEAD >/dev/null && echo reverted; exit 6; fi
if [ "$NORB" != "--no-runbook" ]; then
  python3 - "$N" "$RB" <<'PY' && git add -- "$RB" && git commit -q -m "runbook: next register entry R-$((N+1))

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01XjtfWYSVnj6oCrmgh9M1f5" -- "$RB" && echo "RUNBOOK COMMITTED" || echo "runbook step skipped/held"
import sys,subprocess
from pathlib import Path
n=int(sys.argv[1]); rb=Path(sys.argv[2]); d=subprocess.run(['git','status','--short','--',str(rb)],capture_output=True,text=True).stdout.strip(); assert d=='', f'runbook dirty: {d}'
u=rb.read_text(); old=f"Next register entry after R-{n-1}: **R-{n}**."; assert old in u, 'runbook next line not as expected'; rb.write_text(u.replace(old, f"Next register entry after R-{n}: **R-{n+1}**.")); print('runbook next ->', f'R-{n+1}')
PY
fi
# The ONE sanctioned rebase in the shared tree (SEAT_PROTOCOL rule 21, R-755): only the commit just made, only in a tree with no other dirty path, abort on failure; otherwise STRANDED and reported.
for i in 1 2 3; do git fetch -q origin mm-research; if [ "$(git log --oneline HEAD..origin/mm-research | wc -l)" -gt 0 ]; then if [ "$(git status --short | grep -v '^??' | wc -l)" = "0" ]; then rb=$(git rebase -q origin/mm-research 2>&1) || { git rebase --abort 2>/dev/null; echo "STRANDED (report it): rebase failed and was aborted: ${rb}"; exit 7; }; else echo "STRANDED (report it): origin moved during the landing and another path is dirty -- no rebase over another seat's files (rule 21)"; git status --short | grep -v '^??' | head -3; exit 7; fi; fi; git push -q origin mm-research 2>/dev/null && { echo "PUSHED $(git rev-parse --short HEAD)"; exit 0; }; echo "push refused ($i)"; sleep 5; done; echo "STRANDED (report it): push refused three times"; exit 7
