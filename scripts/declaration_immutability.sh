#!/usr/bin/env bash
# A landed declaration version is immutable (R-711). For every <family>_v<N>.json under DIR,
# no commit AFTER the baseline may modify a file that already existed (its creation commit is
# the one allowed touch). The exit-map chain lost two seats' blocks to in-place edits on
# 2026-09-06 (a81484c -> ba635de -> 40c8903, all 'producer_exit_maps_v2.json').
# Usage: declaration_immutability.sh <dir> [--base <commit>] [--falsify]   (exit 0 clean / 1 unrepaired fork or untracked / 2 only superseded forks)
#   default base: a3de2ef (the R-711 repair) -- edits before it are HISTORY, printed, not refusals.
#   --falsify: base 56d3894 (v1's landing): producer_exit_maps_v2.json must FLAG (3 edits after
#              creation), producer_exit_maps_v1.json must PASS. Exit 1 if either does not.
set -u
DIR="${1:?dir}"; shift; BASE=a3de2ef; MODE=""
while [ $# -gt 0 ]; do case "$1" in --base) BASE="$2"; shift 2;; --falsify) MODE=falsify; BASE=56d3894; shift;; *) echo "unknown arg $1"; exit 2;; esac; done
cd "$(git -C "$DIR" rev-parse --show-toplevel)" || exit 2
# REV 90 §B7 (R-761/R-762): a fork is reported with what decides whether it MATTERS --
#   FORKED_BY_EDIT                 : edited after its creating commit and NO later version names the edited bytes (unrepaired)  -> rc 1
#   FORKED_BY_EDIT_AND_SUPERSEDED  : edited, and a later version's `supersedes.sha256` names the EDITED bytes (the chain resolves
#                                    through the fork; a forward repair, R-760/DA 121)                                          -> rc 2
#   beside each fork: created_by / edited_by / repaired_by (commit ids, so a repair does not count as a breach),
#   pre_edit_digest (the creating commit's bytes) and pre_edit_digest_pinned_by (every declaration or ledger JSON <= 5 MB that
#   names the pre-edit digest, full or 16-hex -- the load-bearing question REV could not answer by grep).
LEDGER=${LEDGER_DIR:-/home/yuqing/ctaNew/data/pm_5min/derived}
fork_detail() { # $1 file -> prints " created_by=.. edited_by=.. repaired_by=.. superseded_by=.. pre_edit_digest=.. pre_edit_digest_pinned_by=[..]" and sets SUP=1 if superseded
  local f="$1" created cur pre pre16 sup edits edited repaired pinned
  created=$(git log --diff-filter=A --format=%h -- "$f" | tail -1)
  edits=$(git log --format=%h "$BASE..HEAD" -- "$f" | grep -v "^$created" | tac | tr '\n' ' ')   # oldest first
  edited=${edits%% *}; repaired=${edits#* }; [ "$repaired" = "$edits" ] && repaired=""
  cur=$(sha256sum "$f" | cut -c1-64); pre=$(git show "$created:$f" 2>/dev/null | sha256sum | cut -c1-64); pre16=${pre:0:16}
  sup=$(python3 - "$f" "$cur" <<'PY2'
import json,sys,glob,os
f,cur=sys.argv[1],sys.argv[2]; d=os.path.dirname(f); fam=os.path.basename(f).rsplit("_v",1)[0]
out=[]
for g in sorted(glob.glob(os.path.join(d,fam+"_v*.json"))):
    if g==f: continue
    try: j=json.load(open(g))
    except Exception: continue
    s=j.get("supersedes") or {}
    if str(s.get("sha256",""))==cur or str(s.get("sha256",""))[:16]==cur[:16]: out.append(os.path.basename(g))
print(",".join(out))
PY2
)
  # size-filter BEFORE grepping: the ledger holds gigabyte tapes as .json, and grepping them first cost the falsifier its 15-min cap (R-764)
  # REV 92 §A6 (R-766/R-767): each hit carries the JSON PATH of the field holding the digest, so a MENTION (an incident
  # record) and a LINK (a `supersedes.sha256`) read differently -- the distinction this field exists to make.
  hits=$(mktemp); if [ "${IMMUT_SKIP_PIN_CENSUS:-0}" = "1" ]; then echo "SKIPPED_IN_THIS_RUN" > "$hits"; else find "$DIR" "$LEDGER" -maxdepth 1 -type f -name '*.json' -size -5M ! -path "$f" -print0 2>/dev/null | xargs -0 -r grep -lF -e "$pre" -e "$pre16" 2>/dev/null | sort > "$hits"; fi
  # (`python3 -` reads its PROGRAM from stdin, so the hit list travels by file, not by pipe -- the first version lost it, R-767)
  pinned=$(python3 - "$pre" "$pre16" "$DIR" "$LEDGER" "$hits" <<'PY3'
import sys,json
pre,pre16,d1,d2,hits=sys.argv[1:6]; out=[]
def walk(o,path):
    if isinstance(o,dict):
        for k,v in o.items(): walk(v, f"{path}.{k}" if path else k)
    elif isinstance(o,list):
        for i,v in enumerate(o): walk(v, f"{path}[{i}]")
    elif isinstance(o,str) and (pre in o or pre16 in o): out.append(path)
for line in open(hits):
    f=line.strip()
    if not f: continue
    if f=="SKIPPED_IN_THIS_RUN": print("SKIPPED_IN_THIS_RUN"); continue
    short=f.replace(d2+"/","").replace(d1+"/","")
    try: j=json.load(open(f))
    except Exception: out_paths=["<unparseable>"]
    else:
        out=[]; walk(j,""); out_paths=out or ["<in-text-not-in-a-field>"]
    for pth in out_paths: print(f"{short}:{pth}")
PY3
)
  rm -f "$hits"; pinned=$(echo "$pinned" | tr '\n' ',' | sed 's/,$//')
  SUP=0; [ -n "$sup" ] && SUP=1
  printf ' created_by=%s edited_by=%s repaired_by=[%s] superseded_by=[%s] pre_edit_digest=%s pre_edit_digest_pinned_by=[%s]' "$created" "$edited" "${repaired% }" "$sup" "$pre16" "$pinned"
}
check() { # prints OK|FORKED_BY_EDIT|FORKED_BY_EDIT_AND_SUPERSEDED|UNTRACKED ; returns 1 unrepaired fork / untracked, 2 superseded fork
  local f="$1"; local created n detail
  created=$(git log --diff-filter=A --format=%H -- "$f" | tail -1)
  [ -n "$created" ] || { echo "UNTRACKED $f"; return 1; }
  n=$(git log --format=%H "$BASE..HEAD" -- "$f" | grep -vc "^$created\$")
  if [ "$n" -eq 0 ]; then echo "OK $f edits_after_base=0"; return 0; fi
  detail=$(fork_detail "$f")
  case "$detail" in *"superseded_by=[]"*) echo "FORKED_BY_EDIT $f edits_after_base=$n$detail"; return 1;; *) echo "FORKED_BY_EDIT_AND_SUPERSEDED $f edits_after_base=$n$detail"; return 2;; esac
}
if [ "$MODE" = "falsify" ]; then
  r1=$(check live/pm_research/declarations/producer_exit_maps_v2.json); r2=$(check live/pm_research/declarations/producer_exit_maps_v1.json); echo "$r1"; echo "$r2"
  case "$r1" in FORKED_BY_EDIT*) ;; *) echo "FALSIFIER FAIL: positive control did not flag"; exit 1;; esac
  case "$r2" in OK*) ;; *) echo "FALSIFIER FAIL: known-good did not pass"; exit 1;; esac
  # REV 89 §5.2: the denominator line is checked on BOTH real directories -- one with pre-base history and one with none --
  # so the control fires on the empty-array defect whichever directory the caller named.
  for d in live/pm_research/declarations live/mm_research/declarations "$DIR"; do
    # the denominator sub-runs need only the HISTORY line; under this old base ~20 files read as forks and each would grep the ledger (R-767: two falsifier runs killed at their caps)
    hist=$(IMMUT_SKIP_PIN_CENSUS=1 "$0" "$d" --base "$BASE" 2>&1 | grep -c "^HISTORY (not judged):"); [ "$hist" -eq 1 ] || { echo "FALSIFIER FAIL: no HISTORY denominator line for $d"; exit 1; }
  done
  # REV 90 §B7 (R-762): the REAL superseded fork is the positive control for the new status -- producer_exit_maps_v7 was edited
  # in place (4e91739) and v8 supersedes the edited bytes by a verifying pair (d61307a); its pre-edit digest is 6084d6e2602d6e6a.
  r3=$(BASE=a3de2ef; check live/pm_research/declarations/producer_exit_maps_v7.json); echo "$r3" | cut -c1-200
  case "$r3" in FORKED_BY_EDIT_AND_SUPERSEDED*superseded_by=\[*producer_exit_maps_v8.json*) ;; *) echo "FALSIFIER FAIL: the real superseded fork (v7 -> v8) did not read FORKED_BY_EDIT_AND_SUPERSEDED naming v8"; exit 1;; esac
  case "$r3" in *"pre_edit_digest=6084d6e2602d6e6a"*) ;; *) echo "FALSIFIER FAIL: pre_edit_digest is not the creating commit's bytes"; exit 1;; esac
  echo "FALSIFIER PASS (base $BASE; denominator line present; superseded-fork status reads on the real v7)"; exit 0
fi
RC=0; HF=0; declare -A HFAM; NF=0; NSUP=0
for f in "$DIR"/*_v[0-9]*.json; do [ -e "$f" ] || continue; NF=$((NF+1)); check "$f"; r=$?; if [ "$r" -eq 1 ]; then RC=1; elif [ "$r" -eq 2 ]; then NSUP=$((NSUP+1)); [ "$RC" -eq 0 ] && RC=2; fi
  created=$(git log --diff-filter=A --format=%H -- "$f" | tail -1)
  if [ -n "$created" ]; then h=$(git log --format=%H "$created..$BASE" -- "$f" 2>/dev/null | wc -l); if [ "$h" -gt 0 ]; then HF=$((HF+1)); fam=$(basename "$f" | sed -E 's/_v[0-9]+\.json$//'); HFAM[$fam]=1; fi; fi
done
# REV 81 §1.3: the denominator -- what this run did NOT judge, named as history, so exit 0 reads as
# "nothing edited since $BASE" and never as "the declarations are immutable".
NFAM=$(set +u; echo "${#HFAM[@]}")  # an EMPTY associative array is "unbound" under set -u (bash 5.1); counted with -u off so the denominator line always prints
echo "HISTORY (not judged): $HF of $NF version files in $NFAM families had in-place edits BEFORE base $BASE"
echo "SUPERSEDED FORKS: $NSUP (rc 2 = every fork is superseded by a verifying pair; rc 1 = an UNREPAIRED fork exists)"
echo "base $BASE; exit $RC"; exit $RC
