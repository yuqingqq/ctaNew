#!/bin/bash
# THE CHAIN FOR 09-09 ONWARD. Stage 0 is the preflight matrix: the day's
# gates run WITHOUT the lock and refuse by name before an offer, so a stale
# record costs seconds, never a 77-minute slot.
# usage: chain_day.sh <YYYY-MM-DD> <days-scored-csv>
set -u
DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
day="${1:?usage: chain_day.sh [--dry-run] <YYYY-MM-DD> <days-scored-csv>}"
scored="${2:?}"
compact="${day//-/}"
D=/home/yuqing/ctaNew/data/pm_5min/derived
TREE=/home/yuqing/ctaNew-wt-deval
PIN=f3096021711904f10c4ecb06319326b4d9fa28f8
CERT=$D/be_score_neutrality_20260903__EV22_vs_NEUTCHK__68e7d23.json
# A COMMIT CANNOT CONTAIN ITS OWN HASH, and landing launcher fixes moves
# this tree's HEAD off the literal pin -- which is what refused the first
# 09-09/09-10 arming. Same rule the driver already uses: a DESCENDANT is
# admissible ONLY when every computing module is byte-identical to the pin.
# LAUNCHER_BYTES_NOT_COMMITTED: the launcher digests ITSELF and refuses
# unless those exact bytes exist as a blob reachable from an origin ref.
# A launcher whose bytes are in no commit cannot be reproduced from the
# tree, and the receipts it produces cannot be traced.
SELF_PATH="${BASH_SOURCE[0]}"
SELF_SHA=$(sha256sum "$SELF_PATH" | cut -d" " -f1)
FOUND=""
for ref in $(git -C "$TREE" for-each-ref --format="%(refname)" refs/remotes/origin 2>/dev/null); do
  b=$(git -C "$TREE" rev-parse "$ref:live/pm_research/launchers/chain_day.sh" 2>/dev/null) || continue
  s=$(git -C "$TREE" cat-file blob "$b" 2>/dev/null | sha256sum | cut -d" " -f1)
  [ "$s" = "$SELF_SHA" ] && { FOUND="$ref"; break; }
done
if [ -z "$FOUND" ]; then
  echo "REFUSED LAUNCHER_BYTES_NOT_COMMITTED: ${SELF_SHA:0:16} is in no blob"\
       "on any origin ref; this launcher cannot be reproduced from the tree"
  exit 10
fi
echo "$(date -u +%H:%M:%SZ) launcher ${SELF_SHA:0:16} found on $FOUND"
H=$(git -C "$TREE" rev-parse HEAD)
if [ "$H" != "$PIN" ]; then
  git -C "$TREE" merge-base --is-ancestor "$PIN" "$H" || {
    echo "REFUSED TREE_IS_NOT_THE_VALUATION_PIN: $H is not a descendant of $PIN"; exit 2; }
  for m in de_settlement_control_run.py de_forward_evaluator.py            de_settlement_control_aggregate.py de_asymmetry_null_run.py            de_matched_cancel_control.py de_multiday_gate1_runner.py            be_score_neutrality.py; do
    a=$(sha256sum "$TREE/live/pm_research/$m" | cut -d" " -f1)
    b=$(git -C "$TREE" show "$PIN:live/pm_research/$m" | sha256sum | cut -d" " -f1)
    [ "$a" = "$b" ] || {
      echo "REFUSED VALUATION_MODULE_BYTES_DIFFER_FROM_THE_PIN: $m"; exit 2; }
  done
  echo "$(date -u +%H:%M:%SZ) tree $H is a descendant of $PIN; 7/7 computing modules identical"
fi
ASOF=/home/yuqing/ctaNew/data/pm_5min/derived/fwd_v2/p003_de_asof_raw_${compact}.json
/home/yuqing/pricer-sol/venv/bin/python3 \
  /home/yuqing/ctaNew-wt-deval/live/pm_research/de_asof_listing.py \
  listing /home/yuqing/ctaNew "$day" > "$ASOF" || {
    echo "REFUSED ASOF_LISTING_NOT_TAKEN"; exit 9; }
echo "$(date -u +%H:%M:%SZ) raw/ as-of: $(/home/yuqing/pricer-sol/venv/bin/python3 -c "
import json,sys; d=json.load(open(sys.argv[1])); print(d['n_files'],'files',d['total_bytes'],'bytes')" "$ASOF")"
# RETIRED on the valuation path (DE 303): the oracle is read ONCE by the
# run itself, so no snapshot root is needed and require_ledger sees the
# canonical ledger. The as-of digests below stay -- they are PROVENANCE for
# the other growing inputs, not the cohort.


export BE_WORKTREE="$TREE" DE_VALUATION_EXPECTED_TREE="$TREE"
cd "$TREE/live/pm_research" || exit 2
# REV 157: the launcher's own provenance, beside the day's receipt, so a
# receipt can be traced to the bytes that launched it after the scratchpad
# is gone.
self=/home/yuqing/ctaNew-wt-deval/live/pm_research/launchers/chain_day.sh
/home/yuqing/pricer-sol/venv/bin/python3 - "$day" "$self" "$H" <<'PYL'
import hashlib, json, os, sys, datetime
from pathlib import Path




    pid = subprocess.run(["systemctl", "--user", "show", unit,
                          "-p", "MainPID", "--value"],
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().decode()
        return dict(kv.split("=", 1) for kv in raw.split("\x00")
                    if "=" in kv).get("PM_DATA_ROOT", "ABSENT_IN_UNIT")
    except OSError:
        return "NOT_LAUNCHED_YET"
day, self_path, head = sys.argv[1], sys.argv[2], sys.argv[3]
f = Path(self_path)
rec = {"protocol": "P003_DE_CHAIN_LAUNCH_PROVENANCE_V1", "day": day,
       "launcher_path": str(f),
       "launcher_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
       "launcher_mtime_utc": datetime.datetime.utcfromtimestamp(
           f.stat().st_mtime).isoformat() + "Z",
       "tree_head": head,
       "data_root": "/home/yuqing/ctaNew/data",
       "oracle": "READ ONCE BY THE RUN (DE 303); no snapshot root",
       "asof_raw_listing": str(Path(
           "/home/yuqing/ctaNew/data/pm_5min/derived/fwd_v2",
           f"p003_de_asof_raw_{day.replace('-', '')}.json")),
       "growing_input_digests": _growing_digests(),
       "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
out = Path("/home/yuqing/ctaNew/data/pm_5min/derived/fwd_v2")
out.mkdir(parents=True, exist_ok=True)
(out / f"p003_de_chain_launch_{day.replace('-', '')}.json").write_text(
    json.dumps(rec, indent=1))
print("  launch provenance:", rec["launcher_sha256"][:16], head[:12])
PYL


REC=/home/yuqing/ctaNew/data/pm_5min/derived/fwd_v2/p003_de_chain_launch_${compact}.json
[ -f "$REC" ] || { echo "REFUSED CHAIN_LAUNCH_RECORD_NOT_WRITTEN: no provenance, no offer"; exit 6; }

# STAGE 0 distinguishes NOT-YET-BUILT from WRONG.
#   exit 3 = WOULD_REFUSE -> a stale or wrong record: stop, named.
#   exit 4 = INPUT_ABSENT -> the book is not built yet: wait, no offer.
# Collapsing them either burns a lock slot on a book that does not exist,
# or halts a population that was only early.
while :; do
  bash /home/yuqing/ctaNew-wt-deval/live/pm_research/launchers/preflight_gate.sh       "$day" "$CERT" >/tmp/pf_$compact.log 2>&1
  rc=$?
  case $rc in
    0) echo "$(date -u +%H:%M:%SZ) stage 0 clean for $day; offering"; break ;;
    4) echo "$(date -u +%H:%M:%SZ) $day not built yet; re-checking in 120s"
       sleep 120 ;;
    *) echo "$(date -u +%H:%M:%SZ) STAGE 0 REFUSED for $day -- no offer made"
       grep -m3 "REFUSED" /tmp/pf_$compact.log; exit 3 ;;
  esac
done
n=0
while :; do
  n=$((n+1)); u="deRV${compact}w${n}"
  systemctl --user reset-failed "$u" 2>/dev/null
  if [ "$DRY" = "1" ]; then
    echo "WOULD LAUNCH, 0 units: $u de_forward_value_day.py --day $day"
    exit 0
  fi
  bash be_heavy_run.sh "$u" de_forward_value_day.py \
    --day "$day" --book $D/be_daybook_${compact}_btc__L250ms__FWD1.pkl \
    --book-receipt $D/be_daybook_receipt_${compact}_btc__L250ms__FWD1.json \
    --score-certification "$CERT" \
    --params $TREE/live/pm_research/declarations/de_multiday_gate1_params_v31.json \
    --out-dir $D/fwd_v2 --n-draws 500 --seed 0 \
    --days-scored "$scored" --n-declared 7 --derived $D >/dev/null 2>&1
  while :; do
    sub=$(systemctl --user show "$u" -p SubState --value 2>/dev/null)
    case "$sub" in exited|dead|failed|"") break ;; esac
    sleep 10
  done
  rc=$(systemctl --user show "$u" -p ExecMainStatus --value 2>/dev/null)
  if [ "$rc" != "75" ]; then
    echo "$(date -u +%H:%M:%SZ) $u ran, rc=$rc"
    # RE-VERIFY the as-of listing: a closed day's own inputs must not move.
    /home/yuqing/pricer-sol/venv/bin/python3 \
      /home/yuqing/ctaNew-wt-deval/live/pm_research/de_asof_listing.py \
      verify /home/yuqing/ctaNew "$day" "$ASOF" || {
        echo "$(date -u +%H:%M:%SZ) as-of re-verification REFUSED"; exit 8; }
    echo "$(date -u +%H:%M:%SZ) as-of re-verified: the day's raw slice did not move"
    exit "$rc"
  fi
  echo "$(date -u +%H:%M:%SZ) $u: lock held, re-offering in 20s"; sleep 20
done
