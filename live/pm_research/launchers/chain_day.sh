#!/bin/bash
# THE CHAIN FOR 09-09 ONWARD. Stage 0 is the preflight matrix: the day's
# gates run WITHOUT the lock and refuse by name before an offer, so a stale
# record costs seconds, never a 77-minute slot.
# usage: chain_day.sh <YYYY-MM-DD> <days-scored-csv>
set -u
day="${1:?usage: chain_day.sh <YYYY-MM-DD> <days-scored-csv>}"
scored="${2:?}"
compact="${day//-/}"
D=/home/yuqing/ctaNew/data/pm_5min/derived
TREE=/home/yuqing/ctaNew-wt-deval
PIN=68e7d2352c7d7aed7963e9842aaa68351b38689b
CERT=$D/be_score_neutrality_20260903__EV22_vs_NEUTCHK__68e7d23.json
H=$(git -C "$TREE" rev-parse HEAD)
[ "$H" = "$PIN" ] || { echo "REFUSED TREE_IS_NOT_THE_VALUATION_PIN: $H"; exit 2; }
export BE_WORKTREE="$TREE" DE_VALUATION_EXPECTED_TREE="$TREE"
cd "$TREE/live/pm_research" || exit 2
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

# REV 157: the launcher's own provenance, beside the day's receipt, so a
# receipt can be traced to the bytes that launched it after the scratchpad
# is gone.
self=/home/yuqing/ctaNew-wt-deval/live/pm_research/launchers/chain_day.sh
/home/yuqing/pricer-sol/venv/bin/python3 - "$day" "$self" "$H" <<'PYL'
import hashlib, json, sys, datetime
from pathlib import Path
day, self_path, head = sys.argv[1], sys.argv[2], sys.argv[3]
f = Path(self_path)
rec = {"protocol": "P003_DE_CHAIN_LAUNCH_PROVENANCE_V1", "day": day,
       "launcher_path": str(f),
       "launcher_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
       "launcher_mtime_utc": datetime.datetime.utcfromtimestamp(
           f.stat().st_mtime).isoformat() + "Z",
       "tree_head": head,
       "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
out = Path("/home/yuqing/ctaNew/data/pm_5min/derived/fwd_v2")
out.mkdir(parents=True, exist_ok=True)
(out / f"p003_de_chain_launch_{day.replace('-', '')}.json").write_text(
    json.dumps(rec, indent=1))
print("  launch provenance:", rec["launcher_sha256"][:16], head[:12])
PYL
n=0
while :; do
  n=$((n+1)); u="deRV${compact}w${n}"
  systemctl --user reset-failed "$u" 2>/dev/null
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
  [ "$rc" != "75" ] && { echo "$(date -u +%H:%M:%SZ) $u ran, rc=$rc"; exit "$rc"; }
  echo "$(date -u +%H:%M:%SZ) $u: lock held, re-offering in 20s"; sleep 20
done
