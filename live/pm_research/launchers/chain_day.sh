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
bash /home/yuqing/ctaNew-wt-deval/live/pm_research/launchers/preflight_gate.sh "$day" "$CERT" || {
  echo "$(date -u +%H:%M:%SZ) STAGE 0 REFUSED for $day -- no offer made"; exit 3; }
echo "$(date -u +%H:%M:%SZ) stage 0 clean for $day; offering"
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
