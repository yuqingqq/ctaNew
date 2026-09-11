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
# A COMMIT CANNOT CONTAIN ITS OWN HASH, and landing launcher fixes moves
# this tree's HEAD off the literal pin -- which is what refused the first
# 09-09/09-10 arming. Same rule the driver already uses: a DESCENDANT is
# admissible ONLY when every computing module is byte-identical to the pin.
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
# A FROZEN ORACLE PER RUN. The settlement ledger is live and grew 55
# records between 09-07's two arms, which refused the combine. The runner
# takes no oracle path (RESOLUTIONS_REL is a constant in the PINNED
# runner), but it resolves the DATA ROOT through be_data_root's accepted
# branch 1_env_PM_DATA_ROOT -- so a snapshot root freezes the oracle with
# NO pinned edit. Every other path is a symlink, and book/receipt/mask
# reach the run as absolute CLI args, so only the ledger is frozen.
SNAP=/home/yuqing/ctaNew-oracle-${compact}
rm -rf "$SNAP"; mkdir -p "$SNAP/data/pm_5min"
for e in /home/yuqing/ctaNew/*; do n=$(basename "$e"); [ "$n" = data ] || ln -s "$e" "$SNAP/$n"; done
for e in /home/yuqing/ctaNew/data/*; do n=$(basename "$e"); [ "$n" = pm_5min ] || ln -s "$e" "$SNAP/data/$n"; done
# REV 162: the first snapshot froze ONE of FIVE growing inputs -- the one
# digested in the cell -- and left the rest as symlinks to live files.
# Every append-only input is COPIED; only static files are symlinked.
GROWING="resolutions.jsonl collector_gaps.jsonl markets.jsonl \
rewards_registry.jsonl collector_runs.jsonl"
for e in /home/yuqing/ctaNew/data/pm_5min/*; do n=$(basename "$e")
  case " $GROWING " in *" $n "*) continue ;; esac
  ln -s "$e" "$SNAP/data/pm_5min/$n"; done
for n in $GROWING; do
  src=/home/yuqing/ctaNew/data/pm_5min/$n
  [ -f "$src" ] && cp "$src" "$SNAP/data/pm_5min/$n"
done
# and mm_hf's own ledger, which also grows
rm -f "$SNAP/data/mm_hf"; mkdir -p "$SNAP/data/mm_hf"
for e in /home/yuqing/ctaNew/data/mm_hf/*; do n=$(basename "$e")
  if [ "$n" = collector_runs.jsonl ]; then cp "$e" "$SNAP/data/mm_hf/$n"
  else ln -s "$e" "$SNAP/data/mm_hf/$n"; fi; done
# REFUSE if any growing input is still a symlink to the live tree.
for n in $GROWING; do
  f="$SNAP/data/pm_5min/$n"
  [ -e "$f" ] || continue
  if [ -L "$f" ]; then
    echo "REFUSED SNAPSHOT_ROOT_HAS_LIVE_INPUT:$n -- a growing file left as"\
         "a symlink is not frozen, and the run would read it as it grows"
    exit 5
  fi
done
[ -L "$SNAP/data/mm_hf/collector_runs.jsonl" ] && {
  echo "REFUSED SNAPSHOT_ROOT_HAS_LIVE_INPUT:mm_hf/collector_runs.jsonl"; exit 5; }
FROZEN_SHAS=""
for n in $GROWING; do
  f="$SNAP/data/pm_5min/$n"; [ -f "$f" ] || continue
  FROZEN_SHAS="$FROZEN_SHAS $n:$(sha256sum "$f" | cut -c1-16)"
done
SNAP_SHA=$(sha256sum "$SNAP/data/pm_5min/resolutions.jsonl" | cut -d" " -f1)
SNAP_N=$(wc -l < "$SNAP/data/pm_5min/resolutions.jsonl")
echo "$(date -u +%H:%M:%SZ) oracle frozen: $SNAP oracle ${SNAP_SHA:0:16} records $SNAP_N"
echo "$(date -u +%H:%M:%SZ) frozen inputs:$FROZEN_SHAS"
export PM_DATA_ROOT="$SNAP"
export BE_WORKTREE="$TREE" DE_VALUATION_EXPECTED_TREE="$TREE"
cd "$TREE/live/pm_research" || exit 2
# REV 157: the launcher's own provenance, beside the day's receipt, so a
# receipt can be traced to the bytes that launched it after the scratchpad
# is gone.
self=/home/yuqing/ctaNew-wt-deval/live/pm_research/launchers/chain_day.sh
/home/yuqing/pricer-sol/venv/bin/python3 - "$day" "$self" "$H" <<'PYL'
[ -f "/home/yuqing/ctaNew/data/pm_5min/derived/fwd_v2/p003_de_chain_launch_${compact}.json" ] || {
  echo "REFUSED CHAIN_LAUNCH_RECORD_NOT_WRITTEN: no provenance record, no offer"; exit 6; }

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
       "oracle_snapshot_root": __import__("os").environ.get("PM_DATA_ROOT"),
       "pm_data_root_read_back_from_unit_environ": (
           dict(kv.split("=", 1) for kv in
                Path(f"/proc/{__import__('os').getppid()}/environ")
                .read_bytes().decode().split("\x00") if "=" in kv)
           .get("PM_DATA_ROOT")),
       "frozen_inputs": {f.name: __import__("hashlib").sha256(
           f.read_bytes()).hexdigest() for f in sorted(
           Path(__import__("os").environ["PM_DATA_ROOT"],
                "data/pm_5min").iterdir()) if f.is_file()
           and not f.is_symlink()},
       "symlinked_inputs": {f.name: str(f.resolve()) for f in sorted(
           Path(__import__("os").environ["PM_DATA_ROOT"],
                "data/pm_5min").iterdir()) if f.is_symlink()},
       "oracle_sha256": __import__("hashlib").sha256(
           Path(__import__("os").environ["PM_DATA_ROOT"],
                "data/pm_5min/resolutions.jsonl").read_bytes()).hexdigest(),
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
