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
# REV 162: the first snapshot froze ONE of FIVE growing inputs. Every
# append-only input is COPIED now; only static files are symlinked.
# REVIEW 163: the UNION. collector_health.jsonl grows ~732 B / 20 s and is
# copied like the rest. raw/ is 4.4 GB and 2,016 files FOR ONE DAY, so it
# cannot be frozen by copying -- it carries an AS-OF ASSERTION instead,
# recorded at launch and re-verified at exit.
GROWING="resolutions.jsonl collector_gaps.jsonl markets.jsonl rewards_registry.jsonl collector_runs.jsonl collector_health.jsonl"
RAW_IS_ASSERTED_NOT_COPIED=1
for e in /home/yuqing/ctaNew/data/pm_5min/*; do n=$(basename "$e")
  case " $GROWING " in *" $n "*) continue ;; esac
  ln -s "$e" "$SNAP/data/pm_5min/$n"; done
for n in $GROWING; do
  [ -f /home/yuqing/ctaNew/data/pm_5min/$n ] && cp /home/yuqing/ctaNew/data/pm_5min/$n "$SNAP/data/pm_5min/$n"
done
rm -f "$SNAP/data/mm_hf"; mkdir -p "$SNAP/data/mm_hf"
for e in /home/yuqing/ctaNew/data/mm_hf/*; do n=$(basename "$e")
  if [ "$n" = collector_runs.jsonl ]; then cp "$e" "$SNAP/data/mm_hf/$n"
  else ln -s "$e" "$SNAP/data/mm_hf/$n"; fi; done
for n in $GROWING; do
  f="$SNAP/data/pm_5min/$n"; [ -e "$f" ] || continue
  if [ -L "$f" ]; then
    echo "REFUSED SNAPSHOT_ROOT_HAS_LIVE_INPUT:$n"; exit 5; fi
done
if [ -L "$SNAP/data/mm_hf/collector_runs.jsonl" ]; then
  echo "REFUSED SNAPSHOT_ROOT_HAS_LIVE_INPUT:mm_hf/collector_runs.jsonl"; exit 5; fi
FROZEN=""
for n in $GROWING; do f="$SNAP/data/pm_5min/$n"; [ -f "$f" ] || continue
  FROZEN="$FROZEN $n:$(sha256sum "$f" | cut -c1-16)"; done
SNAP_SHA=$(sha256sum "$SNAP/data/pm_5min/resolutions.jsonl" | cut -d" " -f1)
SNAP_N=$(wc -l < "$SNAP/data/pm_5min/resolutions.jsonl")
echo "$(date -u +%H:%M:%SZ) oracle frozen: $SNAP sha ${SNAP_SHA:0:16} records $SNAP_N"
echo "$(date -u +%H:%M:%SZ) frozen inputs:$FROZEN"
ASOF=/home/yuqing/ctaNew/data/pm_5min/derived/fwd_v2/p003_de_asof_raw_${compact}.json
/home/yuqing/pricer-sol/venv/bin/python3 \
  /home/yuqing/ctaNew-wt-deval/live/pm_research/de_asof_listing.py \
  listing /home/yuqing/ctaNew "$day" > "$ASOF" || {
    echo "REFUSED ASOF_LISTING_NOT_TAKEN"; exit 9; }
echo "$(date -u +%H:%M:%SZ) raw/ as-of: $(/home/yuqing/pricer-sol/venv/bin/python3 -c "
import json,sys; d=json.load(open(sys.argv[1])); print(d['n_files'],'files',d['total_bytes'],'bytes')" "$ASOF")"
# NOT PM_DATA_ROOT: be_heavy_run.sh passes --setenv=PM_DATA_ROOT to every
# unit and would overwrite it. BE_SNAPSHOT_ROOT is the wrapper's opt-in.
# RETIRED on the valuation path (DE 303): the oracle is read ONCE by the
# run itself, so no snapshot root is needed and require_ledger sees the
# canonical ledger. The as-of digests below stay -- they are PROVENANCE for
# the other growing inputs, not the cohort.
: # BE_SNAPSHOT_ROOT intentionally not exported
: # DE_EXPECT_SNAPSHOT_ROOT / PM_DATA_ROOT intentionally not exported
export BE_WORKTREE="$TREE" DE_VALUATION_EXPECTED_TREE="$TREE"
cd "$TREE/live/pm_research" || exit 2
# REV 157: the launcher's own provenance, beside the day's receipt, so a
# receipt can be traced to the bytes that launched it after the scratchpad
# is gone.
self=/home/yuqing/ctaNew-wt-deval/live/pm_research/launchers/chain_day.sh
/home/yuqing/pricer-sol/venv/bin/python3 - "$day" "$self" "$H" <<'PYL'
import hashlib, json, os, sys, datetime
from pathlib import Path


def _walk_snapshot():
    """STAT EVERY PATH under the snapshot root -- never a literal list.

    The record previously enumerated one directory and reported five
    copies where seven exist. What a file IS on disk is the property; a
    list written beside the code that makes the copies drifts from it.
    """
    root = Path(os.environ["PM_DATA_ROOT"])
    frozen, linked = {}, {}
    for f in sorted(root.rglob("*")):
        rel = str(f.relative_to(root))
        if f.is_symlink():
            linked[rel] = os.readlink(f)
        elif f.is_file():
            frozen[rel] = hashlib.sha256(f.read_bytes()).hexdigest()
    return frozen, linked


def _digests():
    return _walk_snapshot()[0]


def _symlinks():
    return _walk_snapshot()[1]


def _environ_of_unit(unit=None):
    """PM_DATA_ROOT as the VALUATION UNIT actually has it.

    The previous version read /proc/$PPID/environ -- the launcher shell,
    which a bash `export` never rewrites -- so the field was inert and read
    None. Until the unit exists there is nothing to read, and the record
    says NOT_LAUNCHED_YET rather than None: absence must not look like a
    measurement.
    """
    import subprocess
    if not unit:
        return "NOT_LAUNCHED_YET"
    pid = subprocess.run(["systemctl", "--user", "show", unit,
                          "-p", "MainPID", "--value"],
                         capture_output=True, text=True).stdout.strip()
    if not pid or pid == "0":
        return "NOT_LAUNCHED_YET"
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
       "oracle_snapshot_root": os.environ.get("PM_DATA_ROOT"),
       "pm_data_root_from_unit_environ": _environ_of_unit(
           sys.argv[4] if len(sys.argv) > 4 else None),
       "frozen_inputs": _digests(),
       "symlinked_inputs": _symlinks(),
       "oracle_snapshot_root": __import__("os").environ.get("PM_DATA_ROOT"),
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

# DRIVE THE REAL PATH before offering: a probe THROUGH the wrapper, on its
# own lock, refuses VALUATION_DID_NOT_SEE_SNAPSHOT_ROOT in seconds if the
# payload would read the live files.
BE_HEAVY_LOCK=/tmp/de_snap_${compact}.lock bash be_heavy_run.sh \
  "deSNAP${compact}" de_snapshot_probe.py >/dev/null 2>&1
while :; do sub=$(systemctl --user show "deSNAP${compact}" -p SubState --value 2>/dev/null)
  case "$sub" in exited|dead|failed|"") break ;; esac; sleep 3; done
prc=$(systemctl --user show "deSNAP${compact}" -p ExecMainStatus --value 2>/dev/null)
[ "$prc" = "0" ] || { echo "REFUSED VALUATION_DID_NOT_SEE_SNAPSHOT_ROOT (probe rc=$prc)"; exit 7; }
echo "$(date -u +%H:%M:%SZ) snapshot proven on the REAL path (probe rc=0)"

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
