#!/bin/bash
# Day-two emit. The 09-08 chain's own emit stage CANNOT run: val_0908.sh was
# truncated to its saved offset, so that shell exits after its current
# stage. This is a separate unit, in its own file, triggered on the result.
set -u
D=/home/yuqing/ctaNew/data/pm_5min/derived
PY=/home/yuqing/pricer-sol/venv/bin/python3
SRC=/home/yuqing/ctaNew-wt-de2/live/pm_research
PRODUCER="${1:?usage: emit_0908.sh <producer-unit>}"
V2=$D/fwd_v2/p003_de_forward_value_20260908.json
for m in de_revaluation_emit.py de_window_decomposition.py; do
  a=$(sha256sum "$SRC/$m" | cut -d' ' -f1)
  b=$(git -C /home/yuqing/ctaNew-wt-de2 show "origin/mm-research:live/pm_research/$m" | sha256sum | cut -d' ' -f1)
  [ "$a" = "$b" ] || { echo "REFUSED EMIT_MODULE_IS_NOT_THE_LANDED_BYTES: $m"; exit 4; }
done
# The budget is the PRODUCER, not a clock -- see emit_wait2.sh.
echo "$(date -u +%H:%M:%SZ) emit modules verified; waiting for $V2 (producer $PRODUCER)"
while :; do
  if [ -f "$V2" ]; then
    s1=$(stat -c %s "$V2"); sleep 10; s2=$(stat -c %s "$V2")
    [ "$s1" = "$s2" ] && break
  fi
  sub=$(systemctl --user show "$PRODUCER" -p SubState --value 2>/dev/null)
  case "$sub" in
    exited|dead|failed|"")
      sleep 20
      if [ ! -f "$V2" ]; then
        echo "REFUSED EMIT_PRODUCER_ENDED_WITHOUT_RESULT: $PRODUCER is"              "$sub and $V2 does not exist. Nothing to emit."
        exit 3
      fi ;;
  esac
  sleep 15
done
"$PY" - "$V2" "$D/fwd_v2/p003_de_forward_value_20260907.json" <<'PYX'
import json,sys
from pathlib import Path
sys.path.insert(0,"/home/yuqing/ctaNew-wt-de2/live/pm_research")
import de_revaluation_emit as E
v2,v1=sys.argv[1],sys.argv[2]
D=Path("/home/yuqing/ctaNew/data/pm_5min/derived")
r8=E.read_json(v2,"the 09-08 V2 result")
per_arm={a:c["D"] for a,c in r8["cells"].items()}
allD=dict(per_arm)
p7=Path(v1)
if p7.is_file():
    r7=E.read_json(v1,"the 09-07 V2 result")
    allD={f"2026-09-07::{a}":c["D"] for a,c in r7["cells"].items()}
allD={"2026-09-07":min(allD.values()) if allD else 0.0,
      "2026-09-08":min(per_arm.values())}
out=E.emit_single_book_day("2026-09-08", per_arm, allD, 7)
dst=D/"fwd_v2"/"p003_de_revaluation_emit_20260908.json"
dst.write_text(json.dumps(out,indent=1,default=str))
print("  TRIPWIRE_STATUS:",out["TRIPWIRE_STATUS"])
print("  unconditional window rows:",len(out["per_window_table_unconditional"]))
t=out["running_tally"]
print(f"  tally: G={t['G_so_far']}/{t['G_declared']} remaining={t['days_remaining']} "
      f"tolerance={t['tolerance_negative_days_at_G_declared']} negatives={t['negative_or_zero_days']}")
print("  written:",dst)
PYX
