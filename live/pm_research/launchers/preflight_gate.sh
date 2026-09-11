#!/bin/bash
# The day's gates, run WITHOUT the lock, before an offer.
# A SCRIPT UNDER A RUNNING UNIT IS FROZEN BYTES: bash reads by byte offset,
# so editing a running script makes the shell resume at its saved offset in
# a DIFFERENT file. Two chains were mid-flight with offsets landing inside
# a word. This lives in its own file so only FUTURE launches pick it up.
set -u
cd /home/yuqing/ctaNew-wt-deval || exit 2   # DECL is tree-relative
day="${1:?usage: preflight_gate.sh <YYYY-MM-DD> <certificate>}"
cert="${2:?}"
PYBIN=/home/yuqing/pricer-sol/venv/bin/python3
GATE=/home/yuqing/ctaNew-wt-deval/live/pm_research/de_stage0_freeze_gate.py
GOUT=/tmp/stage0_freeze_${day//-/}.json
# THE FROZEN MODULES, CHECKED FROM OUTSIDE EVERY MODULE (DA 255 / REV 221).
# de_forward_value_day.py carries the valuation's own freeze check, so it is
# the one module that vouches for itself: edit the check to lie and nothing
# inside the closure notices. DA's population-freeze verifier is external to
# every module it checks, and the declaration classes de_forward_value_day.py
# PIPELINE, so a byte moving there REFUSES here, by filename, before a lock
# is taken. INSTRUMENT drift reports and passes -- the thing MEASURING
# moving is not the thing MEASURED moving.
# A SCRATCH ROOT IS NEVER SILENT. The override exists so the falsifier can
# drive THIS path against a mirror; it announces itself, and the gate
# records the roots it measured in its own report.
GROOT=()
if [ -n "${DE_STAGE0_ROOT_OVERRIDE:-}" ]; then
  echo "$(date -u +%H:%M:%SZ) STAGE 0 ON A SCRATCH ROOT: $DE_STAGE0_ROOT_OVERRIDE"
  GROOT=(--root "$DE_STAGE0_ROOT_OVERRIDE")
fi
"$PYBIN" "$GATE" "${GROOT[@]}" > "$GOUT" 2>&1
grc=$?
if [ "$grc" != 0 ]; then
  echo "$(date -u +%H:%M:%SZ) STAGE 0 FREEZE GATE REFUSED (rc=$grc)"
  grep -o "REFUSED [A-Z0-9_]*:[^\"]*" "$GOUT" | head -4
  exit 3
fi
# REVIEW 187: THE GATE'S VERDICT GOES INTO THE RUN'S EVIDENCE, BEFORE THE
# OFFER. It used to exist only in /tmp and in a log line, so a record
# carried no proof the gate ran -- indistinguishable from its not having
# run. Written into the launch record, which is under derived/ and
# survives the scratch dir.
"$PYBIN" - "$GOUT" "$day" <<'PYMERGE'
import json, sys, time
from pathlib import Path
gout, day = Path(sys.argv[1]), sys.argv[2]
rec = Path("/home/yuqing/ctaNew/data/pm_5min/derived") / (
    f"p003_de_chain_launch_{day.replace('-', '')}.json")
doc = {}
if rec.is_file():
    try:
        doc = json.loads(rec.read_text())
    except Exception:
        doc = {}
try:
    verdict = json.loads(gout.read_text())
except Exception as exc:
    verdict = {"status": f"STAGE0_REPORT_UNREADABLE: {exc}"}
doc["stage0_verdict"] = verdict
doc["at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
rec.parent.mkdir(parents=True, exist_ok=True)
rec.write_text(json.dumps(doc, indent=1, default=str))
print(f"stage0 verdict recorded in {rec.name}: {verdict.get('status')}")
PYMERGE

/home/yuqing/pricer-sol/venv/bin/python3 \
  /home/yuqing/ctaNew-wt-deval/live/pm_research/de_preflight_matrix.py \
  --tree /home/yuqing/ctaNew-wt-deval \
  --derived /home/yuqing/ctaNew/data/pm_5min/derived \
  --declarations /home/yuqing/ctaNew-wt-deval/live/pm_research/declarations \
  --certification "$cert" --days "$day" --gate
