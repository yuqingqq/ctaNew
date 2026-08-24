"""Make the OP plane REVEAL what is in force, instead of declaring it.

R-42's generalisation, applied to OPS: *the check does not ask the rule what it
is; it makes the rule reveal it.* `OP_PLANE_PLAN` §0a **declares** which sections
are IN FORCE and which are PROSE-ONLY, and a declaration is a rule asserting its
own correctness — the defect R-42 named.

So every in-force claim here names a COMMAND and its EXPECTED value, and this
runs them. Anyone can re-derive the plane's true state in one call without
trusting an OPS report, a revision string, or git.

**It must not fail open.** The coordinator's landing-evidence checks produced two
FALSE ABSENT results (a grep matching a removal comment; a shell variable lost
across a `cd`), reporting present things as absent. A check that fails open is
the mirror of a gate that cannot fire. Here an error is a FAIL, never a skip.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys

REPO = "/home/yuqing/ctaNew"
PY = "/home/yuqing/pricer-sol/venv/bin/python3"


def run(cmd: str) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, shell=True, cwd=REPO, capture_output=True, text=True, timeout=180
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as exc:                      # an error is a FAIL, not a skip
        return 255, f"CHECK ITSELF FAILED: {exc!r}"


SHOW = "systemctl --user show {unit} -p {prop} --value"

CLAIMS: list[tuple[str, str, str]] = [
    ("caps: measurement MemoryMax=16G",
     SHOW.format(unit="pm-measurement-pipeline.service", prop="MemoryMax"), "17179869184"),
    ("caps: evaluation MemoryMax=16G",
     SHOW.format(unit="pm-evaluation-pipeline.service", prop="MemoryMax"), "17179869184"),
    ("R-32: measurement MemoryHigh disabled",
     SHOW.format(unit="pm-measurement-pipeline.service", prop="MemoryHigh"), "infinity"),
    ("R-32: evaluation MemoryHigh disabled",
     SHOW.format(unit="pm-evaluation-pipeline.service", prop="MemoryHigh"), "infinity"),
    ("R-22: measurement is the preferred OOM victim",
     SHOW.format(unit="pm-measurement-pipeline.service", prop="OOMScoreAdjust"), "1000"),
    ("R-22: evaluation is the preferred OOM victim",
     SHOW.format(unit="pm-evaluation-pipeline.service", prop="OOMScoreAdjust"), "1000"),
    ("R-22(3): prices collector NOT re-prioritised",
     SHOW.format(unit="pm-collector-prices.service", prop="OOMScoreAdjust"), "200"),
    ("R-22(3): clob collector NOT re-prioritised",
     SHOW.format(unit="pm-collector-clob.service", prop="OOMScoreAdjust"), "200"),
    ("R-35/Q-OPS-4: eval chained to measurement success",
     SHOW.format(unit="pm-measurement-pipeline.service", prop="OnSuccess"),
     "pm-evaluation-pipeline.service"),
    ("R-40: the checker's own failure is hooked",
     SHOW.format(unit="pm-lane-health.service", prop="OnFailure"),
     "pm-alert@pm-lane-health.service.service"),
    ("R-36(2): every check demonstrated failing",
     f"{PY} live/pm_research/ops/pm_lane_health.py --selftest | tail -1",
     "18/18 checks demonstrated"),
    ("R-40: the checker watches itself",
     f"{PY} -c \"import sys;sys.path.insert(0,'live/pm_research/ops');"
     "import pm_lane_health as h;print('pm-lane-health.service' in h.UNITS)\"", "True"),
    ("R-35/Q-OPS-6: derivation_lag is printed, not inferred",
     f"{PY} live/pm_research/ops/pm_lane_health.py --json --no-notify 2>/dev/null | "
     f"{PY} -c \"import sys,json;d=json.load(sys.stdin);"
     "c=[x for x in d['checks'] if x['name']=='LANE_PROGRESS'][0];"
     "l=c['lanes'][0];print(all(k in l for k in "
     "('derivation_lag_days','outstanding_days','lag_floor_days','state')))\"", "True"),
    ("never-attempted audit: disk headroom is monitored",
     f"{PY} live/pm_research/ops/pm_lane_health.py --json --no-notify 2>/dev/null | "
     f"{PY} -c \"import sys,json;d=json.load(sys.stdin);"
     "print(any(c['name']=='DISK_HEADROOM' for c in d['checks']))\"", "True"),
    ("never-attempted audit: clock sync is monitored",
     f"{PY} live/pm_research/ops/pm_lane_health.py --json --no-notify 2>/dev/null | "
     f"{PY} -c \"import sys,json;d=json.load(sys.stdin);"
     "print(any(c['name']=='CLOCK_SYNC' for c in d['checks']))\"", "True"),
    ("never-attempted audit: capacity pre-flight exists and predicts",
     f"{PY} live/pm_research/ops/capacity_preflight.py --selftest | "
     "grep 'verdict-logic cases'", "5/5 verdict-logic cases"),
    # R-68 discipline applied to THIS script: every unit claim below queries
    # `systemctl show`, i.e. the INSTALLED unit -- while the REPO file is the
    # artifact under review. A pass would confirm the wrong object if the two
    # diverged, which is the same ambiguity as 0/0/0 on a diff. Disambiguated
    # here rather than by a human remembering to check.
    ("units: repo source == installed unit (else every unit claim is ambiguous)",
     "for u in pm-collector-prices.service pm-collector-clob.service "
     "pm-measurement-pipeline.service pm-evaluation-pipeline.service "
     "pm-lane-health.service pm-lane-health.timer pm-evaluation-pipeline.timer "
     "pm-measurement-pipeline.timer 'pm-alert@.service'; do "
     "diff -q \"live/pm_research/ops/$u\" \"$HOME/.config/systemd/user/$u\" >/dev/null 2>&1 "
     "|| echo DIVERGED; done | wc -l", "0"),
    ("R-89: the vacated R-7 licence is surfaced wherever the amendment acts",
     f"{PY} live/pm_research/ops/pm_lane_health.py --json --no-notify 2>/dev/null | "
     f"{PY} -c \"import sys,json;d=json.load(sys.stdin);"
     "c=[x for x in d['checks'] if x['name']=='R7_PROVISIONAL'][0];"
     "print(c['level']=='WARN' and len(c['receipts_citing_vacated_licence'])>0)\"", "True"),
    ("R-28 enforceable: frozen artifacts sealed and intact",
     f"{PY} live/pm_research/ops/frozen_manifest.py --verify | tail -1", 
     "12/12 frozen artifacts intact under R-28"),
    ("R-35: contracts.yaml carries no unratified OPS edit",
     "git status --porcelain live/pm_research/contracts/contracts.yaml | wc -l", "0"),
    ("plan revision, read from the file itself",
     "sed -n '3p' live/pm_research/plans/OP_PLANE_PLAN.md | grep -o 'REVISION [0-9]*'",
     "REVISION 3"),
]


def main() -> int:
    rows = []
    for name, cmd, want in CLAIMS:
        code, out = run(cmd)
        got = out.splitlines()[-1].strip() if out else ""
        ok = (code == 0) and got == want
        rows.append((ok, name, got, want))
    width = max(len(r[1]) for r in rows)
    for ok, name, got, want in rows:
        flag = "PASS" if ok else "FAIL"
        detail = "" if ok else f"   got={got!r} want={want!r}"
        print(f"  {flag}  {name:<{width}}{detail}")
    passed = sum(1 for r in rows if r[0])
    print(f"  {passed}/{len(rows)} landing-evidence claims verified")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
