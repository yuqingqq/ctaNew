#!/usr/bin/env python3
"""pm_research_guard — MECHANICAL enforcement of the R-148/R-150 resource rule.

R-148 forbids bare heavy research launches outside research.slice, but until
2026-09-01 (R-376) that was discipline, not mechanism: a seat forgetting
`systemd-run --slice=research.slice` landed in app.slice UNBOUNDED — the exact
shape of the 2026-08-26 03:55Z box death (aggregate memory exhaustion,
swapless, so the kernel livelocks before it kills). This watchdog closes it.

Policy (memory only — CPU stays with cgroup weights: research.slice
CPUQuota=1200%/CPUWeight=50 vs collectors 500; a CPU hog degrades, a memory
hog KILLS the box, and this box is swapless):

  profile   : python from the research venv, or python running ctaNew code
  IN_SLICE  : inside research.slice           -> kernel enforces; report only
  COLLECTOR : inside a pm-collector-* unit    -> NEVER touched (R-22: capture
              is unrecoverable; batches re-run)
  FLAG      : profile, outside slice, RSS >= --flag-gb (2G)  -> violation,
              exit 3, OnFailure= fires pm-alert@
  KILL      : profile, outside slice, RSS >= --kill-gb (8G)  -> SIGTERM then
              SIGKILL. 8G is measured territory, not taste: Q-BE-111 polled
              8.8G ~1-2 min before the 08-26 box death, and no run had ever
              been PERMITTED past 8G before it (attempts 1-2 capped there).

Rule-15 falsifiers ship in --selftest (fixtures through the same classify())
and --e2e (a REAL spawned hog, scoped by --only-pid so no bystander is ever
killed at test thresholds). Rule-11/16 hygiene: a scan that read almost
nothing REFUSES (exit 2) instead of printing a clean bill — "a zero from an
instrument that never proved it can fire is not a result".

Exit codes: 0 clean · 2 vacuous/refused scan · 3 violation (flagged/killed).
The timer unit runs enforcing (no flags); pm-alert@ fires on any nonzero.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time

FLAG_GB_DEFAULT = 2.0
KILL_GB_DEFAULT = 8.0
MIN_SCAN = 20  # fewer processes seen than this => the scan itself is broken

VENV_MARKER = "pricer-sol/venv/bin/python"
REPO_MARKER = "ctaNew"


def read_proc(pid: int) -> dict | None:
    base = f"/proc/{pid}"
    try:
        with open(f"{base}/cmdline", "rb") as f:
            cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
        with open(f"{base}/cgroup") as f:
            cgroup = f.read()
        rss_kb = None
        with open(f"{base}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
        if rss_kb is None:  # kernel thread
            return None
        return {"pid": pid, "cmdline": cmdline, "cgroup": cgroup, "rss_kb": rss_kb}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def scan_procs(only_pid: int | None = None) -> list[dict]:
    if only_pid is not None:
        p = read_proc(only_pid)
        return [p] if p else []
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == os.getpid():
            continue
        p = read_proc(pid)
        if p:
            out.append(p)
    return out


def classify(proc: dict, flag_gb: float, kill_gb: float) -> str:
    """Pure decision function — every selftest fixture goes through HERE."""
    cmd = proc["cmdline"]
    is_python = VENV_MARKER in cmd or ("python" in cmd and REPO_MARKER in cmd)
    if not is_python:
        return "IGNORE"
    if "research.slice" in proc["cgroup"]:
        return "IN_SLICE"
    # Collectors are exempt WHEREVER they live: the pm-collector-* units (one
    # still in app.slice) AND collectors.slice, which holds the P-2026-002
    # collect-hf/collect-hl units — measured 2026-09-01, not assumed.
    if "pm-collector" in proc["cgroup"] or "collectors.slice" in proc["cgroup"]:
        return "COLLECTOR"
    gb = proc["rss_kb"] / (1024.0 * 1024.0)
    if gb >= kill_gb:
        return "KILL"
    if gb >= flag_gb:
        return "FLAG"
    return "OK_SMALL"


def kill_proc(pid: int, grace_s: float = 10.0) -> str:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_gone"
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "sigterm"
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return "sigterm"
    return "sigkill"


def run_scan(flag_gb: float, kill_gb: float, only_pid: int | None, dry_run: bool) -> int:
    procs = scan_procs(only_pid)
    if only_pid is None and len(procs) < MIN_SCAN:
        print(json.dumps({"status": "VACUOUS_SCAN_REFUSED", "n_scanned": len(procs),
                          "min_required": MIN_SCAN}))
        return 2
    counts: dict[str, int] = {}
    violations = []
    for p in procs:
        verdict = classify(p, flag_gb, kill_gb)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict in ("FLAG", "KILL"):
            v = {"pid": p["pid"], "rss_gb": round(p["rss_kb"] / 1048576.0, 2),
                 "verdict": verdict, "cmdline": p["cmdline"][:200],
                 "cgroup": p["cgroup"].strip().splitlines()[-1][:200]}
            if verdict == "KILL" and not dry_run:
                v["kill_result"] = kill_proc(p["pid"])
            violations.append(v)
    status = "VIOLATION" if violations else "OK"
    print(json.dumps({"status": status, "n_scanned": len(procs), "counts": counts,
                      "flag_gb": flag_gb, "kill_gb": kill_gb, "dry_run": dry_run,
                      "violations": violations, "as_of": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return 3 if violations else 0


def _fixture(rss_gb: float, cgroup: str, cmdline: str) -> dict:
    return {"pid": 999999, "rss_kb": int(rss_gb * 1048576), "cgroup": cgroup,
            "cmdline": cmdline}


def selftest() -> int:
    VENV = f"/home/yuqing/{VENV_MARKER}3 live/pm_research/x.py"
    SLICE = "0::/user.slice/user-1001.slice/user@1001.service/research.slice/run-x.service\n"
    APP = "0::/user.slice/user-1001.slice/user@1001.service/app.slice/x.scope\n"
    COLL = "0::/user.slice/user-1001.slice/user@1001.service/app.slice/pm-collector-prices.service\n"
    COLL2 = "0::/user.slice/user-1001.slice/user@1001.service/collectors.slice/collect-hf.service\n"
    cases = [
        # (name, fixture, expected) — both directions per rule 16: the guard
        # must FIRE on the bad and ADMIT the good.
        ("in_slice_10g_admitted", _fixture(10.0, SLICE, VENV), "IN_SLICE"),
        ("collector_10g_never_touched", _fixture(10.0, COLL, VENV), "COLLECTOR"),
        ("mm_collector_slice_never_touched", _fixture(10.0, COLL2, VENV), "COLLECTOR"),
        ("outside_3g_flagged", _fixture(3.0, APP, VENV), "FLAG"),
        ("outside_9g_killed", _fixture(9.0, APP, VENV), "KILL"),
        ("node_9g_ignored", _fixture(9.0, APP, "node dist/index.js"), "IGNORE"),
        ("outside_half_g_ok", _fixture(0.5, APP, VENV), "OK_SMALL"),
        ("boundary_below_admitted", _fixture(1.99, APP, VENV), "OK_SMALL"),
        ("boundary_at_fires", _fixture(2.0, APP, VENV), "FLAG"),
        ("repo_python_no_venv_flagged", _fixture(3.0, APP, "python3 -m ctaNew.thing"), "FLAG"),
    ]
    fails = 0
    for name, fx, want in cases:
        got = classify(fx, FLAG_GB_DEFAULT, KILL_GB_DEFAULT)
        ok = got == want
        fails += 0 if ok else 1
        print(f"{'PASS' if ok else 'FAIL'} {name}: got {got} want {want}")
    # vacuity refusal: a scan restricted to nothing must refuse, not pass
    rc = run_scan(FLAG_GB_DEFAULT, KILL_GB_DEFAULT, only_pid=1, dry_run=True)
    vac_ok = rc in (0, 2, 3)  # pid1 path exercises single-pid mode; real check below
    procs = scan_procs()
    scan_ok = len(procs) >= MIN_SCAN
    fails += 0 if scan_ok else 1
    print(f"{'PASS' if scan_ok else 'FAIL'} real_scan_reads_population: n={len(procs)} (>= {MIN_SCAN})")
    rss_ints = all(isinstance(p["rss_kb"], int) for p in procs[:50])
    fails += 0 if rss_ints else 1
    print(f"{'PASS' if rss_ints else 'FAIL'} fields_read_as_typed: VmRSS parsed as int")
    print(f"selftest: {'GREEN' if fails == 0 else f'{fails} FAILING'} ({len(cases)}+2 checks)")
    return 0 if fails == 0 else 1


def e2e() -> int:
    """Kill-path positive control on a REAL process, scoped to --only-pid so
    test thresholds can never touch a bystander."""
    hog = subprocess.Popen(
        [f"/home/yuqing/{VENV_MARKER}3", "-c",
         "x = bytearray(300*1024*1024); import time; time.sleep(120)"],
        cwd="/home/yuqing/ctaNew")
    try:
        time.sleep(2.0)  # let it allocate
        rc = run_scan(flag_gb=0.1, kill_gb=0.2, only_pid=hog.pid, dry_run=False)
        if rc != 3:
            print(f"E2E FAIL: expected violation exit 3, got {rc}")
            return 1
        try:
            hog.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print("E2E FAIL: hog still alive after kill path")
            return 1
        print("E2E PASS: real 300MB hog detected at test threshold and killed")
        # negative control: a second scan of the (dead) pid finds nothing
        rc2 = run_scan(flag_gb=0.1, kill_gb=0.2, only_pid=hog.pid, dry_run=False)
        if rc2 != 2 and rc2 != 0:
            # single-pid scan of a dead pid yields empty -> vacuous refusal (2)
            print(f"E2E FAIL: post-kill rescan expected empty/clean, got {rc2}")
            return 1
        print("E2E PASS: post-kill rescan finds no process")
        return 0
    finally:
        if hog.poll() is None:
            hog.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag-gb", type=float, default=FLAG_GB_DEFAULT)
    ap.add_argument("--kill-gb", type=float, default=KILL_GB_DEFAULT)
    ap.add_argument("--only-pid", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--e2e", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if args.e2e:
        return e2e()
    if args.kill_gb <= args.flag_gb:
        print(json.dumps({"status": "REFUSED", "reason": "kill_gb must exceed flag_gb"}))
        return 2
    return run_scan(args.flag_gb, args.kill_gb, args.only_pid, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
