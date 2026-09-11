"""REFUSE when a RUNNING unit's script was modified after its run started.

THE DEFECT THIS EXISTS FOR (REV 157(b)). A unit's ExecStart names a script by
PATH. systemd reads that path once, at launch; nothing re-reads it and nothing
notices if the file changes underneath a run that is still going. The unit
keeps reporting the command it was started with, so `systemctl show` and the
journal both look correct while the bytes on disk are no longer the bytes that
ran.

WHY MTIME *AND* SHA256, which is the whole design. **mtime DETECTS; the digest
DISCRIMINATES.** An mtime later than the run's start says the file was written;
it cannot say whether the content changed. A `touch`, a `chmod -t`, an editor
writing identical bytes all move mtime and change nothing. So the refusal
reports BOTH: mtime establishes that a write happened, and the digest lets a
reader tell an edit from a touch -- by comparing it against whatever record of
the original exists, or across two runs of this checker.

WHAT IT CANNOT DO, named because it bounds the answer: it has NO record of the
bytes that actually ran. It can say "this file was written after the run
started"; it cannot say "the running process is executing different logic",
because the process may have read the whole script into memory at exec, or may
re-read it (bash reads scripts incrementally). The refusal is a flag to
investigate, not a proof of divergence.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

#: TOLERANCE IS DERIVED FROM THE CLOCK, NOT FITTED TO THE DATA. systemd's
#: `ExecMainStartTimestamp` is rendered to WHOLE SECONDS ("Fri 2026-09-11
#: 09:11:29 UTC"), so the start instant it reports can be up to 1.0 s LATER
#: than the true start. An mtime within 1.0 s after the reported start is
#: therefore INDISTINGUISHABLE from a write that preceded the launch -- which
#: is the ordinary write-then-launch pattern, not drift. Anything beyond that
#: cannot be explained by the timestamp's resolution.
#: Found by running the checker: its first pass named two units at +0.7 s and
#: +0.5 s, and a threshold chosen to exclude THOSE would have been fitted to
#: the observations. This one comes from the format.
TOLERANCE_S = 1.0
DRIFT = "UNIT_SCRIPT_MODIFIED_AFTER_ITS_RUN_STARTED"
NO_UNITS = "UNIT_SCRIPT_SCAN_FOUND_NO_RUNNING_UNITS"


class UnitScriptDrift(RuntimeError):
    pass


def _sh(args: list[str]) -> str:
    return subprocess.run(args, capture_output=True, text=True).stdout


def running_units() -> list[str]:
    out = _sh(["systemctl", "--user", "list-units", "--type=service",
               "--state=running", "--no-pager", "--no-legend"])
    return [ln.split()[0] for ln in out.splitlines() if ln.strip()]


def _start_epoch(unit: str) -> float | None:
    raw = _sh(["systemctl", "--user", "show", unit,
               "-p", "ExecMainStartTimestampMonotonic",
               "-p", "ExecMainStartTimestamp"])
    m = re.search(r"ExecMainStartTimestamp=(.+)", raw)
    if not m or not m.group(1).strip() or m.group(1).strip() == "n/a":
        return None
    try:
        dt = datetime.strptime(m.group(1).strip(), "%a %Y-%m-%d %H:%M:%S %Z")
        return dt.replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _argv_paths(unit: str) -> list[str]:
    raw = _sh(["systemctl", "--user", "show", unit, "-p", "ExecStart"])
    m = re.search(r"argv\[\]=(.*?) ; ignore_errors", raw, re.S)
    if not m:
        return []
    # every token that names an existing FILE -- the interpreter, the script,
    # and any path-valued option VALUE (--book <path>) alike.
    return [t for t in m.group(1).split() if t.startswith("/") and os.path.isfile(t)]


def scan(units=None) -> dict:
    units = running_units() if units is None else units
    if not units:
        raise UnitScriptDrift(f"REFUSED {NO_UNITS}: nothing is running to check")
    rows, drifted = [], []
    for u in units:
        start = _start_epoch(u)
        if start is None:
            continue
        for p in _argv_paths(u):
            st = os.stat(p)
            row = {"unit": u, "path": p, "mtime_utc": datetime.fromtimestamp(
                       st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "start_utc": datetime.fromtimestamp(
                       start, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "sha256_16": hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16],
                   "seconds_after_start": round(st.st_mtime - start, 1)}
            row["modified_after_start"] = row["seconds_after_start"] > TOLERANCE_S
            row["within_timestamp_resolution"] = (
                0 < row["seconds_after_start"] <= TOLERANCE_S)
            rows.append(row)
            if row["modified_after_start"]:
                drifted.append(row)
    near = [r for r in rows if r["within_timestamp_resolution"]]
    return {"n_units": len(units), "n_paths": len(rows), "tolerance_s": TOLERANCE_S,
            "drifted": drifted,
            "WITHIN_TIMESTAMP_RESOLUTION_reported_not_dropped": near,
            "rows": rows}


def check(units=None) -> dict:
    r = scan(units)
    if r["drifted"]:
        names = ", ".join(f"{d['unit']}:{Path(d['path']).name}" for d in r["drifted"])
        raise UnitScriptDrift(
            f"REFUSED {DRIFT}: {len(r['drifted'])} path(s) written after their "
            f"unit started -- {names}. Each row carries mtime AND sha256: mtime "
            f"says a write happened, the digest says whether the bytes changed. "
            f"First: {r['drifted'][0]}")
    return {"status": "NO_RUNNING_UNIT_SCRIPT_DRIFTED",
            "n_units": r["n_units"], "n_paths": r["n_paths"],
            "tolerance_s": r["tolerance_s"],
            "within_timestamp_resolution":
                r["WITHIN_TIMESTAMP_RESOLUTION_reported_not_dropped"]}


def falsify() -> int:
    import tempfile, time
    checks, fails = [], 0

    def ck(label, cond):
        nonlocal fails
        checks.append(label)
        if not cond:
            fails += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    with tempfile.TemporaryDirectory() as td:
        s = Path(td) / "fixture.sh"
        s.write_text("#!/bin/bash\necho hi\n")
        old = s.stat().st_mtime

        def fake(start_offset):
            # a synthetic row, exercising the SAME comparison the scan makes
            st = os.stat(s)
            start = old + start_offset
            return {"modified_after_start": st.st_mtime > start,
                    "sha256_16": hashlib.sha256(s.read_bytes()).hexdigest()[:16]}

        def flags(off):
            return (os.stat(s).st_mtime - (old + off)) > TOLERANCE_S
        ck("a script older than its start does NOT flag", not flags(+60))
        ck("a script newer than its start DOES flag", flags(-60))
        ck("a write INSIDE the timestamp's 1 s resolution does NOT flag", not flags(-0.7))
        ck("  and a write just OUTSIDE it DOES", flags(-1.6))
        d0 = hashlib.sha256(s.read_bytes()).hexdigest()[:16]

        # a TOUCH moves mtime and leaves the digest alone -- the discrimination
        os.utime(s, (old + 120, old + 120))
        ck("a TOUCH flags on mtime", os.stat(s).st_mtime > old)
        ck("  and the digest is UNCHANGED, so a reader can tell touch from edit",
           hashlib.sha256(s.read_bytes()).hexdigest()[:16] == d0)
        # an EDIT moves both
        s.write_text("#!/bin/bash\necho changed\n")
        ck("an EDIT moves the digest too",
           hashlib.sha256(s.read_bytes()).hexdigest()[:16] != d0)

    # the real scan must be reachable and shaped right
    try:
        r = scan()
        ck("the real scan returns rows", r["n_paths"] > 0)
        ck("every row carries BOTH mtime and sha256",
           all("mtime_utc" in x and "sha256_16" in x for x in r["rows"]))
    except UnitScriptDrift as e:
        ck(f"the real scan ran ({str(e)[:40]})", False)

    print(json.dumps({"falsifier": "da_unit_script_drift",
                      "n": len(checks), "n_failed": fails}))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--falsify", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.falsify:
        sys.exit(falsify())
    try:
        print(json.dumps(check(), indent=1))
    except UnitScriptDrift as e:
        print(str(e))
        if a.json:
            print(json.dumps(scan()["drifted"], indent=1))
        sys.exit(3)
