"""A UNIT THAT IS GONE MUST NOT READ AS A UNIT THAT SUCCEEDED.

`systemctl --user show -p Result -p ExecMainStatus <name>` returns
`Result=success ExecMainStatus=0` for a unit that NEVER EXISTED, because
show prints DEFAULTS for an unknown unit. A watcher that polls those two
properties cannot tell "ran and succeeded" from "garbage-collected" or
"never launched" -- and `--collect` removes a failed unit the moment it
exits, so the FAILURE case is exactly the one that disappears.

`LoadState` is the property that tells a known unit from an unknown one,
and it is returned by the SAME query -- but DRIVING IT SHOWED THAT IS NOT
ENOUGH. A transient `systemd-run` unit that SUCCEEDS is freed the moment
it exits, so it reads `not-found` exactly like a failure that was
`--collect`ed. Measured, both ways, in this file's own falsifier.

So the answer has two parts, and the second is the one that matters:

  1. READ IT, NEVER INFER IT. LoadState first; when the unit is gone,
     consult the JOURNAL, which outlives the unit and records
     `EXIT_STATUS` and "Failed with result" for a failure. A freed unit
     with no failure record is AMBIGUOUS and REFUSES -- absence of a
     failure line is not evidence of success.

  2. MAKE THE RECORD SURVIVE. Launch with
     `--property=RemainAfterExit=yes` and the unit stays loaded carrying
     its own `Result`, so the ambiguity never arises. Driven below: the
     same /bin/true that vanishes without it reads SUCCEEDED with it.

  UNIT_IS_ABSENT_NOT_SUCCEEDED   gone, and the journal knows nothing
  UNIT_IS_GONE_AND_ITS_OUTCOME_IS_NOT_RECORDED
                                 gone, journal has a start but no
                                 outcome -- the dangerous case
  UNIT_NEVER_RAN                 loaded, but no main process ever exited
                                 and it is not running now
  UNIT_FAILED                    it failed, whether or not it still exists

Only `SUCCEEDED` is a pass, and it requires a POSITIVE record.

Usage:  de_unit_verdict.py --falsify
        de_unit_verdict.py <unit-name>
"""
from __future__ import annotations

CALL_SITE = {
    "kind": "REQUIRED_BUT_ABSENT",
    "by": "the chain launcher's watcher, which today reads Result and ExecMainStatus directly",
    "why": "OPEN: this exists to stop a collected failure reading as success, and nothing calls it yet",
    "gates": "the verdict a watcher draws from a finished unit",
}

import json
import subprocess
import sys
import time
import uuid

PROTOCOL = "P003_DE_UNIT_VERDICT_V1"

ABSENT = "UNIT_IS_ABSENT_NOT_SUCCEEDED"
UNRECORDED = "UNIT_IS_GONE_AND_ITS_OUTCOME_IS_NOT_RECORDED"
NEVER_RAN = "UNIT_NEVER_RAN"
FAILED = "UNIT_FAILED"
SUCCEEDED = "SUCCEEDED"
RUNNING = "RUNNING"

#: Read in ONE query. LoadState is the discriminator; the rest are only
#: meaningful once it says the unit exists.
PROPERTIES = ("LoadState", "ActiveState", "SubState", "Result",
              "ExecMainStatus", "ExecMainExitTimestampMonotonic",
              "NRestarts")


class UnitVerdictRefused(ValueError):
    """The unit's state cannot be read as a result."""


def show(unit: str, user: bool = True) -> dict:
    cmd = ["systemctl"] + (["--user"] if user else [])
    cmd += ["show"] + [f"-p{p}" for p in PROPERTIES] + [unit]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    got = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            got[k] = v
    return got


def journal_outcome(unit: str, user: bool = True) -> dict:
    """WHAT OUTLIVES THE UNIT. The journal keeps the exit record after
    systemd has freed the unit itself."""
    cmd = ["journalctl"] + (["--user"] if user else [])
    cmd += ["-u", unit, "-o", "json", "--no-pager", "-n", "200"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    started = failed = False
    exit_status = None
    for line in out.splitlines():
        try:
            rec = json.loads(line)
        except Exception:                                   # noqa: BLE001
            continue
        msg = rec.get("MESSAGE") or ""
        if isinstance(msg, list):
            msg = " ".join(map(str, msg))
        if "Started" in msg:
            started = True
        if "Failed with result" in msg or "/FAILURE" in msg:
            failed = True
        if rec.get("EXIT_STATUS") not in (None, ""):
            exit_status = rec["EXIT_STATUS"]
            if str(exit_status) != "0":
                failed = True
    return {"has_journal": bool(out.strip()), "started": started,
            "failed": failed, "exit_status": exit_status}


def verdict(unit: str, user: bool = True, props: dict = None,
            journal: dict = None) -> dict:
    """THE VERDICT. Read, never inferred -- and never from a default."""
    p = props if props is not None else show(unit, user)
    load = p.get("LoadState", "")
    if load != "loaded":
        j = journal if journal is not None else journal_outcome(unit, user)
        if j.get("failed"):
            return {"unit": unit, "verdict": FAILED, "properties": p,
                    "journal": j, "result": "from the journal",
                    "exec_main_status": j.get("exit_status"),
                    "why": "the unit is gone, and the journal records "
                           "that it FAILED -- --collect removes the unit, "
                           "not the record"}
        if j.get("started"):
            raise UnitVerdictRefused(
                f"REFUSED {UNRECORDED}: {unit} is gone (LoadState="
                f"{load!r}) and the journal has a start but no outcome. A "
                f"transient unit that SUCCEEDS is freed the instant it "
                f"exits, so this looks identical to a failure that was "
                f"--collect'ed. `systemctl show` reports Result="
                f"{p.get('Result', '?')!r} here, which is a DEFAULT. "
                f"Launch with --property=RemainAfterExit=yes so the unit "
                f"carries its own Result, or write a completion receipt; "
                f"absence of a failure line is not evidence of success.")
        raise UnitVerdictRefused(
            f"REFUSED {ABSENT}: {unit} has LoadState={load!r} and the "
            f"journal knows nothing about it. `systemctl show` prints "
            f"DEFAULTS for an unknown unit -- Result="
            f"{p.get('Result', '?')!r}, ExecMainStatus="
            f"{p.get('ExecMainStatus', '?')!r} -- which is exactly what a "
            f"SUCCESS looks like.")
    active = p.get("ActiveState", "")
    exited_at = p.get("ExecMainExitTimestampMonotonic", "0")
    if active in ("activating", "active") and exited_at in ("0", ""):
        return {"unit": unit, "verdict": RUNNING, "properties": p,
                "why": "loaded and still running; no result yet"}
    if exited_at in ("0", ""):
        raise UnitVerdictRefused(
            f"REFUSED {NEVER_RAN}: {unit} is loaded but no main process "
            f"ever exited (ExecMainExitTimestampMonotonic=0) and it is "
            f"not running (ActiveState={active!r}). An absent exit is "
            f"not a zero exit.")
    result = p.get("Result", "")
    status = p.get("ExecMainStatus", "")
    if result != "success" or status != "0":
        return {"unit": unit, "verdict": FAILED, "properties": p,
                "result": result, "exec_main_status": status,
                "why": "loaded, ran, and did not succeed"}
    return {"unit": unit, "verdict": SUCCEEDED, "properties": p,
            "checked": {"present": True, "has_exit_record": True,
                        "result": result, "exec_main_status": status},
            "why": "present, exited, Result=success, ExecMainStatus=0"}


def _run_unit(name: str, cmd, collect: bool = False) -> None:
    args = ["systemd-run", "--user", f"--unit={name}"]
    if collect:
        args.append("--collect")
    args += ["--quiet", "--"] + cmd
    subprocess.run(args, capture_output=True, text=True)
    for _ in range(60):
        p = show(name)
        if p.get("LoadState") != "loaded":
            return
        if p.get("ActiveState") not in ("activating", "active"):
            return
        time.sleep(0.25)


def falsify() -> int:
    n = ok = 0

    def ck(name, cond, note=""):
        nonlocal n, ok
        n += 1
        ok += bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  {note}" if note else ""))

    tag = uuid.uuid4().hex[:8]

    print("== THE DEFECT, DEMONSTRATED ON A UNIT THAT NEVER EXISTED ==")
    ghost = f"de-verdict-ghost-{tag}.service"
    p = show(ghost)
    ck("a NON-EXISTENT unit reports Result=success ExecMainStatus=0",
       p.get("Result") == "success" and p.get("ExecMainStatus") == "0",
       f"Result={p.get('Result')} ExecMainStatus={p.get('ExecMainStatus')}")
    ck("  and LoadState is the ONE property that tells them apart",
       p.get("LoadState") == "not-found", f"LoadState={p.get('LoadState')}")
    try:
        verdict(ghost)
        ck("  so the verdict REFUSES on it", False)
    except UnitVerdictRefused as exc:
        ck("  so the verdict REFUSES on it", ABSENT in str(exc))

    print("== THE DISCOVERY: a transient unit that SUCCEEDS also "
          "vanishes ==")
    good = f"de-verdict-good-{tag}.service"
    _run_unit(good, ["/bin/true"])
    pgood = show(good)
    ck("a SUCCESSFUL transient unit is freed too -- LoadState=not-found",
       pgood.get("LoadState") == "not-found",
       "so LoadState alone cannot separate success from a collected "
       "failure")
    try:
        verdict(good)
        ck("  so the verdict REFUSES it as UNRECORDED rather than "
           "guessing", False)
    except UnitVerdictRefused as exc:
        ck("  so the verdict REFUSES it as UNRECORDED rather than "
           "guessing", UNRECORDED in str(exc))

    print("== THE FIX, DRIVEN: RemainAfterExit keeps the record ==")
    kept = f"de-verdict-kept-{tag}.service"
    subprocess.run(["systemd-run", "--user", f"--unit={kept}", "--quiet",
                    "--property=RemainAfterExit=yes", "--", "/bin/true"],
                   capture_output=True, text=True)
    time.sleep(1.0)
    try:
        v = verdict(kept)
        ck("the SAME /bin/true reads SUCCEEDED when the unit is kept",
           v["verdict"] == SUCCEEDED,
           f"LoadState={v['properties'].get('LoadState')} "
           f"Result={v['properties'].get('Result')}")
    except UnitVerdictRefused as exc:
        ck("the SAME /bin/true reads SUCCEEDED when the unit is kept",
           False, str(exc)[:90])

    print("== a real unit that FAILS ==")
    bad = f"de-verdict-bad-{tag}.service"
    _run_unit(bad, ["/bin/false"])
    try:
        v = verdict(bad)
        ck("a unit that ran and failed reads FAILED, never SUCCEEDED",
           v["verdict"] == FAILED,
           f"Result={v.get('result')} status={v.get('exec_main_status')}")
    except UnitVerdictRefused as exc:
        ck("a unit that ran and failed reads FAILED, never SUCCEEDED",
           False, str(exc)[:80])

    print("== THE ONE THAT COSTS A NIGHT: --collect on a FAILURE ==")
    gone = f"de-verdict-collected-{tag}.service"
    _run_unit(gone, ["/bin/false"], collect=True)
    pg = show(gone)
    ck("the collected FAILURE is indistinguishable from success on "
       "Result alone",
       pg.get("Result") == "success" and pg.get("ExecMainStatus") == "0",
       f"Result={pg.get('Result')} after /bin/false with --collect")
    v = verdict(gone)
    ck("  and the verdict reads FAILED from the JOURNAL, which outlives "
       "the unit",
       v["verdict"] == FAILED,
       f"journal exit_status={v['journal'].get('exit_status')}")

    print("== the discriminator is ONE property in the SAME query ==")
    ck("LoadState travels with the properties a watcher already asks for",
       "LoadState" in PROPERTIES and len(PROPERTIES) <= 8,
       f"{len(PROPERTIES)} properties, one query")
    ck("and NO path returns SUCCEEDED without a positive record",
       _no_success_by_default())
    ck("a loaded unit with NO exit record refuses rather than passing",
       _never_ran_refuses())

    for u in (good, bad, kept):
        subprocess.run(["systemctl", "--user", "reset-failed", u],
                       capture_output=True)
        subprocess.run(["systemctl", "--user", "stop", u],
                       capture_output=True)

    print(f"\n{ok}/{n} cells pass")
    return 0 if ok == n else 1


def _no_success_by_default() -> bool:
    """A unit that is gone, with a journal that says nothing, must never
    come back SUCCEEDED."""
    props = {"LoadState": "not-found", "ActiveState": "inactive",
             "Result": "success", "ExecMainStatus": "0",
             "ExecMainExitTimestampMonotonic": "0"}
    for j in ({"has_journal": False, "started": False, "failed": False},
              {"has_journal": True, "started": True, "failed": False}):
        try:
            verdict("fixture.service", props=props, journal=j)
            return False
        except UnitVerdictRefused:
            continue
    return True


def _never_ran_refuses() -> bool:
    props = {"LoadState": "loaded", "ActiveState": "inactive",
             "SubState": "dead", "Result": "success",
             "ExecMainStatus": "0", "ExecMainExitTimestampMonotonic": "0"}
    try:
        verdict("fixture.service", props=props)
        return False
    except UnitVerdictRefused as exc:
        return NEVER_RAN in str(exc)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--falsify" in argv:
        return falsify()
    if argv:
        try:
            print(json.dumps(verdict(argv[0]), indent=2))
            return 0
        except UnitVerdictRefused as exc:
            print(str(exc))
            return 3
    print(json.dumps({"protocol": PROTOCOL, "properties": PROPERTIES},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
