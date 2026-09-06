"""R-628 DRIVEN AGAINST A REAL SESSION MANAGER: scope dies with its
launcher, a transient service does not.

WHY THIS IS A SEPARATE FILE. The runner's battery must not need a systemd
user manager -- every seat runs it, and a checker that fails where there is
no session bus is a checker that teaches people to ignore it. So the
runner's battery checks the COMMAND'S SHAPE (`assert_launch_form`) and this
probe checks the BEHAVIOUR the shape is chosen for.

WHAT IT COSTS THE PROGRAMME TO NOT HAVE HAD THIS. The 09-03 re-run was
launched under `systemd-run --scope` from a background tool shell. A scope
registers processes the CALLER forks, so the run sat in that shell's
process group; when the harness stopped the background task the run died at
35 minutes with 34m52s of CPU spent and nothing written.

THE THREE LEGS, each with its falsifier:
  1. a SCOPE launched from a shell dies when that shell's process group is
     TERMed                                      -- the failure, reproduced
  2. a SERVICE survives the same TERM             -- the fix, driven
     and `systemctl --user stop` still ends it    -- so it is stoppable
  3. a SERVICE whose ExecStart is `flock -n` on a HELD lock exits 1 and
     the payload NEVER RUNS -- read `ExecMainStatus`, never assume the run
     started

    python3 live/pm_research/de_launch_form_probe.py --selftest
    python3 live/pm_research/de_launch_form_probe.py --emit OUT.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PROTOCOL = "P003_DE_LAUNCH_FORM_PROBE_V1"
HERE = Path(__file__).resolve().parent
SLICE = "research.slice"
SETTLE_S = 2.0


def _sc(*a) -> str:
    r = subprocess.run(["systemctl", "--user", *a],
                       capture_output=True, text=True, timeout=30)
    return r.stdout.strip()


def _state(unit: str) -> str:
    return _sc("show", unit, "-p", "ActiveState", "--value") or "gone"


def _prop(unit: str, name: str) -> str:
    return _sc("show", unit, "-p", name, "--value")


def _cleanup(unit: str) -> None:
    subprocess.run(["systemctl", "--user", "stop", unit],
                   capture_output=True, timeout=30)
    subprocess.run(["systemctl", "--user", "reset-failed", unit],
                   capture_output=True, timeout=30)


def _launch_from_a_shell(argv: list, unit: str) -> int:
    """Start `argv` from a SHELL of our own, in its own process group, and
    return that shell's PGID -- the thing a harness stops."""
    proc = subprocess.Popen(argv, start_new_session=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    time.sleep(SETTLE_S)
    try:
        return os.getpgid(proc.pid)
    except ProcessLookupError:
        # systemd-run returns immediately for a service; the launcher is
        # already gone, which is itself the point.
        return proc.pid


def leg_scope(seconds: int = 60) -> dict:
    """THE FAILURE, REPRODUCED: a scope dies with its launcher's group."""
    unit = "de94leg1.scope"
    _cleanup(unit)
    argv = ["systemd-run", "--user", "--scope", f"--unit={unit}",
            f"--slice={SLICE}", "/bin/sleep", str(seconds)]
    pgid = _launch_from_a_shell(argv, unit)
    before = _state(unit)
    try:
        os.killpg(pgid, 15)
    except (ProcessLookupError, PermissionError):
        pass
    time.sleep(SETTLE_S)
    after = _state(unit)
    _cleanup(unit)
    return {"form": "--scope", "unit": unit,
            "state_before_the_group_TERM": before,
            "state_after_the_group_TERM": after,
            "died_with_its_launcher": before == "active" and after != "active",
            "why": ("a scope registers processes the CALLER forks, so the "
                    "payload is in the launching shell's process group")}


def leg_service(seconds: int = 60) -> dict:
    """THE FIX, DRIVEN: a service survives the same TERM, and stops."""
    unit = "de94leg2.service"
    _cleanup(unit)
    argv = ["systemd-run", "--user", f"--unit={unit}", f"--slice={SLICE}",
            "--working-directory=/tmp", "--", "/bin/sleep", str(seconds)]
    pgid = _launch_from_a_shell(argv, unit)
    before = _state(unit)
    main = _prop(unit, "MainPID")
    ppid = None
    if main and main.isdigit() and int(main) > 0:
        try:
            ppid = os.stat(f"/proc/{main}").st_uid and int(
                open(f"/proc/{main}/status").read().split(
                    "PPid:")[1].split()[0])
        except (OSError, IndexError, ValueError):
            ppid = None
    try:
        os.killpg(pgid, 15)
    except (ProcessLookupError, PermissionError):
        pass
    time.sleep(SETTLE_S)
    survived = _state(unit)
    _sc("stop", unit)
    time.sleep(SETTLE_S)
    stopped = _state(unit)
    _cleanup(unit)
    return {"form": "transient service", "unit": unit,
            "main_pid": main, "main_pid_PPid": ppid,
            "forked_by_the_manager": ppid is not None and ppid != os.getpid(),
            "state_before_the_group_TERM": before,
            "state_after_the_group_TERM": survived,
            "survived_the_group_TERM": before == "active"
                                       and survived == "active",
            "state_after_systemctl_stop": stopped,
            "systemctl_stop_ends_it": stopped != "active",
            "why": ("the MANAGER forks a service, so it is in its own "
                    "process group and nothing that happens to a tool "
                    "shell can reach it")}


def leg_held_lock(lock_path: str) -> dict:
    """A HELD LOCK EXITS 1 INSIDE THE UNIT and the payload never runs."""
    unit = "de94leg3.service"
    _cleanup(unit)
    # THE PAYLOAD LEAVES A FILE, NOT A STRING IN THE LOG. The first
    # version echoed a marker and searched the journal for it -- and
    # systemd's own `Started <unit> - <the full command>.` line CONTAINS
    # the command, marker and all, so the check reported the payload had
    # run when nothing had run at all. The needle matched its own prose,
    # in the checker written to catch exactly that class.
    with tempfile.TemporaryDirectory() as td:
        ran = Path(td) / "THE_PAYLOAD_RAN"
        holder = subprocess.Popen(
            ["flock", "-n", lock_path, "/bin/sleep", "20"],
            start_new_session=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
        time.sleep(1.0)
        held_by_us = holder.poll() is None
        subprocess.run(
            ["systemd-run", "--user", f"--unit={unit}", f"--slice={SLICE}",
             f"--working-directory={td}", "--",
             "flock", "-n", lock_path, "/bin/touch", str(ran)],
            capture_output=True, timeout=30)
        time.sleep(SETTLE_S)
        status = _prop(unit, "ExecMainStatus")
        result = _prop(unit, "Result")
        payload_ran = ran.exists()
        jr = subprocess.run(
            ["journalctl", "--user", "-u", unit, "--no-pager", "-n", "20"],
            capture_output=True, text=True, timeout=30).stdout
        _cleanup(unit)
        holder.terminate()
        holder.wait(timeout=10)
    return {"unit": unit,
            "the_lock_was_held_by_this_probe": held_by_us,
            "note_if_not": ("another seat's run may hold it; the leg is "
                            "valid either way -- what it tests is that a "
                            "HELD lock exits the unit, not who held it"),
            "ExecMainStatus": status, "Result": result,
            "exited_nonzero": status not in ("", "0"),
            "the_payload_never_ran": not payload_ran,
            "how_that_is_known": (
                "the payload would have CREATED A FILE. Searching the "
                "journal for a marker string reported the opposite: "
                "systemd's own `Started <unit> - <command>.` line carries "
                "the command, marker included"),
            "journal_tail": jr.strip().splitlines()[-3:],
            "why": ("`flock -n` is the unit's OWN ExecStart, so a held "
                    "lock is the unit's exit status -- read it, never "
                    "assume the run started")}


def probe(lock_path: str) -> dict:
    scope, service = leg_scope(), leg_service()
    lock = leg_held_lock(lock_path)
    return {
        "protocol": PROTOCOL,
        "as_of": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "ruling": "R-628: a heavy run is NEVER a child of a tool shell",
        "leg_1_the_failure_reproduced": scope,
        "leg_2_the_fix_driven": service,
        "leg_3_a_held_lock_inside_the_unit": lock,
        "the_ruling_holds": bool(
            scope["died_with_its_launcher"]
            and service["survived_the_group_TERM"]
            and service["systemctl_stop_ends_it"]
            and lock["exited_nonzero"] and lock["the_payload_never_ran"]),
        "what_this_is_not": {
            "a_day_result": False,
            "heavy": "three sleeps and an echo; it takes no heavy lock and "
                     "holds one only for the second it needs to prove a "
                     "held lock refuses",
        },
    }


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            print(f"  FAIL  {label}")
            raise SystemExit(f"[de_launch_form_probe] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    d = probe("/home/yuqing/ctaNew/data/.heavy_run.lock")
    s1, s2, s3 = (d["leg_1_the_failure_reproduced"],
                  d["leg_2_the_fix_driven"],
                  d["leg_3_a_held_lock_inside_the_unit"])
    ok(s1["died_with_its_launcher"] is True,
       f"LEG 1, THE FAILURE REPRODUCED: a `--scope` run is "
       f"{s1['state_before_the_group_TERM']} and becomes "
       f"{s1['state_after_the_group_TERM']} when its launcher's PROCESS "
       f"GROUP is TERMed. That is how the 09-03 re-run lost 35 minutes")
    ok(s2["survived_the_group_TERM"] is True,
       f"LEG 2, THE FIX: a transient SERVICE is "
       f"{s2['state_after_the_group_TERM']} after the SAME group TERM. "
       f"The manager forks it (MainPID {s2['main_pid']}, PPid "
       f"{s2['main_pid_PPid']}), so no tool shell is in its ancestry")
    ok(s2["systemctl_stop_ends_it"] is True,
       "AND IT IS STILL STOPPABLE -- `systemctl --user stop` ends it. A "
       "run nothing can stop would be a worse defect than one anything "
       "can stop")
    ok(s3["exited_nonzero"] is True and s3["the_payload_never_ran"] is True,
       f"LEG 3: with the lock HELD, the unit exits "
       f"{s3['ExecMainStatus']} (Result {s3['Result']}) and the payload "
       f"NEVER RAN. The lock is the unit's own ExecStart, so `read "
       f"ExecMainStatus, never assume the run started` is a fact a caller "
       f"can check rather than advice")
    ok(d["the_ruling_holds"] is True,
       "R-628 HOLDS on all three legs, driven against a real session "
       "manager rather than reasoned about")
    print(f"[de_launch_form_probe] PASS -- {n[0]} checks")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit", type=Path)
    ap.add_argument("--lock", default="/home/yuqing/ctaNew/data/"
                                      ".heavy_run.lock")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    d = probe(a.lock)
    if a.emit:
        a.emit.parent.mkdir(parents=True, exist_ok=True)
        a.emit.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"the_ruling_holds": d["the_ruling_holds"],
                      "emitted": str(a.emit) if a.emit else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
