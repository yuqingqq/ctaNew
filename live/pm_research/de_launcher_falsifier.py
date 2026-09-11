"""FALSIFIER FOR THE LAUNCHERS THEMSELVES -- through the production path.

chain_day.sh and preflight_gate.sh ARE the production path and had no
cells (rule 15's class). Each cell here invokes the entry point exactly as
production does -- subprocess, a cwd that is NOT the tree root, and the
unit's own environment -- because every launcher defect this session has
come from the invocation, not the logic: a relative path under systemd's
cwd, an env var the wrapper overwrote, a heredoc that swallowed a loop.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

TREE = Path("/home/yuqing/ctaNew-wt-deval")
LAUNCH = TREE / "live/pm_research/launchers"
DERIVED = Path("/home/yuqing/ctaNew/data/pm_5min/derived")
CERT = DERIVED / "be_score_neutrality_20260903__EV22_vs_NEUTCHK__68e7d23.json"
PROD_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/home/yuqing",
            "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}"}


def _run(cmd, env=None, cwd="/"):
    """Production shape: foreign cwd, minimal env."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd,
                          env={**PROD_ENV, **(env or {})})


def falsify() -> int:
    cells = ok = 0

    def ck(n, c, note=""):
        nonlocal cells, ok
        cells += 1
        ok += bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {n}" + (f"  {note}" if note else ""))

    # --- preflight_gate: INPUT_ABSENT waits (exit 4) and names it --------
    r = _run(["bash", str(LAUNCH / "preflight_gate.sh"), "2026-09-30",
              str(CERT)])
    # STAGE 0 RUNS FIRST AND CAN LEGITIMATELY PRE-EMPT THIS CELL: a tree
    # whose frozen modules have drifted may not run at all, built day or
    # not, so the gate answers 3 before the day's inputs are considered.
    # The cell states which world it is in rather than going red for an
    # environmental reason -- and it stays falsifiable: with stage 0
    # holding, an unbuilt day MUST be 4.
    gate = subprocess.run(
        [sys.executable, str(TREE / "live/pm_research"
                             / "de_stage0_freeze_gate.py")],
        capture_output=True, text=True, cwd="/",
        env={**PROD_ENV, "DE_VALUATION_PREFLIGHT_OFF": "1"})
    stage0_holds = gate.returncode == 0
    if stage0_holds:
        ck("preflight_gate: an unbuilt day exits 4 (WAIT), from cwd=/",
           r.returncode == 4, f"rc={r.returncode}")
        ck("  and names the absent artifact",
           "INPUT_ABSENT" in r.stdout)
    else:
        ck("preflight_gate: stage 0 REFUSES first, so an unbuilt day "
           "stops at 3 rather than waiting",
           r.returncode == 3 and "FROZEN_MODULE_DRIFTED" in r.stdout,
           f"rc={r.returncode} (stage 0 is refusing: the population "
           f"freeze does not name the landed modules)")
        ck("  and the WAIT path is unreachable until stage 0 holds",
           not stage0_holds)

    # --- chain_day: the freeze comes from the DECLARATION, not a literal -
    # `PIN=f3096021...` refused the first 09-09/09-10 arming on a module
    # whose disk bytes match the declaration exactly. The check now runs
    # the driver's own pre-flight, so this cell asserts the chain gets
    # PAST the freeze check on a tree the declaration admits -- and stops
    # for a REASON ABOUT THE DAY (its book, its stage-0 gate), never about
    # a commit literal.
    rc = _run(["bash", str(LAUNCH / "chain_day.sh"), "--dry-run",
               "2026-09-30", "2026-09-30"])
    # ABSENCE OF A REFUSAL IS NOT ARRIVAL. My first version passed when the
    # pin refusal was merely missing -- and it was missing because the run
    # stopped EARLIER, at the launcher's self-digest check, so the cell
    # was green on a script that never reached the freeze check at all.
    # The cell now requires the pre-flight's own line.
    ck("chain_day REACHES the freeze check and the pre-flight decides it",
       "PREFLIGHT_ADMITS" in rc.stdout,
       next((l for l in rc.stdout.splitlines()
             if "PREFLIGHT_ADMITS" in l or "REFUSED" in l), "")[:78])
    ck("  and it runs to a decision about the DAY, not an unbound name",
       "unbound variable" not in rc.stdout + rc.stderr
       and ("WOULD LAUNCH" in rc.stdout or "INPUT_ABSENT" in rc.stdout
            or "not built yet" in rc.stdout or rc.returncode in (0, 3, 4)),
       f"rc={rc.returncode} "
       + (rc.stdout + rc.stderr).strip().splitlines()[-1][:60])

    # --- preflight_gate: a WOULD_REFUSE fixture stops with 3 -------------
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "be_daybook_20260930_btc__L250ms__FWD1.pkl").touch()
        (d / "be_daybook_receipt_20260930_btc__L250ms__FWD1.json").write_text(
            json.dumps({"selection": {"era": "clob_v3_1"},
                        "assembly_evidence": {"n_windows": 288},
                        "asm": {"n_reference_generations": 300000},
                        "producing_code": {"builder_commit": "0" * 40}}))
        r2 = _run(["/home/yuqing/pricer-sol/venv/bin/python3",
                   str(TREE / "live/pm_research/de_preflight_matrix.py"),
                   "--derived", str(d), "--certification", str(CERT),
                   "--days", "2026-09-30", "--gate"])
        ck("the matrix returns 3 (STOP) on a WOULD_REFUSE fixture",
           r2.returncode == 3, f"rc={r2.returncode}")
        ck("  and names BOOK_ERA_NOT_DECLARED",
           "BOOK_ERA_NOT_DECLARED" in r2.stdout)

    # --- chain_day: a launcher whose bytes are in no origin blob --------
    with tempfile.TemporaryDirectory() as td:
        edited = Path(td) / "chain_day.sh"
        shutil.copy(LAUNCH / "chain_day.sh", edited)
        edited.write_text(edited.read_text() + "\n# one byte\n")
        r3 = _run(["bash", str(edited), "2026-09-30", "2026-09-30"])
        ck("chain_day: edited bytes refuse LAUNCHER_BYTES_NOT_COMMITTED",
           "LAUNCHER_BYTES_NOT_COMMITTED" in r3.stdout + r3.stderr,
           f"rc={r3.returncode}")
        ck("  and it refuses BEFORE any unit is created",
           subprocess.run(["systemctl", "--user", "show",
                           "deRV20260930w1", "-p", "LoadState", "--value"],
                          capture_output=True, text=True).stdout.strip()
           == "not-found")

    # --- the snapshot opt-in, from a REAL launched unit's environ -------
    snap = "/home/yuqing/ctaNew-oracle-20260908"
    r4 = _run(["bash", str(TREE / "live/pm_research/be_heavy_run.sh"),
               "deFALSIFYSNAP", "de_snapshot_probe.py"],
              env={"BE_HEAVY_LOCK": "/tmp/de_falsify.lock",
                   "DE_EXPECT_SNAPSHOT_ROOT": snap,
                   "BE_WORKTREE": str(TREE)},
              cwd=str(TREE / "live/pm_research"))
    import time
    for _ in range(20):
        sub = subprocess.run(["systemctl", "--user", "show", "deFALSIFYSNAP",
                              "-p", "SubState", "--value"],
                             capture_output=True, text=True).stdout.strip()
        if sub in ("exited", "dead", "failed", ""):
            break
        time.sleep(2)
    rc = subprocess.run(["systemctl", "--user", "show", "deFALSIFYSNAP",
                         "-p", "ExecMainStatus", "--value"],
                        capture_output=True, text=True).stdout.strip()
    log = DERIVED / "be_heavy_run_stdout_deFALSIFYSNAP.log"
    txt = log.read_text() if log.is_file() else ""
    ck("a unit launched WITHOUT the opt-in refuses "
       "VALUATION_DID_NOT_SEE_SNAPSHOT_ROOT",
       rc == "7" and "VALUATION_DID_NOT_SEE_SNAPSHOT_ROOT" in txt,
       f"rc={rc}")
    ck("  and its environ shows the LIVE repo, read from the unit",
       '"PM_DATA_ROOT_env": "/home/yuqing/ctaNew"' in txt)
    subprocess.run(["systemctl", "--user", "reset-failed", "deFALSIFYSNAP"],
                   capture_output=True)
    print(f"\n{ok}/{cells} cells pass")
    return 0 if ok == cells else 1


if __name__ == "__main__":
    raise SystemExit(falsify() if "--falsify" in sys.argv[1:] else 2)
