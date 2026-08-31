#!/usr/bin/env python3
"""Falsifier for the v5 heartbeat DEADLINE tests — retained, not ad hoc.

Codex V5-C6-1: the previous "behavioural" probes could not tell a correct
deadline from a hard-coded one — `missing_pong` timed out under both and
`healthy` succeeded under both, so an independent mutation replacing the
coroutine's timeout with `0.03` left the whole 22-check suite green. The
repair uses a DELAYED PONG whose delay sits between the wrong and requested
deadlines, so the deadline decides the outcome.

A control that has never fired is not a control (repo rule 15), so this
script PROVES the repair bites: it hard-codes the wrong deadline and requires
the suite to go RED.

It works entirely on COPIES in a temp directory and NEVER writes the repo.
The first version mutated the real `collect_pm.py` in place and restored it —
which silently failed a CONCURRENT `collect_pm_v4_behavior_tests.py` run that
imported the mutated bytes. A control that can corrupt its neighbours is not
a safe control either.

Exit 0 = the falsifier fired (mutant killed, source restored).
Exit 1 = the suite survived the mutation, i.e. the deadline tests are still
         mutation-insensitive and must not be trusted.
"""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "collect_pm.py"
SUITE = HERE / "collect_pm_v5_heartbeat_tests.py"
SITE = "timeout=APP_HEARTBEAT_TIMEOUT_S"
WRONG = "timeout=0.03"


# Every failure the mutant causes must be a DEADLINE check — that is what
# distinguishes a real kill from a flaky or syntax-error red (audit F5).
DEADLINE_MARKERS = ("a deadline ABOVE the same PONG delay",
                    "a deadline BELOW the PONG delay",
                    "at a second scale",
                    "at the same scale")


def run_suite_in(workdir: pathlib.Path):
    r = subprocess.run([sys.executable, str(workdir / SUITE.name)],
                       capture_output=True, text=True, timeout=300,
                       cwd=str(workdir))
    fails = [ln for ln in r.stdout.splitlines() if ln.startswith("  FAIL")]
    return r.returncode, fails, r.stdout


def main() -> int:
    original = SOURCE.read_text()
    before = hashlib.sha256(original.encode()).hexdigest()
    if SITE not in original:
        print(f"FALSIFIER INVALID: mutation site {SITE!r} not found — the "
              f"candidate changed shape and this control now tests nothing")
        return 1

    with tempfile.TemporaryDirectory() as td:
        work = pathlib.Path(td)
        shutil.copy2(SOURCE, work / SOURCE.name)
        shutil.copy2(SUITE, work / SUITE.name)

        # positive control first: the unmutated COPY must pass, or a red
        # result below would prove nothing about the mutation.
        rc0, _, out0 = run_suite_in(work)
        if rc0 != 0 or "PASS" not in out0:
            print("FALSIFIER INVALID: the UNMUTATED copy does not pass, so a "
                  "red mutant would be meaningless")
            return 1
        print("control: unmutated copy passes")

        # audit F6: replace() takes the FIRST textual occurrence, so a
        # doc comment mentioning the site would redirect the mutation and
        # produce a WRONG accusation ("the tests are insensitive") when the
        # mutation never landed.
        if original.count(SITE) != 1:
            print(f"FALSIFIER INVALID: {SITE!r} occurs "
                  f"{original.count(SITE)} times — the mutation would not "
                  f"land where this control claims")
            return 1
        mutated = original.replace(SITE, WRONG, 1)
        if mutated == original:
            print("FALSIFIER INVALID: the mutation changed nothing")
            return 1
        (work / SOURCE.name).write_text(mutated)
        rc, fails, out = run_suite_in(work)

    after = hashlib.sha256(SOURCE.read_text().encode()).hexdigest()
    if after != before:
        print(f"FALSIFIER TOUCHED THE REPO: {before[:16]} -> {after[:16]}")
        return 1
    print(f"repo source untouched, sha256 {after[:16]}")

    if rc == 0:
        print("SURVIVOR: a hard-coded wrong deadline left the suite GREEN — "
              "the deadline tests are mutation-insensitive")
        return 1
    # audit F5: a returncode alone cannot tell a real kill from an unrelated
    # red — a flaky suite or a syntax-error mutant would both have been
    # reported as "fired". Require the mutant to fail EXACTLY the deadline
    # check, and require the run to have executed checks at all.
    if "PASS" not in out:
        print("FALSIFIER INVALID: the mutant run executed no checks (a "
              "syntax error or import failure, not a deadline kill)")
        return 1
    if not fails:
        print("FALSIFIER INVALID: the mutant run reported no FAIL line, so "
              "its non-zero exit was not a check failure")
        return 1
    stray = [f for f in fails
             if not any(m in f for m in DEADLINE_MARKERS)]
    if stray:
        print(f"FALSIFIER INVALID: the mutant failed {len(stray)} "
              f"NON-deadline check(s), so the red is not attributable to "
              f"the deadline: {stray[:1]}")
        return 1
    print(f"falsifier fired: a hard-coded wrong deadline fails "
          f"{len(fails)} check(s), ALL of them deadline checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
