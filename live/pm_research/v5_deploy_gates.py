#!/usr/bin/env python3
"""Run EVERY gate the v5 deploy package must pass, and fail loud.

Why this exists, and it is not a convenience wrapper.

Each gate already returns a correct exit code. The harnesses GATE, they do
not merely report (DA `21ae821` raised the distinction; both sides check
out). The defect was one layer UP: I ran them by hand, piped the output
through `tail` to read the summary, and `tail`'s exit code replaced the
gate's. **Twice in this programme that put a commit on a red suite**
(`22ad2c4`, `435dfe7`). A correct gate invoked through a pipe is a gate that
cannot fail.

So the rule this file mechanises: **the thing that decides must be the thing
that ran.** Output is captured, never piped; the exit code is read directly
from the process; a non-zero anywhere fails the whole run; and a gate that
cannot be FOUND is a failure, not a skip — an absent gate and a passing gate
must never look alike (repo rule 4: exclusions are statuses, never silent
drops).

    python3 live/pm_research/v5_deploy_gates.py          # run them all
    python3 live/pm_research/v5_deploy_gates.py --falsify  # prove it fails

`--falsify` is the control this instrument ships under repo rule 15: it
injects a gate that is KNOWN to fail and requires the runner to report
FAILED. A runner that has never been seen to fail is not evidence that
anything passed.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

# (label, argv). Order is cheapest-first so a broken tree fails fast.
GATES: list[tuple[str, list[str]]] = [
    ("collector selftest", [PY, str(HERE / "collect_pm.py"), "--selftest"]),
    ("v5 heartbeat behaviour",
     [PY, str(HERE / "collect_pm_v5_heartbeat_tests.py")]),
    ("v5 deadline falsifier",
     [PY, str(HERE / "collect_pm_v5_deadline_falsifier.py")]),
    ("preflight selftest",
     [PY, str(HERE / "v5_boundary_preflight.py"), "--selftest"]),
    ("DA day-verifier selftest",
     [PY, str(HERE / "da_forward_day_verify.py"), "--selftest"]),
    # A GATE NOBODY RUNS IS NOT A GATE (R-370, which added four such
    # instruments at once). Added with the rule itself so its selftest cannot
    # sit committed and uninvoked; the rule GOVERNS nothing either way.
    ("DA content-liveness rule (DRAFT, governs nothing)",
     [PY, str(HERE / "da_content_liveness_rule.py"), "--selftest"]),
    ("DA closed-day verdict checker",
     [PY, str(HERE / "da_verdict_check.py"), "--selftest"]),
    ("DA cross-venue forensics", 
     [PY, str(HERE / "da_cross_venue_forensics.py"), "--selftest"]),
    # DRAFT checker: its SELFTEST runs, nothing consumes its output. This is
    # not wiring -- the v2 rule is not in any verdict path.
    ("DA content-liveness v2 amendment checker (DRAFT)",
     [PY, str(HERE / "da_content_liveness_v2_check.py"), "--selftest"]),
    ("chain equivalence (one fixture, two consumers)",
     [PY, str(HERE / "v5_chain_equivalence_test.py")]),
    ("chain differential fuzz",
     [PY, str(HERE / "v5_chain_differential_fuzz.py")]),
    ("preflight mutation audit",
     [PY, str(HERE / "v5_preflight_mutation_audit.py")]),
    ("tape density", [PY, str(HERE / "pm_tape_density.py"), "--selftest"]),
    ("host-load join", [PY, str(HERE / "pm_host_load_join.py"), "--selftest"]),
    ("shadow observer", [PY, str(HERE / "pm_shadow_observer.py"),
                         "--selftest"]),
    ("v4 behaviour (git-extracted)",
     [PY, str(HERE / "collect_pm_v4_behavior_tests.py")]),
    ("v4_1 boundary gate",
     [PY, str(HERE / "v41_boundary_preflight.py"), "--selftest"]),
    ("v4_1 mutation audit",
     [PY, str(HERE / "v41_preflight_mutation_audit.py")]),
    # Tier-1 normalisation was NOT in this list while it held the check that
    # hard-blocked tier1:full and tier2 for 146 h (`0 < price < 1` on real
    # settlement-edge prints). Its selftest costs 0.4 s. A file this suite
    # never invokes is a file this suite does not gate.
    ("tier1 normalisation",
     [PY, "-m", "live.pm_research.tier1_pipeline", "--selftest"]),
]

# A gate whose script is missing must FAIL, not vanish from the tally.
FAILING_CANARY = ("INJECTED FAILING CANARY (--falsify only)",
                  [PY, "-c", "import sys; print('canary'); sys.exit(3)"])


def run_one(label: str, argv: list[str]) -> tuple[bool, str, float]:
    """Capture; never pipe. The exit code read here is the gate's own."""
    script = Path(argv[1]) if len(argv) > 1 and argv[1].endswith(".py") \
        else None
    if script is not None and not script.exists():
        return False, f"gate script NOT FOUND: {script}", 0.0
    t0 = time.time()
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=3600,
                           cwd=str(HERE.parent.parent))
    except subprocess.TimeoutExpired:
        return False, "TIMED OUT after 3600s", time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    tail = [ln for ln in out.strip().splitlines() if ln.strip()]
    return p.returncode == 0, (tail[-1][:120] if tail else "(no output)"), \
        time.time() - t0


def main() -> int:
    falsify = "--falsify" in sys.argv
    gates = list(GATES) + ([FAILING_CANARY] if falsify else [])
    if falsify:
        print("FALSIFIER MODE: a gate that MUST fail is injected. This run "
              "is expected to report FAILED; if it reports PASSED, the "
              "runner cannot detect a red gate and none of its green "
              "results mean anything.\n")

    failed = []
    for label, argv in gates:
        okay, summary, secs = run_one(label, argv)
        print(f"  {'PASS' if okay else 'FAIL'}  {label}  ({secs:.0f}s)  "
              f"{summary}")
        if not okay:
            failed.append(label)

    print()
    if failed:
        print(f"FAILED: {len(failed)} of {len(gates)} gates — {failed}")
        if falsify:
            print("falsifier fired: the runner DOES report a red gate")
            return 0 if failed == [FAILING_CANARY[0]] else 1
        return 1
    if falsify:
        print("FALSIFIER DID NOT FIRE: an injected always-failing gate was "
              "reported as a pass — this runner cannot detect a red gate")
        return 1
    print(f"ALL {len(gates)} GATES PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
