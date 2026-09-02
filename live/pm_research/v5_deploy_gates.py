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
    ("DA blackout mask + complement (R-409)",
     [PY, str(HERE / "da_blackout_mask.py"), "--selftest"]),
    # DA10-R3: a launcher break cannot sit uninvoked (R-370). This module's
    # `-m` launch was broken by round 10's import and no gate would have said
    # so, because the module was not in this list.
    ("DA hf/pm window alignment",
     [PY, str(HERE / "da_hf_pm_alignment.py"), "--selftest"]),
    ("DA governed-verdict preflight (read-only)",
     [PY, str(HERE / "da_governed_verdict_preflight.py"), "--selftest"]),
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
    # The runner's OWN twinning contract. `--selftest` returns before the
    # roster is built, so this does not recurse; and R-370 applies to this
    # file as much as to the modules it gates.
    ("gate-runner twinning contract",
     [PY, str(HERE / "v5_deploy_gates.py"), "--selftest"]),
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


#: DA11-R2 / DA12-R1: EVERY `--selftest` GATE IS DERIVED IN BOTH DIRECTIONS,
#: AND EVERY GATE THAT IS NOT IS NAMED.
#:
#: The first version derived `-m` twins from PATH gates only. That covered the
#: break DA10-R3 fixed and left the mirror image uncovered: the one gate
#: already written in `-m` form got no path twin, nothing said so, and it is
#: precisely the gate whose other launcher FAILS. Reproduced here rather than
#: taken on report -- `python3 live/pm_research/tier1_pipeline.py --selftest`
#: exits 1 with `ModuleNotFoundError: No module named 'live'`, because
#: `tier1_pipeline.py:55` imports `from live.pm_research.coverage_ledger`,
#: which a path launch (whose `sys.path[0]` is this directory) cannot resolve;
#: `-m` exits 0.
#:
#: So one gate is EXCLUDED from twinning, and the exclusion is a NAMED STATUS
#: printed with its reason -- the same shape an absent input took when it
#: became a named SKIP. A silent exclusion is the thing this whole mechanism
#: exists to stop: it reads as coverage.
#:
#: RULED (coordinator, this round): tier1_pipeline's path-launchability is a
#: SEPARATE question on its merits. Name the exclusion; do NOT repair it here.
#:
#: Derived, never transcribed: a second roster of module names would go stale
#: the first time a gate is added, which is the failure this closes.
TWIN_EXCLUSIONS: dict[str, str] = {
    "tier1 normalisation": ("already `-m`; the path launch FAILS with "
                            "ModuleNotFoundError: No module named 'live' -- "
                            "tier1_pipeline.py:55 uses a package-absolute "
                            "import. Ruled a separate question this round; "
                            "named, not repaired."),
}


def _launch_twins(gates):
    """(twins, excluded) -- the other launcher for every gate, or the reason.

    PATH gate  -> its `-m` twin.
    `-m` gate  -> its path twin.
    Either     -> excluded ONLY by name, with a reason, in TWIN_EXCLUSIONS.
    """
    twins, excluded = [], []
    for label, argv in gates:
        if label in TWIN_EXCLUSIONS:
            excluded.append((label, TWIN_EXCLUSIONS[label]))
            continue
        is_path = (len(argv) == 3 and argv[0] == PY
                   and str(argv[1]).endswith(".py")
                   and Path(argv[1]).parent == HERE and "--selftest" in argv)
        is_mod = (len(argv) == 4 and argv[0] == PY and argv[1] == "-m"
                  and str(argv[2]).startswith("live.pm_research.")
                  and "--selftest" in argv)
        if is_path:
            mod = Path(argv[1]).stem
            twins.append((f"{label} [-m]",
                          [PY, "-m", f"live.pm_research.{mod}", "--selftest"]))
        elif is_mod:
            mod = str(argv[2]).rsplit(".", 1)[1]
            twins.append((f"{label} [path]",
                          [PY, str(HERE / f"{mod}.py"), "--selftest"]))
        else:
            # Not a `--selftest` module gate at all (a behavioural script, a
            # falsifier runner). It has no other launcher to derive, and
            # saying so is the difference between "covered" and "not asked".
            excluded.append((label, "not a --selftest module gate; no second "
                                    "launcher exists to derive"))
    return twins, excluded


def _selftest() -> int:
    """The twinning contract, driven on a SYNTHETIC roster.

    Both survivors of this round's mutation audit lived here: on the REAL
    roster the only `-m` gate is the excluded one, so NO path twin is ever
    derived and the reverse direction -- the whole point of DA12-R1 -- is
    exercised by nothing. And every real gate lands in `twins` or `excluded`
    by construction, so the accounting equation could not fail on any real
    input. A synthetic roster makes both able to fire.
    """
    n = 0

    def ok(c, label):
        nonlocal n
        n += 1
        if not c:
            print(f"FAIL: {label}")
            raise SystemExit(1)

    roster = [
        ("pathgate", [PY, str(HERE / "pm_tape_density.py"), "--selftest"]),
        ("modgate", [PY, "-m", "live.pm_research.pm_tape_density",
                     "--selftest"]),
        ("tier1 normalisation",
         [PY, "-m", "live.pm_research.tier1_pipeline", "--selftest"]),
        ("behavioural", [PY, str(HERE / "pm_tape_density.py"), "--json"]),
    ]
    twins, excluded = _launch_twins(roster)
    tw = dict(twins)
    ok("pathgate [-m]" in tw and tw["pathgate [-m]"][1] == "-m",
       "a PATH gate derives its `-m` twin")
    ok("modgate [path]" in tw
       and str(tw["modgate [path]"][1]).endswith("pm_tape_density.py"),
       "an `-m` gate derives its PATH twin -- the reverse direction DA12-R1 "
       "asked for, which the real roster exercises nowhere because its only "
       "`-m` gate is the excluded one")
    ex = dict(excluded)
    ok("tier1 normalisation" in ex
       and "package-absolute import" in ex["tier1 normalisation"],
       "a NAMED exclusion is returned with its reason, not silently skipped")
    ok("behavioural" in ex and "no second launcher" in ex["behavioural"],
       "a gate with no second launcher to derive is named too -- 'not asked' "
       "and 'covered' must not look the same")
    ok(len(twins) + len(excluded) == len(roster),
       "every declared gate is either twinned or named as excluded")
    # DELETED, DELIBERATELY, AND THE REASON IS THE POINT (DA14-R1).
    #
    # A line here dropped one twin and asserted the totals no longer balanced.
    # Given the invariant asserted immediately above -- every entry lands in
    # exactly ONE list -- that expression is ARITHMETIC, true under every
    # arrangement, and no change to `_launch_twins` could make it false. It
    # read as a falsifier and tested nothing.
    #
    # WHICH CLOSURE, AND WHY: deleted rather than given a production test hook.
    # The runner's `twins + excluded != len(GATES)` guard documents a
    # STRUCTURAL invariant of `_launch_twins` (each branch appends to exactly
    # one list), and the invariant ITSELF is tested two lines above. Adding a
    # hook to production code so a structurally-impossible branch can be
    # driven would buy a red for a state the function cannot reach, at the
    # cost of surface that exists only for the test. The guard stays as a
    # cheap tripwire on a FUTURE edit to that function -- which is what it is,
    # not a checked behaviour, and this comment is the honest label.
    print(f"v5_deploy_gates selftests: {n} checks passed")
    return 0


def main() -> int:
    if "--selftest" in sys.argv:
        return _selftest()
    falsify = "--falsify" in sys.argv
    twins, excluded = _launch_twins(GATES)
    gates = list(GATES) + twins + ([FAILING_CANARY] if falsify else [])
    # THE EXCLUSIONS ARE PRINTED BEFORE THE RESULTS, so a reader meets them
    # before the row of PASSes rather than inferring them from a count.
    # THE RUNNER READS ITS OWN OUTPUT. An exclusion that is computed but not
    # PRINTED is invisible to the reader, and "we excluded it" then lives only
    # in the code -- so the header is captured and asserted against, rather
    # than printed and hoped for.
    _emitted: list[str] = []

    def emit(line: str) -> None:
        _emitted.append(line)
        print(line)

    named = [(l, r) for l, r in excluded if l in TWIN_EXCLUSIONS]
    if named:
        emit(f"{len(named)} gate(s) EXCLUDED from twinning:")
        for l, r in named:
            emit(f"  - {l}: {r}")
    _no_twin = [l for l, _ in excluded if l not in TWIN_EXCLUSIONS]
    if _no_twin:
        emit(f"{len(_no_twin)} gate(s) have no second launcher to derive "
             f"(not --selftest module gates): {', '.join(_no_twin)}")
    emit(f"roster: {len(GATES)} declared + {len(twins)} derived twins"
         + (" + 1 injected canary" if falsify else "")
         + f" = {len(gates)}")
    _head = "\n".join(_emitted)
    _unprinted = [l for l, _ in named if l not in _head]
    if _unprinted:
        raise SystemExit(
            f"REFUSED: {_unprinted} are excluded from twinning but were not "
            f"PRINTED. An exclusion the reader never sees is indistinguishable "
            f"from coverage -- which is the whole reason exclusions are named.")
    _labels = {l for l, _ in gates}
    _dropped = [l for l, _ in twins if l not in _labels]
    if _dropped:
        raise SystemExit(
            f"REFUSED: {len(_dropped)} derived twin(s) were built and then "
            f"DROPPED from the roster: {_dropped}. A twin that is derived but "
            f"never run is the uninvoked gate this mechanism exists to stop.")
    # A ROSTER THAT SILENTLY SHRINKS IS THE DEFECT. Every declared gate is
    # either twinned or named as excluded -- asserted, so a gate that falls
    # out of both lists fails here instead of reducing the count.
    if len(twins) + len(excluded) != len(GATES):
        raise SystemExit(
            f"REFUSED: {len(twins)} twinned + {len(excluded)} excluded = "
            f"{len(twins) + len(excluded)}, but {len(GATES)} gates are "
            f"declared. A gate in neither list has VANISHED from the "
            f"accounting, which is exactly what this mechanism exists to "
            f"prevent.")
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
