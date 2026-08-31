#!/usr/bin/env python3
"""Mutation audit for v41_boundary_preflight: blank each refusal, prove the
suite notices. Ships its own falsifiers (rule 15 — DA's Q-DA-170 controls,
adopted after DA showed this harness was itself unfalsified):

  A. the UNMUTATED scratch copy must pass — otherwise a copy that fails to
     import for any unrelated reason exits non-zero at every site and the
     harness reports a PERFECT kill rate ("a broken mutation harness and a
     perfectly-covered suite produce identical output");
  B. a CANARY refusal nothing tests must be reported as a SURVIVOR — a zero
     from an instrument that never proved it can fire is not a result. The
     canary is an APPENDED never-called function (an inserted line once made
     an IndentationError read as a pass on DA's side);
  C. an UNPARSEABLE mutant must classify as CRASH, never as an assertion
     kill; kills are split assertion-vs-crash because a crash-kill only
     proves the code cannot proceed, not that the refusal is asserted.
"""
import ast
import pathlib
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parent / "v41_boundary_preflight.py"
SCRATCH_DIR = pathlib.Path("/tmp")
CHECKERS = {"check_boundary_current", "installed_mode_v41",
            "check_candidate_bytes", "require_target_admissible",
            "require_ruled_instant", "check_pre_arm", "make_stamp",
            "make_rollback", "_check_restored_v4", "check_health_identity",
            "check_restart_counter", "check_coin_progress",
            "observe_coin_msgs", "era_state", "make_abort_row",
            "make_recovery_bundle", "_canary_refusal"}
CANARY = ("\n\ndef _canary_refusal():\n"
          "    raise Refused('CANARY: no test reaches this')\n")


def run_suite(text: str) -> tuple:
    # The scratch copy MUST live beside the real module: its REPO/RUNBOOK
    # paths derive from __file__, and a scratchpad copy fails for path
    # reasons — which control A caught on this harness's own first run (the
    # original 53/53 was a FALSE perfect kill rate from exactly that).
    scratch = SRC.parent / "_mut_scratch_v41.py"
    try:
        scratch.write_text(text)
        r = subprocess.run([sys.executable, str(scratch), "--selftest"],
                           capture_output=True, text=True, timeout=120)
    finally:
        scratch.unlink(missing_ok=True)
    if r.returncode == 0:
        return "green", ""
    # Classify on the line that ENDED the run, not on the marker appearing
    # ANYWHERE: a Python traceback echoes the offending source line, so a
    # crash inside a suite whose source contains the marker text would read
    # as an assertion-kill — the flattering direction (DA's classifier
    # defect, found by DA building its own falsifier; adopted here).
    tail = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    last = tail[-1] if tail else ""
    if last.startswith("SELFTEST FAILED"):
        return "assertion", last[:80]
    return "crash", last[:80]


def raise_sites(text: str) -> list:
    out = []
    for fn in ast.walk(ast.parse(text)):
        if isinstance(fn, ast.FunctionDef) and fn.name in CHECKERS:
            for node in ast.walk(fn):
                if isinstance(node, ast.Raise):
                    out.append((fn.name, node.lineno, node.end_lineno))
    return out


def blank(text: str, lo: int, hi: int) -> str:
    lines = text.splitlines()
    indent = len(lines[lo - 1]) - len(lines[lo - 1].lstrip())
    for i in range(lo - 1, hi):
        lines[i] = " " * indent + "pass  # MUTATED"
    return "\n".join(lines)


def audit_scope(src: str) -> None:
    """Control D (audit F6): every refusal-bearing function in the module
    must be IN scope. `check_post_recovery` — the whole --post-recovery
    path, 15 refusals — sat outside CHECKERS, 9 of them untested, and the
    harness reported a clean run while saying nothing about the 18 sites it
    skipped. A scope list that silently omits a checker is the stale-
    mutation-list defect one level up."""
    tree = ast.parse(src)
    defined = {fn.name for fn in ast.walk(tree)
               if isinstance(fn, ast.FunctionDef)
               and any(isinstance(n, ast.Raise) for n in ast.walk(fn))}
    # `ok` is the SELFTEST'S OWN assertion helper, not a gate — auditing it
    # measures the test harness, not the instrument. Excluded by name rather
    # than by adding it to CHECKERS, which would have hidden a real gap
    # behind a permanently-surviving fixture.
    missing = sorted(d for d in defined
                     if d not in CHECKERS and d != "ok"
                     and not d.startswith("observe_")
                     and d not in ("main", "selftest"))
    if missing:
        sys.exit(f"HARNESS INVALID: refusal-bearing function(s) {missing} "
                 f"are NOT in CHECKERS — they would be audited by nothing "
                 f"and the run would still report a clean sweep (control D)")
    total = sum(1 for fn in ast.walk(tree)
                if isinstance(fn, ast.FunctionDef)
                for n in ast.walk(fn) if isinstance(n, ast.Raise))
    covered = len(raise_sites(src))
    print(f"control D: scope complete — {covered} of {total} module raise "
          f"sites in scope (the rest are observers/CLI)")


def main() -> int:
    src = SRC.read_text()
    audit_scope(src)

    # Control A — the harness can tell a working copy from a broken one.
    verdict, _ = run_suite(src)
    if verdict != "green":
        sys.exit("HARNESS INVALID: the UNMUTATED copy does not pass — every "
                 "kill this harness reports would be meaningless (control A)")
    print("control A: unmutated copy passes")

    # Control C — an unparseable mutant classifies as crash, never assertion.
    verdict, _ = run_suite(src + "\ndef broken(:\n")
    if verdict != "crash":
        sys.exit(f"HARNESS INVALID: a SyntaxError mutant classified as "
                 f"{verdict!r}, not crash (control C)")
    print("control C: unparseable mutant classifies as crash")

    # Control B — the harness DEMONSTRABLY reports a survivor.
    canary_src = src + CANARY
    sites = raise_sites(canary_src)
    canary_sites = [x for x in sites if x[0] == "_canary_refusal"]
    if len(canary_sites) != 1:
        sys.exit("HARNESS INVALID: canary not found in the walk (control B)")
    lo, hi = canary_sites[0][1], canary_sites[0][2]
    verdict, _ = run_suite(blank(canary_src, lo, hi))
    if verdict != "green":
        sys.exit(f"HARNESS INVALID: blanking the untested CANARY gave "
                 f"{verdict!r}, not a green suite — the harness cannot "
                 f"report a survivor (control B)")
    print("control B: the canary IS reported as a survivor — the harness "
          "can fire")

    # The audit proper, on the real source (no canary).
    sites = [x for x in raise_sites(src) if x[0] != "_canary_refusal"]
    print(f"{len(sites)} refusal sites in {len(CHECKERS) - 1} checkers")
    survivors, crash_kills, assertion_kills = [], [], []
    for name, lo, hi in sites:
        verdict, tail = run_suite(blank(src, lo, hi))
        if verdict == "green":
            survivors.append((name, lo))
            print(f"  SURVIVOR: {name}:{lo}")
        elif verdict == "crash":
            crash_kills.append((name, lo))
        else:
            assertion_kills.append((name, lo))
    print(f"assertion-killed {len(assertion_kills)} | crash-killed "
          f"{len(crash_kills)} | survivors {len(survivors)}")
    if crash_kills:
        print(f"  crash-kills (loud but UNASSERTED — a later defensive "
              f"check elsewhere would silently delete this coverage): "
              f"{crash_kills}")
    # ACCEPTED SURVIVORS — each needs a PROVEN reason, not a note. Blanking
    # make_recovery_bundle's "era in force" guard leaves the same input
    # REFUSED by the chain validation of the assembled row ("transitioned row
    # 3 claims supersedes='clob_v4' while the era in force is 'clob_v6'"),
    # executed. The guard is kept anyway because it refuses EARLIER and names
    # the era mismatch directly, which is the better operator message at a
    # boundary; it is redundant for safety, not for legibility.
    accepted = {("make_recovery_bundle", 658)}
    real = [x for x in survivors if x not in accepted]
    if survivors and not real:
        print(f"SURVIVORS: {survivors} — ALL ACCEPTED with proven "
              f"downstream coverage; see ACCEPTED SURVIVORS in this file")
        return 0
    if real:
        print(f"SURVIVORS: {real}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
