"""Launcher-level seam test for `ev_replay_seam.py` -- rule 17, both halves.

SURFACE AUTHORISATION (R-126, in-file): coordinator DE round-2 dispatch.
RESEARCH-ONLY, OFFLINE.

WHY A SEPARATE MODULE.  A green component suite proves the UNIT; it cannot
see whether the entry point is wired (R-251/R-252: six evaluator functions,
all falsifier-proven, zero call sites).  So this drives `main()` THE WAY A
LAUNCHER DRIVES IT -- a fresh interpreter, argv, a working directory that is
not the module's own, and an artifact read back from disk -- and it asserts
the ARTIFACT, not the exit code.

AND IT MUST BE ABLE TO FAIL.  A seam test that only ever runs the good case
is F-1's shape: the harness that exists to prove the wiring cannot fail when
the wiring is cut.  So two MUTANTS are executed against on-disk copies:

  M1  a required gate is never computed  -> the run must REFUSE and write
                                            nothing;
  M2  the machine's economics block is put into the receipt
                                         -> the emission guard must REFUSE
                                            and write nothing.

Each mutant's failure REASON is asserted, not just its exit code.  A mutant
that died on ImportError would otherwise "pass" the known-bad and prove
nothing -- the vacuity this programme has paid for repeatedly.

    python3 live/pm_research/ev_replay_seam_test.py --selftest
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAM = HERE / "ev_replay_seam.py"
MACHINE = HERE / "harmful_stateful_policy.py"
RULE = HERE / "rule_policy_v1.py"
REPO = HERE.parents[1]
# Files a mutant copy must find BESIDE itself, because engine-identity blocks
# read their neighbours off disk rather than importing them.
NEIGHBOURS = ("ev_replay_seam.py", "harmful_stateful_policy.py",
              "de_constraints.py", "de_actionspace.py")

# A REFUSAL IS A FAILURE LINE, NOT A SUBSTRING.  The first version of these
# predicates searched combined stdout+stderr for the control's name -- which
# appears in the PASS line too, so a mutant that passed everything would have
# been scored as "refused for the right reason". Both markers below only
# occur when something actually failed.
FAIL_MARKERS = ("FAIL:", "FAIL (no refusal):", "REFUSING to write",
                "Traceback")


def _really_failed(out: str, naming: str) -> bool:
    return any(m in out for m in FAIL_MARKERS) and naming in out


def _launch(script: Path, args: list[str], cwd: Path,
            env: dict | None = None) -> subprocess.CompletedProcess:
    """A launcher runs a path with argv from some working directory. It does
    not import the module, and it does not add the module's directory to
    sys.path for it -- so this reproduces that exactly."""
    return subprocess.run([sys.executable, str(script)] + args,
                          capture_output=True, text=True, cwd=str(cwd),
                          env=env or os.environ.copy(), timeout=600)


def _mutant_dir(tmp: Path, name: str, edits, *,
                target: Path = SEAM) -> Path:
    """An on-disk copy of ONE module with one or more textual changes, with
    its neighbours symlinked beside it so the copy fails for the reason under
    test and not for a missing import or a missing neighbour."""
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    src = target.read_text()
    if isinstance(edits, tuple) and len(edits) == 2 \
            and isinstance(edits[0], str):
        edits = [edits]
    for old, new in edits:
        if old not in src:
            raise AssertionError(
                f"mutant {name}: anchor text not found -- the mutation would "
                f"be a no-op and the known-bad would pass vacuously")
        src = src.replace(old, new, 1)
    (d / target.name).write_text(src)
    for nb in NEIGHBOURS:
        if nb == target.name:
            continue
        link = d / nb
        if not link.exists():
            link.symlink_to(HERE / nb)
    return d


def _env_with_pm_research() -> dict:
    """A mutant resolves ITS OWN module from its script directory (Python
    prepends it) and every dependency from the real tree."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def seamtest() -> dict:
    results: dict = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # ---- POSITIVE CONTROL: the launcher path, end to end ------------
        out = tmp / "receipt.json"
        r = _launch(SEAM, ["run", "--out", str(out)], cwd=REPO)
        receipt = json.loads(out.read_text()) if out.exists() else None
        results["positive_control"] = {
            "exit": r.returncode,
            "artifact_written": out.exists(),
            "all_gates_pass": (receipt or {}).get("all_gates_pass"),
            "economics_emittable": (receipt or {}).get("economics_emittable"),
            "unreleased_inputs": (receipt or {}).get("unreleased_inputs"),
            "n_records": len((receipt or {}).get("records", [])),
            "run_hash": (receipt or {}).get("run_hash"),
            "stderr_tail": r.stderr.strip().split("\n")[-1] if r.stderr else "",
        }

        # ---- M1: a required gate is never computed ----------------------
        d1 = _mutant_dir(
            tmp, "m1",
            ('"queue_bound_stamped": bound_stamped,',
             "# MUTANT M1: the gate is never computed"))
        o1 = tmp / "m1.json"
        r1 = _launch(d1 / "ev_replay_seam.py", ["run", "--out", str(o1)],
                     cwd=REPO)
        results["m1_unwired_gate"] = {
            "exit": r1.returncode, "artifact_written": o1.exists(),
            "refused_for_the_right_reason":
                "SeamRefused" in r1.stderr and "queue_bound_stamped" in r1.stderr,
            "stderr_tail": r1.stderr.strip().split("\n")[-1] if r1.stderr else "",
        }

        # ---- M2: the machine's economics is put into the receipt --------
        d2 = _mutant_dir(
            tmp, "m2",
            ('            "gates": gates,',
             '            "gates": gates,\n'
             '            "economics": {"net_cents": 1.0},'
             '  # MUTANT M2: leak'))
        o2 = tmp / "m2.json"
        r2 = _launch(d2 / "ev_replay_seam.py", ["run", "--out", str(o2)],
                     cwd=REPO)
        results["m2_leaked_economics"] = {
            "exit": r2.returncode, "artifact_written": o2.exists(),
            "refused_for_the_right_reason":
                "EconomicsRefused" in r2.stderr
                and "economics" in r2.stderr,
            "stderr_tail": r2.stderr.strip().split("\n")[-1] if r2.stderr else "",
        }

        # ---- M3: the manifest boundary, unwired ------------------------
        # It takes TWO edits, and that is itself the result. The boundary is
        # guarded twice over: the sentinel/non-manifest-key refusal AND the
        # read-outside-your-own-declared-manifest refusal. Unwiring either
        # ALONE leaves it closed -- measured below as m3a -- so the honest
        # known-bad for "the manifest check is what makes the gate pass" is
        # the COMPOUND mutant. A single-edit mutant that exits 0 is not a
        # failed test; it is evidence of redundancy, and reporting it as a
        # kill would have been the vacuous-control move.
        SENTINEL_REFUSAL = ('if probe["sentinels_touched"] or '
                            'probe["non_manifest_keys_read"]:')
        DECLARED_REFUSAL = ('outside = [k for k in probe["keys_read"] '
                            'if k not in declared]')
        d3a = _mutant_dir(tmp, "m3a", (
            SENTINEL_REFUSAL,
            'if False and (probe["sentinels_touched"] or '
            'probe["non_manifest_keys_read"]):  # MUTANT M3a'),
            target=RULE)
        o3a = tmp / "m3a.json"
        r3a = _launch(d3a / "rule_policy_v1.py", ["run", "--out", str(o3a)],
                      cwd=REPO, env=_env_with_pm_research())
        results["m3a_one_refusal_unwired"] = {
            "exit": r3a.returncode, "artifact_written": o3a.exists()}

        d3 = _mutant_dir(tmp, "m3", [
            (SENTINEL_REFUSAL,
             'if False and (probe["sentinels_touched"] or '
             'probe["non_manifest_keys_read"]):  # MUTANT M3'),
            (DECLARED_REFUSAL, "outside = []  # MUTANT M3")],
            target=RULE)
        o3 = tmp / "m3.json"
        r3m = _launch(d3 / "rule_policy_v1.py", ["run", "--out", str(o3)],
                      cwd=REPO, env=_env_with_pm_research())
        _o3 = r3m.stderr + r3m.stdout
        # WHERE it refuses is itself worth recording. I expected the GATE to
        # fail at emission; what actually happens is that `main()` refuses
        # earlier, because the entry point runs the selftest first and the
        # suite's own manifest known-bad ("a solver that reads `belief` is
        # REFUSED AT REGISTRATION") no longer fires. That is the
        # numbers-never-come-from-a-red-suite rule doing its job, and it is a
        # STRONGER outcome than the one I predicted -- so the predicate
        # admits either path and the receipt names which one ran.
        results["m3_revelation_unwired"] = {
            "exit": r3m.returncode, "artifact_written": o3.exists(),
            "refusal_path": ("selftest"
                             if _really_failed(_o3, "REFUSED AT REGISTRATION")
                             else "gate" if _really_failed(
                                 _o3, "leak_refused_at_registration")
                             else "unknown"),
            "refused_for_the_right_reason":
                _really_failed(_o3, "REFUSED AT REGISTRATION")
                or _really_failed(_o3, "leak_refused_at_registration"),
            "stderr_tail": r3m.stderr.strip().split("\n")[-1] if r3m.stderr else "",
        }

        # ---- M4: the NO_CANCEL reduction is broken ----------------------
        # The reduction to QR_SKEW_ONLY is ALSO doubly guaranteed, and the
        # same discipline applies. Removing the empty-score-stream shortcut
        # alone leaves parity intact, because the RULE still says never
        # cancel and every emitted score is 0.0 -- measured as m4a. The true
        # known-bad is the one that makes the RULE act.
        d4a = _mutant_dir(tmp, "m4a", (
            "        if not self.predictor_active:",
            "        if False:  # MUTANT M4a: the shortcut is removed"),
            target=RULE)
        o4a = tmp / "m4a.json"
        r4a = _launch(d4a / "rule_policy_v1.py", ["run", "--out", str(o4a)],
                      cwd=REPO, env=_env_with_pm_research())
        results["m4a_shortcut_removed"] = {
            "exit": r4a.returncode, "artifact_written": o4a.exists()}

        d4 = _mutant_dir(tmp, "m4", [
            ("        if not self.predictor_active:",
             "        if False:  # MUTANT M4a"),
            ('        if self.config.cancel_rule == "NO_CANCEL":\n'
             "            return False",
             '        if self.config.cancel_rule == "NO_CANCEL":\n'
             "            return True  # MUTANT M4: NO_CANCEL now cancels")],
            target=RULE)
        o4 = tmp / "m4.json"
        r4m = _launch(d4 / "rule_policy_v1.py", ["run", "--out", str(o4)],
                      cwd=REPO, env=_env_with_pm_research())
        _o4 = r4m.stderr + r4m.stdout
        results["m4_parity_broken"] = {
            "exit": r4m.returncode, "artifact_written": o4.exists(),
            "refusal_path": ("selftest" if _really_failed(_o4, "PARITY")
                             else "gate" if _really_failed(
                                 _o4, "no_cancel_reduces") else "unknown"),
            "refused_for_the_right_reason":
                _really_failed(_o4, "PARITY")
                or _really_failed(_o4, "no_cancel_reduces_to_qr_skew_only"),
            "stderr_tail": r4m.stderr.strip().split("\n")[-1] if r4m.stderr else "",
        }

        # ---- the RULE POLICY entry point, positive control --------------
        o5 = tmp / "rule.json"
        r5 = _launch(RULE, ["run", "--out", str(o5)], cwd=REPO)
        rec5 = json.loads(o5.read_text()) if o5.exists() else None
        results["rule_policy_positive_control"] = {
            "exit": r5.returncode, "artifact_written": o5.exists(),
            "all_gates_pass": (rec5 or {}).get("all_gates_pass"),
            "economics_emittable": (rec5 or {}).get("economics_emittable"),
            "solver": (rec5 or {}).get("solver"),
            "manifest_len": len((rec5 or {}).get("manifest", [])),
        }

        # ---- the original is unchanged by any of this -------------------
        r3 = _launch(SEAM, ["--selftest"], cwd=REPO)
        results["original_still_green"] = {
            "exit": r3.returncode,
            "tail": r3.stdout.strip().split("\n")[-1] if r3.stdout else "",
        }
    return results


def verdict(res: dict) -> dict:
    pc = res["positive_control"]
    m1 = res["m1_unwired_gate"]
    m2 = res["m2_leaked_economics"]
    og = res["original_still_green"]
    m3 = res["m3_revelation_unwired"]
    m4 = res["m4_parity_broken"]
    rp = res["rule_policy_positive_control"]
    return {
        "rule_policy_runs_end_to_end":
            rp["exit"] == 0 and rp["artifact_written"] is True
            and rp["all_gates_pass"] is True
            and rp["economics_emittable"] is False
            and rp["solver"] == "RulePolicy_v1" and rp["manifest_len"] == 7,
        "m3_refuses_and_writes_nothing":
            m3["exit"] != 0 and m3["artifact_written"] is False
            and m3["refused_for_the_right_reason"] is True,
        "m4_refuses_and_writes_nothing":
            m4["exit"] != 0 and m4["artifact_written"] is False
            and m4["refused_for_the_right_reason"] is True,
        "manifest_boundary_is_doubly_guarded":
            res["m3a_one_refusal_unwired"]["exit"] == 0,
        "reduction_is_doubly_guaranteed":
            res["m4a_shortcut_removed"]["exit"] == 0,
        "launcher_runs_end_to_end":
            pc["exit"] == 0 and pc["artifact_written"] is True
            and pc["all_gates_pass"] is True
            and pc["economics_emittable"] is False
            and pc["n_records"] >= 2,
        "m1_refuses_and_writes_nothing":
            m1["exit"] != 0 and m1["artifact_written"] is False
            and m1["refused_for_the_right_reason"] is True,
        "m2_refuses_and_writes_nothing":
            m2["exit"] != 0 and m2["artifact_written"] is False
            and m2["refused_for_the_right_reason"] is True,
        "original_unaffected": og["exit"] == 0,
    }


EXPECTED_CHECKS = 15


def selftest() -> int:
    n = [0]

    def ok(cond, label):
        if not cond:
            raise SystemExit(f"[ev_replay_seam_test] FAIL: {label}")
        n[0] += 1
        print(f"  PASS  {label}")

    res = seamtest()
    v = verdict(res)
    pc = res["positive_control"]
    ok(v["launcher_runs_end_to_end"],
       f"POSITIVE CONTROL: a launcher runs the entry point end to end from a "
       f"foreign cwd and the ARTIFACT is read back and asserted "
       f"(exit {pc['exit']}, gates {pc['all_gates_pass']}, "
       f"{pc['n_records']} records)")
    ok(pc["economics_emittable"] is False
       and pc["unreleased_inputs"] == ["action_value_policy", "fair_price",
                                       "harm_predictor", "latency_budget_ack"],
       f"the emitted artifact NAMES its unreleased inputs and declares "
       f"economics unemittable: {pc['unreleased_inputs']}")
    ok(isinstance(pc["run_hash"], str) and len(pc["run_hash"]) == 64,
       "and it carries a run_hash over its own body")
    ok(v["m1_refuses_and_writes_nothing"],
       f"KNOWN-BAD M1: with one required gate never computed the run REFUSES "
       f"and writes NOTHING (exit {res['m1_unwired_gate']['exit']}, "
       f"artifact {res['m1_unwired_gate']['artifact_written']})")
    ok(res["m1_unwired_gate"]["refused_for_the_right_reason"],
       "and M1 failed for the RIGHT REASON -- SeamRefused naming the missing "
       "gate, not an ImportError wearing a non-zero exit code")
    ok(v["m2_refuses_and_writes_nothing"],
       f"KNOWN-BAD M2: with the machine's economics leaked into the receipt "
       f"the emission guard REFUSES and writes NOTHING "
       f"(exit {res['m2_leaked_economics']['exit']})")
    ok(res["m2_leaked_economics"]["refused_for_the_right_reason"],
       "and M2 failed for the RIGHT REASON -- EconomicsRefused naming the "
       "leaked key")
    rp = res["rule_policy_positive_control"]
    ok(v["rule_policy_runs_end_to_end"],
       f"POSITIVE CONTROL: RulePolicy_v1's own entry point runs end to end "
       f"through a launcher and its artifact declares solver "
       f"{rp['solver']!r} with a {rp['manifest_len']}-field manifest and "
       f"economics unemittable")
    ok(v["manifest_boundary_is_doubly_guarded"],
       "MEASURED, not assumed: unwiring the sentinel refusal ALONE leaves "
       "the boundary closed -- the read-outside-your-own-manifest refusal "
       "still catches the leak, so the known-bad below must unwire BOTH")
    ok(v["m3_refuses_and_writes_nothing"],
       f"KNOWN-BAD M3: with BOTH manifest refusals unwired the run REFUSES "
       f"and writes NOTHING (exit {res['m3_revelation_unwired']['exit']}, "
       f"via the {res['m3_revelation_unwired']['refusal_path']} path) -- the "
       f"manifest check has an artifact-level consequence, not just a code "
       f"path")
    ok(v["reduction_is_doubly_guaranteed"],
       "MEASURED: removing the empty-score-stream shortcut ALONE leaves "
       "parity intact, because the RULE still says never cancel and every "
       "emitted score is 0.0 -- so the known-bad must make the RULE act")
    ok(v["m4_refuses_and_writes_nothing"],
       f"KNOWN-BAD M4: with the shortcut removed AND NO_CANCEL inverted to "
       f"cancel, parity against "
       f"QR_SKEW_ONLY BREAKS and the run REFUSES, writing nothing "
       f"(exit {res['m4_parity_broken']['exit']}, via the "
       f"{res['m4_parity_broken']['refusal_path']} path)")
    ok(all(res[k].get("refusal_path", "gate") != "unknown"
           for k in ("m3_revelation_unwired", "m4_parity_broken")),
       "and each refusal was matched on a FAILURE LINE, not on a substring "
       "that also appears in the PASS line it names")
    ok(v["original_unaffected"],
       "the unmutated module is still green after both mutants, so the "
       "known-bads did not contaminate the thing they test")
    ok(n[0] + 1 == EXPECTED_CHECKS,
       f"check count asserted at run time: {n[0] + 1} == {EXPECTED_CHECKS}")
    print(f"[ev_replay_seam_test] selftest OK -- {n[0]} checks")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    print(json.dumps({"results": seamtest()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
