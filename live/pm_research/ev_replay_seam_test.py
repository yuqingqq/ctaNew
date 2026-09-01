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
REPO = HERE.parents[1]


def _launch(script: Path, args: list[str], cwd: Path,
            env: dict | None = None) -> subprocess.CompletedProcess:
    """A launcher runs a path with argv from some working directory. It does
    not import the module, and it does not add the module's directory to
    sys.path for it -- so this reproduces that exactly."""
    return subprocess.run([sys.executable, str(script)] + args,
                          capture_output=True, text=True, cwd=str(cwd),
                          env=env or os.environ.copy(), timeout=600)


def _mutant_dir(tmp: Path, name: str, old: str, new: str) -> Path:
    """An on-disk copy with ONE textual change, plus the machine beside it so
    the copy fails for the reason under test and not for a missing import."""
    d = tmp / name
    d.mkdir(parents=True, exist_ok=True)
    src = SEAM.read_text()
    if old not in src:
        raise AssertionError(
            f"mutant {name}: anchor text not found -- the mutation would be a "
            f"no-op and the known-bad would pass vacuously")
    (d / "ev_replay_seam.py").write_text(src.replace(old, new, 1))
    shutil.copy2(MACHINE, d / "harmful_stateful_policy.py")
    return d


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
            '"queue_bound_stamped": bound_stamped,',
            "# MUTANT M1: the gate is never computed")
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
            '            "gates": gates,',
            '            "gates": gates,\n'
            '            "economics": {"net_cents": 1.0},'
            '  # MUTANT M2: leak')
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
    return {
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


EXPECTED_CHECKS = 9


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
