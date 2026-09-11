"""VALUE ONE FORWARD DAY: both arms, the null, and the four fields.

AN ENTRY POINT, NOT AN ESTIMAND. Every number here is computed by
`de_settlement_control_run.run_one_day_arm` and `de_forward_evaluator`;
this file parses arguments, calls them, and writes what they return. It
contains no statistic, no threshold and no decision. That distinction is
load-bearing: the pipeline commit is fixed at 7ed5a90 and THIS FILE IS NOT
IN IT -- there was no committed driver for a valuation at all, which is
itself a finding (step 2's draws were driven ad hoc, and rule 12 records
that a scratch-dir builder once voided a freeze).

So the receipt records BOTH: this driver's own commit, AND the digest of
every module that computes, verified against 7ed5a90. A reader can then
see that the arithmetic came from the declared commit even though the
`python -m` target did not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import de_forward_evaluator as E          # noqa: E402
import de_settlement_control_run as SC    # noqa: E402

PROTOCOL = "P003_DE_FORWARD_VALUE_DAY_V1"
PIPELINE_COMMIT = "7ed5a9015f75de64feeeeaad21d97e4eecc2b15c"
COMPUTING_MODULES = ("de_settlement_control_run.py",
                     "de_settlement_control_aggregate.py",
                     "de_forward_evaluator.py",
                     "de_asymmetry_null_run.py",
                     "de_matched_cancel_control.py",
                     "de_multiday_gate1_runner.py")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def computing_module_provenance() -> dict:
    """Each computing module's digest here AND at the pipeline commit."""
    out, all_match = {}, True
    for name in COMPUTING_MODULES:
        here = _sha(HERE / name)
        r = subprocess.run(["git", "-C", str(HERE),
                            "show", f"{PIPELINE_COMMIT}:live/pm_research/{name}"],
                           capture_output=True)
        there = (hashlib.sha256(r.stdout).hexdigest()
                 if r.returncode == 0 else None)
        match = (here == there)
        all_match &= match
        out[name] = {"digest_here": here[:16],
                     "digest_at_pipeline_commit":
                         (there[:16] if there else None),
                     "identical": match}
    return {"pipeline_commit": PIPELINE_COMMIT, "modules": out,
            "every_computing_module_matches_the_pipeline_commit": all_match}


def pipeline_provenance_limit() -> dict:
    """DA refuses a result that does not state this. It is not decoration."""
    return {
        "LIMIT": ("THE ENTIRE FORWARD TEST -- BUILD AND VALUATION -- RUNS "
                  "ON CODE THE DEVELOPMENT SCREEN NEVER RAN ON."),
        "what_moved": ["de_head_scoring.py", "de_phase4_diag_runner.py"],
        "what_did_not": ("the other eleven pinned scoring modules are "
                         "byte-identical to what the screen ran"),
        "score_neutrality": ("NOT ESTABLISHED. BE's certification (09-03 on "
                             "both commits) is the instrument that can move "
                             "it; one day with zero flips is NECESSARY, NOT "
                             "SUFFICIENT (REV 173)."),
        "the_waiver": "UNSUPPORTED, NOT REFUTED (R-885)",
        "what_did_NOT_change": ("no parameter, threshold, theta, latency, "
                                "protection mode, repost model, null "
                                "construction or decision rule. The freeze "
                                "moved; the arms did not."),
        "this_driver_is_not_at_the_pipeline_commit": (
            "there was no committed valuation driver at all. It contains no "
            "estimand; see computing_module_provenance for the digests that "
            "do the arithmetic."),
    }


def power_block(emit: dict) -> dict:
    """REV 175: nothing REFUSES a result that drops these, so they are here
    by care rather than by construction. DO NOT let an edit remove them."""
    fb = emit["floor_at_the_G_DECLARED"]
    return {
        "attainable_minimum_p_at_the_G_declared":
            fb["attainable_minimum_p"]["day_sign_component"],
        "attainable_minimum_p_at_the_G_achieved_so_far":
            emit["floor_at_the_G_ACHIEVED_SO_FAR"]
                ["attainable_minimum_p"]["day_sign_component"],
        "tolerance_negative_days_at_the_G_declared":
            fb["tolerance_negative_days"],
        "a_pass_was_possible_at_the_G_declared":
            fb["a_pass_was_possible_at_this_G"],
        "THE_SENTENCE_THESE_TWO_NUMBERS_SUPPORT": (
            "NOT_ESTABLISHED_AT_THIS_POWER, never NO_EFFECT. A test that "
            "cannot attain its own threshold has measured nothing about the "
            "arms -- it has measured the calendar."),
        "REV_175": ("the power fields are on the failure path and driven, "
                    "but NOTHING REFUSES a result that drops them. Until "
                    "that refusal is wired they are present by care."),
    }


def margin_block(emit: dict) -> dict:
    """DE 235: the only fragile comparison here is the SIGN OF D."""
    out = {}
    for arm, v in emit["futility"].items():
        per_day = {d: abs(x) for d, x in
                   v.get("per_day_D_cents", {}).items()} or None
        out[arm] = {"days": per_day}
    return {"per_arm": out,
            "why": ("futility decides on `D <= 0`, an EXACT comparison with "
                    "no tolerance. Its fragility is |D| to ZERO. A margin "
                    "measured on other days does not transfer to this one, "
                    "so it is reported per day."),
            "reference_delta_cents": 1.8e-08,
            "reference_delta_source": "REV's REL_BAR finding"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    ap.add_argument("--book", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-draws", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--days-scored", required=True,
                    help="comma-separated, the days valued SO FAR")
    ap.add_argument("--n-declared", type=int, required=True)
    ap.add_argument("--derived", default=None)
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cells = {}
    for arm in E.ARMS:
        print(json.dumps({"stage": "valuing", "day": a.day, "arm": arm}),
              flush=True)
        res = SC.run_one_day_arm(a.day, a.book, arm, n_draws=a.n_draws,
                                 seed=a.seed, out_dir=out_dir)
        c = a.day.replace("-", "")
        (out_dir / f"de_settle_result_{c}_{arm}.json").write_text(
            json.dumps(res, indent=1, default=str))
        cells[arm] = {"D": res["observed_D_cents"],
                      "p": res["p_two_sided"], "n": res["n_draws"],
                      "resumed_from_draw": res["resumed_from_draw"]}
        print(json.dumps({"stage": "valued", "arm": arm,
                          "D": res["observed_D_cents"],
                          "p": res["p_two_sided"]}), flush=True)

    days = [d for d in a.days_scored.split(",") if d]
    emit = E.progress_emit(out_dir, days, n_declared=a.n_declared,
                           derived=Path(a.derived) if a.derived else None)
    record = {"protocol": PROTOCOL, "day": a.day, "book": a.book,
              "book_sha256": _sha(Path(a.book)),
              "cells": cells, "emit": emit,
              "POWER": power_block(emit),
              "MARGIN_OF_D_TO_ZERO": margin_block(emit),
              "PIPELINE_PROVENANCE_LIMIT": pipeline_provenance_limit(),
              "computing_module_provenance": computing_module_provenance(),
              "elapsed_s": round(time.time() - t0, 1)}
    dst = out_dir / f"p003_de_forward_value_{a.day.replace('-', '')}.json"
    dst.write_text(json.dumps(record, indent=1, default=str))
    for line in emit["per_day_lines"]:
        print("  " + line, flush=True)
    print(json.dumps({"stage": "emitted", "path": str(dst),
                      "ANY_ARM_ALREADY_DEAD": emit["ANY_ARM_ALREADY_DEAD"],
                      "STOP_ADVICE": emit["STOP_ADVICE"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
