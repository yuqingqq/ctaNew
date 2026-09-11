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
import de_multiday_gate1_runner as R      # noqa: E402
import de_settlement_control_run as SC    # noqa: E402

PROTOCOL = "P003_DE_FORWARD_VALUE_DAY_V1"
PIPELINE_COMMIT = "7efea16b39b89c2ddececc90b68e2f206d6c3500"
WRONG_TREE = "VALUATION_COMPUTING_MODULES_ARE_NOT_AT_THE_PIPELINE_COMMIT"
RELOCATED = "VALUATION_RAN_FROM_A_DIFFERENT_TREE_THAN_ITS_LAUNCHER_SELECTED"
NO_PREFLIGHT = "VALUATION_RECORD_CARRIES_NO_PREFLIGHT_SO_NOTHING_WAS_CHECKED"
COMPUTING_MODULES = ("de_settlement_control_run.py",
                     "de_settlement_control_aggregate.py",
                     "de_forward_evaluator.py",
                     "de_asymmetry_null_run.py",
                     "de_matched_cancel_control.py",
                     "de_multiday_gate1_runner.py")


class ValuationRefused(RuntimeError):
    """A named refusal."""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assert_computing_modules_at_the_pipeline_commit(modules=None) -> dict:
    """THE PRE-FLIGHT NOTHING ELSE PERFORMS. Runs AT IMPORT, not in main().

    `be_rule22.assert_unchanged` catches a tree HEAD MOVING under a run.
    NOTHING catches a run STARTING from the wrong tree -- and for a
    valuation that is the worse case, because the BOOK's provenance
    verifies fine while the INSTRUMENT READING IT is wrong, so every other
    check passes and the number looks plausible.

    The tree is derived from the RESOLVED `__file__` of the modules that
    actually compute -- never from the working directory, never from the
    command line. A path in a command line is a LABEL; `git rev-parse` in
    the directory the import actually resolved to is the FACT.
    """
    mods = modules if modules is not None else [
        sys.modules[n] for n in
        ("de_settlement_control_run", "de_settlement_control_aggregate",
         "de_forward_evaluator", "de_asymmetry_null_run",
         "de_matched_cancel_control", "de_multiday_gate1_runner")
        if n in sys.modules]
    if not mods:
        raise ValuationRefused(
            f"REFUSED {WRONG_TREE}: no computing module is imported, so "
            f"there is nothing to check. An unchecked valuation is the "
            f"failure this exists to prevent, so absence REFUSES.")
    seen, trees = {}, set()
    for m in mods:
        f = Path(m.__file__).resolve()
        trees.add(str(f.parents[2]))
        seen[f.name] = f
    if len(trees) != 1:
        raise ValuationRefused(
            f"REFUSED {WRONG_TREE}: the computing modules resolved from "
            f"MORE THAN ONE TREE {sorted(trees)}. A valuation assembled "
            f"from two trees has no single provenance.")
    tree = trees.pop()
    head = subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    # A COMMIT CANNOT CONTAIN ITS OWN HASH. PIPELINE_COMMIT names the
    # commit the COMPUTING MODULES are pinned at; the tree may be a
    # DESCENDANT of it (this driver's own landing is one), and that is
    # admissible ONLY when every computing module is byte-identical to the
    # pin -- which the digest loop below proves. A descendant that moved a
    # computing module is refused exactly as a stranger would be.
    descendant = head != PIPELINE_COMMIT and subprocess.run(
        ["git", "-C", tree, "merge-base", "--is-ancestor",
         PIPELINE_COMMIT, head], capture_output=True).returncode == 0
    if head != PIPELINE_COMMIT and not descendant:
        raise ValuationRefused(
            f"REFUSED {WRONG_TREE}: the computing modules resolved from "
            f"{tree}, whose HEAD is {head[:12] or 'NONE'}, and the ruled "
            f"pipeline commit is {PIPELINE_COMMIT[:12]}. A valuation from "
            f"the wrong tree produces a number from the wrong instrument "
            f"and the book's provenance still verifies -- so nothing else "
            f"would catch it.")
    bad = []
    for name, f in seen.items():
        r = subprocess.run(["git", "-C", tree, "show",
                            f"{PIPELINE_COMMIT}:live/pm_research/{name}"],
                           capture_output=True)
        if r.returncode != 0 or hashlib.sha256(r.stdout).hexdigest() != _sha(f):
            bad.append(name)
    if bad:
        raise ValuationRefused(
            f"REFUSED {WRONG_TREE}: right tree, WRONG BYTES in {bad}. The "
            f"HEAD check and the digest check catch different faults and "
            f"neither substitutes for the other.")
    want = __import__("os").environ.get("DE_VALUATION_EXPECTED_TREE")
    if want and str(Path(want).resolve()) != tree:
        raise ValuationRefused(
            f"REFUSED {RELOCATED}: the launcher selected {want} and the "
            f"computing modules actually resolved from {tree}. "
            f"`be_heavy_run.sh` hardcodes BE_WORKTREE and cds to it, which "
            f"silently relocated the 09-07 valuation and left NO ERROR -- "
            f"only a missing field in the record. Prevention can fail; "
            f"this is the detection.")
    return {"tree": tree, "head": head, "n_modules_checked": len(seen),
            "launcher_expected_tree": want,
            "resolved_files": {k: str(v) for k, v in seen.items()}}


def assert_record_carries_preflight(path) -> dict:
    """A RECORD WITHOUT A PRE-FLIGHT REFUSES -- absence is not a pass.

    On 09-07 the pre-flight never ran, and the ONLY trace was the ABSENCE
    of `PREFLIGHT_RESOLVED_TREE` from the emitted record. Absence read
    exactly like a passing check, which is this session's dominant failure
    mode arriving inside our own guard. So absence is now a named refusal
    that any reader -- DA, REV, a later me -- can drive over any record."""
    rec = json.loads(Path(path).read_text())
    pf = rec.get("PREFLIGHT_RESOLVED_TREE")
    if not pf:
        raise ValuationRefused(
            f"REFUSED {NO_PREFLIGHT}: {Path(path).name} carries no "
            f"PREFLIGHT_RESOLVED_TREE. Nothing established which tree "
            f"computed these numbers. This is NOT a pass -- it is the "
            f"absence of the check, which is indistinguishable from one "
            f"unless it refuses.")
    if pf.get("head") != PIPELINE_COMMIT:
        raise ValuationRefused(
            f"REFUSED {WRONG_TREE}: the record's pre-flight names head "
            f"{str(pf.get('head'))[:12]}, not the pipeline commit.")
    return pf


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


def runner_provenance() -> dict:
    """THE RUNNER'S OWN DIGEST AND ITS IMPORT CLOSURE (rule 22).

    Today `be_module` records be_cancel_axis_null's digest -- not the
    runner's -- so a result cannot prove WHICH RUNNER produced it. Day one
    could only be shown clean by REV reconstructing from the launch record
    and the reflog; a result must prove itself from its own artifact.

    THE HEADER-SIDE IS NOT DONE HERE AND I SAY SO: the checkpoint HEADER is
    written inside `run_one_day_arm`, which is now PINNED at the valuation
    commit. Adding a field there would move the pin the freeze just set.
    So the closure is captured BEFORE and AFTER each arm instead, and a
    mid-run rewrite shows up as a difference between the two -- the same
    detectability, without editing a frozen module.
    """
    out = {}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or "/pm_research/" not in str(f):
            continue
        fp = Path(f)
        if fp.is_file():
            out[fp.name] = _sha(fp)[:16]
    return {"runner": out.get("de_settlement_control_run.py"),
            "driver": out.get("de_forward_value_day.py"),
            "n_modules_in_closure": len(out), "closure": out}


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
    preflight = assert_computing_modules_at_the_pipeline_commit()
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
    ap.add_argument("--book-receipt", required=True)
    ap.add_argument("--score-certification", action="append", required=True)
    # PARAMS_REL is a LITERAL inside the PINNED runner, so the version
    # cannot be selected there without moving the pin. load_params(path)
    # is the declared override and this driver is not pinned, so the
    # selection changes HERE and nowhere else.
    ap.add_argument("--params", default=None)
    a = ap.parse_args(argv)

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cells = {}
    for arm in E.ARMS:
        print(json.dumps({"stage": "valuing", "day": a.day, "arm": arm}),
              flush=True)
        before = runner_provenance()
        # V2 DERIVES the seed from (book digest, arm) and REFUSES a
        # supplied one that differs -- so None is passed deliberately, not
        # by omission. book_receipt and score_certifications are required.
        res = SC.run_one_day_arm(a.day, a.book, arm, n_draws=a.n_draws,
                                 seed=None, out_dir=out_dir,
                                 book_receipt=a.book_receipt,
                                 score_certifications=a.score_certification,
                                 params=(R.load_params(Path(a.params))
                                         if a.params else None))
        after = runner_provenance()
        moved = sorted(k for k in set(before["closure"]) | set(after["closure"])
                       if before["closure"].get(k) != after["closure"].get(k))
        res["runner_provenance"] = {
            "before": before, "after": after,
            "modules_that_changed_under_the_run": moved,
            "A_MODULE_CHANGED_UNDER_THIS_RUN": bool(moved),
            "why_before_and_after": (
                "the checkpoint HEADER is written inside a PINNED module; "
                "capturing the closure either side of the call detects a "
                "mid-run rewrite without editing frozen bytes")}
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
              "runner_provenance": runner_provenance(),
              "params_file": a.params,
              "PREFLIGHT_RESOLVED_TREE": preflight,
              "elapsed_s": round(time.time() - t0, 1)}
    dst = out_dir / f"p003_de_forward_value_{a.day.replace('-', '')}.json"
    dst.write_text(json.dumps(record, indent=1, default=str))
    for line in emit["per_day_lines"]:
        print("  " + line, flush=True)
    print(json.dumps({"stage": "emitted", "path": str(dst),
                      "ANY_ARM_ALREADY_DEAD": emit["ANY_ARM_ALREADY_DEAD"],
                      "STOP_ADVICE": emit["STOP_ADVICE"]}), flush=True)
    return 0


# AT IMPORT. A different invocation -- `python -c`, a notebook, another
# driver -- cannot skip a module-level check the way it can skip main().
# THE RESIDUAL IS NAMED, NOT HIDDEN: a caller that imports
# `de_settlement_control_run` DIRECTLY, without this module, still
# bypasses it. The bypass-proof placement is inside the pinned computing
# modules -- which would change their bytes and so break the very pin this
# enforces. It lands with the refusal renames, after the last forward book.
if not __import__("os").environ.get("DE_VALUATION_PREFLIGHT_OFF"):
    _PREFLIGHT = assert_computing_modules_at_the_pipeline_commit()


if __name__ == "__main__":
    raise SystemExit(main())
