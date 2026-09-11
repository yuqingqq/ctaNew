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
def _frozen_commit() -> str:
    """The frozen commit, READ FROM DA's declaration by declared identity.

    A literal here froze the valuation against a commit it no longer is:
    V2 compared its own seven modules to 7ed5a90 and refused ITSELF. The
    commit is a declaration, resolved through the freeze chain like every
    other identity, so the freeze can move without a code edit.
    """
    import de_multiday_gate1_runner as _R
    d = HERE / "declarations"
    pin = (_R.resolve_declaration_pins(d) or {}).get("code_freeze_declaration")
    f = d / (Path(str(pin["path"])).name if pin else
             "da_code_freeze_declaration_v1.json")
    if pin and pin.get("sha256") and f.is_file():
        if hashlib.sha256(f.read_bytes()).hexdigest() != pin["sha256"]:
            raise ValuationRefused(
                f"REFUSED {WRONG_TREE}: {f.name} is not the declared "
                f"identity {str(pin['sha256'])[:16]}.")
    if not f.is_file():
        raise ValuationRefused(
            f"REFUSED {WRONG_TREE}: no code-freeze declaration at {f}.")
    # DA's declarations carry commits as "<sha> -- <prose>" (BUILD_PIN does
    # too). Take the sha, not the sentence: passing the whole string to git
    # made every module read ABSENT and the self-check blamed six modules
    # for one parsing mistake.
    return str(json.loads(f.read_text())["FREEZE_COMMIT"]).split()[0]


# NO LITERAL AND NO SENTINEL. The frozen commit is READ from DA's code
# freeze declaration by declared identity (`_frozen_commit()`), at every
# use site. A module-level sentinel was worse than the literal it replaced:
# it read like a value, so two sites kept interpolating it -- one into
# `git show`, crashing the record build with a NameError's cousin, one into
# `!=`, refusing every valid record. A name that cannot hold the answer
# must not exist.
WRONG_TREE = "VALUATION_COMPUTING_MODULES_ARE_NOT_AT_THE_PIPELINE_COMMIT"
NOT_DESCENDANT = "VALUATION_TREE_IS_NOT_A_DESCENDANT_OF_THE_FROZEN_COMMIT"
ROW_MOVED = "VALUATION_CLOSURE_DIGEST_MOVED"
ROW_ABSENT = "VALUATION_CLOSURE_MODULE_ABSENT"
# A MODULE CANNOT VOUCH FOR ITSELF (USER RULING, DE 331). V2's digest is
# RECORDED in the result and verified OUTSIDE, by the launcher's matrix row
# against the same declaration -- the way the certificate vouches for the
# comparator. Everything else in the declaration's rows is asserted here.
SELF = "de_settlement_control_run.py"
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
    # A COMMIT CANNOT CONTAIN ITS OWN HASH. The frozen commit names the
    # commit the COMPUTING MODULES are pinned at; the tree may be a
    # DESCENDANT of it (this driver's own landing is one), and that is
    # admissible ONLY when every computing module is byte-identical to the
    # pin -- which the digest loop below proves. A descendant that moved a
    # computing module is refused exactly as a stranger would be.
    # THE BUILD PIN'S FORM, NOT TIP EQUALITY (USER RULING, DE 331). A
    # declaration naming a tip can never name the commit that contains it:
    # DA's re-declaration lands ON TOP of the freeze, so the tip is always
    # one past it, and a tip-equality check refuses the correct tree
    # forever. Provenance is ANCESTRY; identity is DIGESTS.
    frozen = _frozen_commit()
    rows = _declared_closure_digests()
    if head != frozen and subprocess.run(
            ["git", "-C", tree, "merge-base", "--is-ancestor", frozen, head],
            capture_output=True).returncode != 0:
        raise ValuationRefused(
            f"REFUSED {NOT_DESCENDANT}: the computing modules resolved "
            f"from {tree}, whose HEAD is {head[:12] or 'NONE'}, which does "
            f"not descend from the declared freeze commit {frozen[:12]}. A "
            f"valuation from an unrelated tree produces a number from the "
            f"wrong instrument while the book's provenance still verifies "
            f"-- so nothing else would catch it.")
    moved, absent = [], []
    for name, want_sha in sorted(rows.items()):
        if name == SELF:
            continue
        f = Path(tree) / "live" / "pm_research" / name
        if not f.is_file():
            absent.append(name)
        elif _sha(f) != want_sha:
            moved.append(name)
    if absent:
        raise ValuationRefused(
            f"REFUSED {ROW_ABSENT}: the declaration names {absent}, which "
            f"the tree does not carry. A closure the declaration names and "
            f"the tree lacks is not a closure.")
    if moved:
        raise ValuationRefused(
            f"REFUSED {ROW_MOVED}: right tree, WRONG BYTES in {moved}, "
            f"against the digests declared at {frozen[:12]}. Ancestry and "
            f"digests catch different faults and neither substitutes for "
            f"the other.")
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
    # DESCENDANT, SAME RULE AS THE PRE-FLIGHT ITSELF. The tree that
    # computes may descend from the frozen commit; the pre-flight admits
    # that only when every computing module is byte-identical to the pin.
    # A reader holding only the record must apply the SAME rule from the
    # record's own evidence -- not a stricter one, or every valid record
    # refuses, and not a looser one.
    frozen = _frozen_commit()
    head = str(pf.get("head") or "")
    if head != frozen:
        prov = rec.get("computing_module_provenance") or {}
        if not (prov.get("pipeline_commit") == frozen
                and prov.get(
                    "every_computing_module_matches_the_pipeline_commit")):
            raise ValuationRefused(
                f"REFUSED {WRONG_TREE}: the record's pre-flight names head "
                f"{head[:12]}, which is not the frozen commit "
                f"{frozen[:12]}, and the record does not show every "
                f"computing module identical to it.")
    return pf


def _declared_closure_digests() -> dict:
    """The frozen digest ROWS, from the same declaration as the commit."""
    import de_multiday_gate1_runner as _R
    d = HERE / "declarations"
    pin = (_R.resolve_declaration_pins(d) or {}).get("code_freeze_declaration")
    f = d / (Path(str(pin["path"])).name if pin
             else "da_code_freeze_declaration_v1.json")
    if not f.is_file():
        raise ValuationRefused(
            f"REFUSED {ROW_ABSENT}: the code freeze declaration is absent "
            f"at {f}. Without it there are no rows to check against.")
    rows = json.loads(f.read_text()).get(
        "VALUATION_CLOSURE_DIGESTS_AT_THE_FREEZE") or {}
    if not rows:
        raise ValuationRefused(
            f"REFUSED {ROW_ABSENT}: {f.name} carries no "
            f"VALUATION_CLOSURE_DIGESTS_AT_THE_FREEZE. An empty row set "
            f"would pass every module, which is the absence of the check.")
    return {k: v for k, v in rows.items() if isinstance(v, str)
            and len(v) == 64}


def computing_module_provenance() -> dict:
    """Each module's digest here AND as the declaration froze it.

    V2's own row is RECORDED, never asserted (rule: a module cannot vouch
    for itself). The launcher's matrix carries it as an external row.
    """
    frozen = _frozen_commit()
    rows = _declared_closure_digests()
    out, all_match = {}, True
    for name, want_sha in sorted(rows.items()):
        f = HERE / name
        here = _sha(f) if f.is_file() else None
        match = (here == want_sha)
        if name != SELF:
            all_match &= match
        out[name] = {"digest_here": (here or "")[:16],
                     "digest_at_the_freeze": want_sha[:16],
                     "identical": match,
                     "asserted_here": name != SELF}
    return {"pipeline_commit": frozen, "modules": out,
            "every_computing_module_matches_the_pipeline_commit": all_match,
            "self_recorded_not_asserted": {
                "module": SELF, "digest": (_sha(HERE / SELF)
                                           if (HERE / SELF).is_file()
                                           else None),
                "verified_by": "the launcher's pre-flight matrix row, "
                               "against the same declaration"}}


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
    # ONE ORACLE READ FOR THE WHOLE RUN, before the first arm.
    oracle = R.winner_source()
    (out_dir / f"winner_source_{a.day.replace('-', '')}.json").write_text(
        json.dumps({k: oracle.get(k) for k in
                    ("path", "sha256", "n_records", "n_closed_records",
                     "n_slugs", "method", "is_final_for_quotation")},
                   indent=1, default=str))
    print(json.dumps({"stage": "oracle_read_once",
                      "sha256": oracle["sha256"][:16],
                      "n_records": oracle["n_records"]}), flush=True)
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
                                         if a.params else None),
                                 winner_source=oracle)
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
        # THE ADMITTING ARM TRAVELS WITH THE NUMBER. A receipt that admits
        # as DESCENDANT and one that admits EXACTLY are different claims
        # about which code built the book, and a reader holding only the
        # combined record could not tell them apart -- it had to open the
        # per-arm result file, which is not where anyone looks.
        cells[arm] = {"D": res["observed_D_cents"],
                      "p": res["p_two_sided"], "n": res["n_draws"],
                      "resumed_from_draw": res["resumed_from_draw"],
                      "book_receipt": res.get("book_receipt")}
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
