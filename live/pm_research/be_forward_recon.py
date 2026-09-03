#!/usr/bin/env python3
"""RECONCILE THE FORWARD METRIC PATH AGAINST A PUBLISHED ANSWER.

WHY THIS EXISTS. `be_forward_metric.py` is a path with no output yet. A path
that has never reproduced a known number is not evidence, whatever it later
emits, and the cheapest moment to discover that is now -- on data that is
ALREADY CONSUMED -- rather than at G=5 on data that is not.

WHAT IS RECONCILED, AND WHAT IS NOT. Iteration 011 published
`Q4_combined_ev` -- net cents against the incumbent, action-deduplicated, per
budget -- together with its FULL per-window increment map. So this module can
drive the aggregation, the paired null and the disclosure over the published
inputs and require them to reproduce the published outputs EXACTLY.

**IT DOES NOT RECONCILE THE UPSTREAM HALF**, and that is stated here rather
than left for a reader to notice: rows -> actions -> per-window net cents is
produced by `harmful_action_eval.evaluate_policy` over a feature pass, and
reproducing THAT needs the tape index and the feature assembly. What this
module proves is that everything DOWNSTREAM of `increment_by_window` -- the
identity, the null, the sidedness, the multiplicity and the disclosure -- is
byte-faithful to a published result. What it leaves unproven is named in the
receipt as `NOT_RECONCILED_HERE`, never omitted.

THE TOLERANCES ARE DECLARED IN THIS FILE AND COMMITTED BEFORE THE RUN, and the
declaring commit's sha is recorded in the artifact this emits. A tolerance
chosen after seeing a disagreement is not a tolerance, it is a repair. If a
predicate fails, the finding is the failure: this module has no branch that
widens a tolerance and none that retries.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import be_forward_metric as FM
import phase2_iter011 as I11

REPO = Path("/home/yuqing/ctaNew")
DERIVED = REPO / "data/pm_5min/derived"

#: The published artifact reconciled against. Bound by sha at run time.
RECON_ARTIFACT = DERIVED / "iter011_conditional_value_v1__coin_btc.json"

#: The cells. All three budgets of both arms -- the whole Q4 family for this
#: coin, so the reconciliation cannot be a lucky single cell.
RECON_ARMS = ("composed_lgbm", "composed_linear")
RECON_BUDGETS = ("5%", "10%", "15%")

# ---------------------------------------------------------------------------
# THE TOLERANCES. DECLARED HERE, COMMITTED BEFORE THE RUN.
# ---------------------------------------------------------------------------
#: Counts and permutation p-values are EXACT. A permutation p is the rational
#: (ge + 1) / (n_perm + 1) over integers, so a correct replay reproduces it
#: bit-for-bit; any difference at all means a different stream, a different
#: order or a different tally, and each of those is a finding rather than a
#: rounding artefact.
TOL_EXACT = 0.0
#: Cent sums are float64 reductions. `math.fsum` is exact, but the published
#: value was produced by a different call sequence, so a few ulps are allowed.
#: 1e-6 cents on a ~3.9e3 cents statistic is ~2.6e-10 relative -- about three
#: orders of magnitude looser than float64 eps at that magnitude, and three
#: orders TIGHTER than any difference that could hide a real defect.
TOL_CENTS_ABS = 1e-6
#: The delivered cancellation rate against its nominal budget. REPORTED, never
#: a pass/fail of this path: it is a property of the score distribution, and
#: reporting it beside the nominal rate is the declared risk of the
#: FROZEN_FROM_TRAIN_QUANTILE form.
TOL_RATE_REPORTED_ONLY = None

DECLARED_PREDICATES = {
    "P1_increment_identity": (
        "sum(increment_by_window) == net_cents - incumbent_net_cents",
        TOL_CENTS_ABS),
    "P2_statistic_matches_sum": (
        "sum(increment_by_window) == the cell's adjudicated `statistic`",
        TOL_CENTS_ABS),
    "P3_window_count": (
        "len(increment_by_window) == statistic_n == n_actions", TOL_EXACT),
    "P4_p_two_sided": (
        "paired_null(published increments)['p_two_sided'] == the cell's "
        "p_two_sided_REPORTED_NOT_ADJUDICATED", TOL_EXACT),
    "P5_p_one_sided": (
        "paired_null(...)['p_value'] == the cell's adjudicated p_value",
        TOL_EXACT),
    "P6_holm": (
        "holm over the published family reproduces the cell's holm_p",
        TOL_EXACT),
    "P7_order_invariance": (
        "the borrowed null returns the same p on the published increments "
        "SHUFFLED -- R-234 at the point of consumption", TOL_EXACT),
}


#: The commit in which the tolerances above were fixed, BEFORE the reconciler
#: had ever been run. Named as a constant so the claim "declared first" is
#: checkable by anyone from the artifact alone, and so it survives later edits
#: to this file: `tolerances_unchanged_since` re-reads the declaration OUT OF
#: this commit and compares it to what actually ran.
TOLERANCE_DECLARING_COMMIT = "1e9b6626e23ddab1de10a216cad1d425cd46973f"

#: The lines that constitute the declaration. Compared verbatim.
_DECL_MARKERS = ("TOL_EXACT =", "TOL_CENTS_ABS =", "TOL_RATE_REPORTED_ONLY =",
                 "DECLARED_PREDICATES = {")


def _declaration_lines(text: str) -> list:
    """The declaration block, extracted the same way from any version."""
    out, inside = [], False
    for ln in text.splitlines():
        if ln.startswith("DECLARED_PREDICATES = {"):
            inside = True
        if inside:
            out.append(ln.rstrip())
            if ln.startswith("}"):
                inside = False
            continue
        if any(ln.startswith(m) for m in _DECL_MARKERS):
            out.append(ln.rstrip())
    return out


def tolerances_unchanged_since(commit: str = None, path: Path = None) -> dict:
    """Are the tolerances that RAN the ones that were DECLARED?

    A later edit to this file is legitimate -- adding a disclosure, fixing a
    check count -- but it must not be able to move a tolerance silently. This
    re-reads the declaration block out of the declaring commit and compares it
    line for line with the block in the file that just ran."""
    commit = commit or TOLERANCE_DECLARING_COMMIT
    f = Path(path or __file__).resolve()
    tree = f.parents[2]
    rel = str(f.relative_to(tree))
    r = subprocess.run(["git", "-C", str(tree), "show", f"{commit}:{rel}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return {"checked": False,
                "why": f"cannot read {rel} at {commit[:12]}: {r.stderr[:120]}"}
    then = _declaration_lines(r.stdout)
    now = _declaration_lines(f.read_text())
    return {
        "checked": True, "declaring_commit": commit,
        "n_declaration_lines": len(now),
        "unchanged": then == now,
        "declaration_sha16_then": hashlib.sha256(
            "\n".join(then).encode()).hexdigest()[:16],
        "declaration_sha16_now": hashlib.sha256(
            "\n".join(now).encode()).hexdigest()[:16],
        "why": ("the tolerances that judged this reconciliation are compared "
                "line-for-line with the ones committed before it was ever "
                "run. A later edit to this file cannot move a tolerance "
                "without turning this False."),
    }


def _sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def declaring_commit(path: Path = None) -> dict:
    """The commit that carries THIS file's declared tolerances.

    Recorded in the emitted artifact so a reader can check that the tolerance
    predates the number. `dirty` is reported rather than hidden: a dirty tree
    means the committed sha does not describe the bytes that ran, and that is
    the reader's business."""
    f = Path(path or __file__).resolve()
    tree = f.parents[2]

    def git(*a):
        try:
            r = subprocess.run(["git", "-C", str(tree), *a],
                               capture_output=True, text=True, timeout=30)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:                             # noqa: BLE001
            return None
    rel = str(f.relative_to(tree))
    return {
        "file": rel,
        "file_sha256_prefix": _sha256(f)[:16],
        "last_commit_touching_this_file": git("log", "-1", "--format=%H",
                                              "--", rel),
        "commit_time_utc": git("log", "-1", "--format=%cI", "--", rel),
        "head": git("rev-parse", "HEAD"),
        "file_dirty": bool(git("status", "--porcelain", "--", rel)),
        "why": ("the tolerances in DECLARED_PREDICATES are constants in this "
                "file; the commit above is when they were fixed. A tolerance "
                "committed AFTER the number it judges is a repair."),
    }


def _cell(art: dict, arm: str, budget: str) -> dict:
    return art["family"]["cells"][f"{arm}/Q4_combined_ev/{budget}"]


def _economics(art: dict, arm: str, budget: str) -> dict:
    return art["results"]["btc"][arm]["economics"][budget]


def reconcile_cell(art: dict, arm: str, budget: str) -> dict:
    """Every declared predicate for one cell, COMPUTED (rule 10).

    No verdict string is written beside a number: each predicate carries its
    observed value, its expected value, its tolerance and its boolean."""
    import math
    import random
    c = _cell(art, arm, budget)
    e = _economics(art, arm, budget)
    inc = {k: float(v) for k, v in e["increment_by_window"].items()}

    got_sum = math.fsum(inc[k] for k in sorted(inc))
    want_id = float(e["net_cents"]) - float(e["incumbent_net_cents"])
    want_stat = float(c["statistic"])

    null = FM.paired_null(inc, n_perm=int(c["permutation_floor"]["n_draws"]),
                          seed=I11.PERM_SEED_011)
    keys = list(inc)
    random.Random(4242).shuffle(keys)
    null_shuf = FM.paired_null({k: inc[k] for k in keys},
                               n_perm=int(c["permutation_floor"]["n_draws"]),
                               seed=I11.PERM_SEED_011)

    def pred(name, observed, expected, tol):
        if tol is None:
            okv = None
        elif tol == 0.0:
            okv = observed == expected
        else:
            okv = abs(observed - expected) <= tol
        return {"predicate": DECLARED_PREDICATES[name][0],
                "observed": observed, "expected": expected,
                "tolerance": tol, "holds": okv,
                "abs_difference": (None if not isinstance(observed, (int, float))
                                   or not isinstance(expected, (int, float))
                                   else abs(observed - expected))}

    n_tot = int(e["n_actions_total"])
    n_can = int(e["n_cancelled_actions"])
    nominal = float(budget.rstrip("%")) / 100.0
    return {
        "cell": c["cell"], "arm": arm, "budget": budget,
        "n_windows": len(inc),
        "P1_increment_identity": pred("P1_increment_identity", got_sum,
                                      want_id, TOL_CENTS_ABS),
        "P2_statistic_matches_sum": pred("P2_statistic_matches_sum", got_sum,
                                         want_stat, TOL_CENTS_ABS),
        "P3_window_count": pred("P3_window_count", len(inc),
                                int(c["statistic_n"]), TOL_EXACT),
        "P4_p_two_sided": pred("P4_p_two_sided", null["p_two_sided"],
                               float(c["p_two_sided_REPORTED_NOT_ADJUDICATED"]),
                               TOL_EXACT),
        "P5_p_one_sided": pred("P5_p_one_sided", null["p_value"],
                               float(c["p_value"]), TOL_EXACT),
        "P7_order_invariance": pred("P7_order_invariance",
                                    null_shuf["p_two_sided"],
                                    null["p_two_sided"], TOL_EXACT),
        "delivered_rate_REPORTED": {
            "nominal_budget": nominal,
            "delivered": n_can / n_tot if n_tot else None,
            "n_cancelled_actions": n_can, "n_actions_total": n_tot,
            "tolerance": TOL_RATE_REPORTED_ONLY,
            "uninformative_for_the_forward_case": (
                "these cells were produced RETROSPECTIVELY (kk = int(n*b), "
                "cutoff read off the ranking), so delivered EQUALS nominal by "
                "construction. Under a declared FROZEN_FROM_TRAIN_QUANTILE "
                "theta it will not, and that gap is the form's declared risk. "
                "Read this column as arithmetic, never as reassurance."),
            "why_reported_not_judged": (
                "the delivered rate is a property of the score distribution, "
                "not of this path. It is the FROZEN_FROM_TRAIN_QUANTILE "
                "form's own declared risk and is reported beside the nominal "
                "rate, never used to pass or fail the reconciliation."),
        },
        "null_seed": null["perm_seed"], "null_n_perm": null["n_perm"],
        "null_unit_order": null["unit_order"],
    }


def pairing_divergence(art: dict) -> dict:
    """A DIVERGENCE THE RECONCILIATION SURFACED, reported not adjudicated.

    Every declared predicate held, so this is not a failure -- it sits in the
    half this module names NOT_RECONCILED_HERE, and it is exactly the kind of
    thing that half was flagged for.

    Iteration 011 pairs the arms BY COUNT: `kk = max(1, int(len(order) * b))`
    and each arm cancels ITS OWN top-kk actions, so the two arms use DIFFERENT
    cutoff scores and the same number of cancellations. `be_forward_metric`'s
    `increment()` pairs them BY THRESHOLD: one declared theta is applied to
    both arms, so the cutoffs are equal and the counts differ.

    Under a declared FROZEN_FROM_TRAIN_QUANTILE grid these are different
    estimands, and which one the forward read uses is a DECLARATION nobody has
    made. Computed here from the artifact's own fields rather than asserted:
    a single `n_cancelled_actions` per cell, shared by both arms, is what
    count-matching looks like."""
    ev = []
    for arm in RECON_ARMS:
        for b in RECON_BUDGETS:
            try:
                e = _economics(art, arm, b)
            except KeyError:
                continue
            ev.append({"arm": arm, "budget": b,
                       "n_cancelled_actions": e["n_cancelled_actions"],
                       "n_actions_total": e["n_actions_total"]})
    by_budget = {}
    for r in ev:
        by_budget.setdefault(r["budget"], set()).add(r["n_cancelled_actions"])
    return {
        "published_pairing": "BY COUNT (each arm cancels its own top-kk)",
        "evidence": ("one `n_cancelled_actions` per cell, identical across "
                     "arms at each budget"),
        "n_cancelled_identical_across_arms_per_budget": {
            b: (len(v) == 1) for b, v in sorted(by_budget.items())},
        "be_forward_metric_pairing": ("BY THRESHOLD (one declared theta "
                                      "applied to both arms; counts differ)"),
        "same_estimand": False,
        "consequence": ("a forward run under a declared theta grid will NOT "
                        "reproduce these published numbers, and should not be "
                        "expected to: the pairing rule differs. Which rule "
                        "the forward read declares is an OPEN DECLARATION."),
        "who_declares": "the USER (rule 14); this module selects neither",
    }


def reconcile(artifact: Path = None, outdir: Path = None) -> dict:
    """Every cell, every predicate. Emits a receipt; adjudicates nothing.

    There is no branch here that widens a tolerance, retries a draw or drops a
    cell. If a predicate is False the receipt says so and `all_hold` is False;
    what that MEANS is the coordinator's and the USER's to rule."""
    ap = Path(artifact or RECON_ARTIFACT)
    art = json.loads(ap.read_text())
    cells = []
    for arm in RECON_ARMS:
        for b in RECON_BUDGETS:
            try:
                cells.append(reconcile_cell(art, arm, b))
            except KeyError as e:
                cells.append({"cell": f"{arm}/Q4_combined_ev/{b}",
                              "status": "ABSENT_FROM_ARTIFACT",
                              "missing_key": str(e)})
    # P6: Holm over the published family, reproduced from the artifact's own
    # p-values and compared to its own holm_p -- the multiplicity arithmetic,
    # not just the per-cell p.
    fam = art["family"]["cells"]
    pvals = {k: v["p_value"] for k, v in fam.items() if "p_value" in v}
    holm_got = I11.holm(pvals)
    holm_pred = []
    for k, v in sorted(fam.items()):
        if "holm_p" not in v or k not in holm_got:
            continue
        g = holm_got[k]
        g = g["holm_p"] if isinstance(g, dict) else g
        holm_pred.append({"cell": k, "observed": g, "expected": v["holm_p"],
                          "holds": g == v["holm_p"], "tolerance": TOL_EXACT})
    evaluated = [p for c in cells for k, p in c.items()
                 if k.startswith("P") and isinstance(p, dict)
                 and p.get("holds") is not None]
    n_true = sum(1 for p in evaluated if p["holds"])
    n_holm_true = sum(1 for p in holm_pred if p["holds"])
    return {
        "protocol": "BE_FORWARD_RECON_V1",
        "as_of_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "reconciled_against": {
            "artifact": str(ap), "sha256": _sha256(ap),
            "bytes": ap.stat().st_size,
            "as_of": art.get("as_of"),
            "class": art.get("development_evidence"),
            "why_free": ("iteration 011's populations are DEVELOPMENT "
                         "(`is_a_validation` false) and already consumed. "
                         "Reconciling here opens no seal and spends no "
                         "forward day."),
        },
        "declaration": {
            "predicates": {k: {"statement": v[0], "tolerance": v[1]}
                           for k, v in DECLARED_PREDICATES.items()},
            "declared_in": declaring_commit(),
            "tolerances_unchanged_since_declaration":
                tolerances_unchanged_since(),
            "declared_before_the_run": ("the tolerances are constants in "
                                        "be_forward_recon.py and the commit "
                                        "above carries them; this receipt "
                                        "records that commit so a reader can "
                                        "check the order for themselves"),
        },
        "cells": cells,
        "P6_holm": {"per_cell": holm_pred, "n": len(holm_pred),
                    "n_holds": n_holm_true,
                    "predicate": DECLARED_PREDICATES["P6_holm"][0]},
        "summary": {
            "n_cells": len(cells),
            "n_predicates_evaluated": len(evaluated),
            "n_predicates_true": n_true,
            "n_predicates_false": len(evaluated) - n_true,
            "all_hold": (n_true == len(evaluated)
                         and n_holm_true == len(holm_pred)
                         and len(evaluated) > 0),
        },
        "NOT_RECONCILED_HERE": {
            "what": ("rows -> actions -> per-window net cents, i.e. "
                     "`reduce_window` feeding `evaluate_policy`"),
            "why": ("reproducing `increment_by_window` from rows needs the "
                    "tape index and the feature assembly; this module "
                    "reconciles everything DOWNSTREAM of that map"),
            "consequence": ("a PASS here does not license a forward number "
                            "on its own -- it licenses the aggregation, the "
                            "null, the sidedness, the multiplicity and the "
                            "disclosure, and says so"),
        },
        "PAIRING_CONVENTION_DIVERGENCE": pairing_divergence(art),
        "adjudicates": None,
        "who_adjudicates": "the coordinator verifies; the USER rules (rule 14)",
    }


# ---------------------------------------------------------------------------
# SELFTEST. The reconciler is itself an instrument, so it ships falsifiers:
# a doctored artifact must FAIL the predicate it doctors, and the real one
# must be ADMITTED. Without the first half a reconciliation that always
# passed would be indistinguishable from one that worked.
# ---------------------------------------------------------------------------
EXPECTED_CHECKS = 21


def selftest() -> int:
    import copy
    checks = 0
    fails = []

    def ok(c, label):
        nonlocal checks
        checks += 1
        print(f"PASS: {label}" if c else f"FAIL: {label}")
        if not c:
            fails.append(label)

    art = json.loads(RECON_ARTIFACT.read_text())

    # --- the declaration must be inspectable and complete
    ok(set(DECLARED_PREDICATES) == {"P1_increment_identity",
                                    "P2_statistic_matches_sum",
                                    "P3_window_count", "P4_p_two_sided",
                                    "P5_p_one_sided", "P6_holm",
                                    "P7_order_invariance"},
       "the declared predicate set is fixed in the file and complete")
    ok(all(t == TOL_EXACT for n, (_, t) in DECLARED_PREDICATES.items()
           if n in ("P3_window_count", "P4_p_two_sided", "P5_p_one_sided",
                    "P6_holm", "P7_order_invariance")),
       "counts and permutation p-values are declared EXACT -- a permutation p "
       "is a rational over integers, so any difference is a finding")
    dc = declaring_commit()
    ok(dc["file"].endswith("be_forward_recon.py") and dc["head"],
       "the receipt can name the commit the tolerances were declared in")
    tu = tolerances_unchanged_since()
    ok(tu["checked"] and tu["unchanged"] is True,
       f"POSITIVE CONTROL: the tolerances that ran are byte-identical to "
       f"those committed at {TOLERANCE_DECLARING_COMMIT[:12]} "
       f"({tu.get('n_declaration_lines')} declaration lines)")
    _tamp = _declaration_lines(
        Path(__file__).read_text().replace("TOL_CENTS_ABS = 1e-6",
                                           "TOL_CENTS_ABS = 1e6"))
    ok(_tamp != _declaration_lines(Path(__file__).read_text()),
       "KNOWN-BAD: widening a tolerance CHANGES the declaration block, so the "
       "comparison above would turn False -- the check can fail")

    # --- POSITIVE CONTROL: a real cell reconciles on every predicate
    r = reconcile_cell(art, "composed_lgbm", "10%")
    for k in ("P1_increment_identity", "P2_statistic_matches_sum",
              "P3_window_count", "P4_p_two_sided", "P5_p_one_sided",
              "P7_order_invariance"):
        ok(r[k]["holds"] is True,
           f"POSITIVE CONTROL: the published composed_lgbm/10% cell satisfies "
           f"{k} ({r[k]['observed']!r} vs {r[k]['expected']!r})")

    # --- KNOWN-BAD: doctor one window by one cent; P1 and P2 must FAIL and
    # the p predicates must fail too, because the statistic moved.
    bad = copy.deepcopy(art)
    e = bad["results"]["btc"]["composed_lgbm"]["economics"]["10%"]
    k0 = sorted(e["increment_by_window"])[0]
    e["increment_by_window"][k0] += 1.0
    rb = reconcile_cell(bad, "composed_lgbm", "10%")
    ok(rb["P1_increment_identity"]["holds"] is False,
       "KNOWN-BAD: moving ONE window by 1 cent breaks the increment identity "
       "-- the tolerance is tight enough to see a single-cent edit")
    ok(rb["P2_statistic_matches_sum"]["holds"] is False,
       "KNOWN-BAD: and it breaks the statistic match")
    ok(abs(rb["P1_increment_identity"]["abs_difference"] - 1.0) < 1e-9,
       "KNOWN-BAD: the reported difference is exactly the 1 cent injected, so "
       "the predicate measures what it claims to")

    # --- KNOWN-BAD: doctor the published p; P4 must FAIL
    bad2 = copy.deepcopy(art)
    c2 = bad2["family"]["cells"]["composed_lgbm/Q4_combined_ev/10%"]
    c2["p_two_sided_REPORTED_NOT_ADJUDICATED"] = 0.5
    r2 = reconcile_cell(bad2, "composed_lgbm", "10%")
    ok(r2["P4_p_two_sided"]["holds"] is False,
       "KNOWN-BAD: a doctored published p FAILS P4 -- the predicate compares "
       "against the artifact rather than against itself")

    # --- KNOWN-BAD: drop a window; P3 must FAIL
    bad3 = copy.deepcopy(art)
    e3 = bad3["results"]["btc"]["composed_lgbm"]["economics"]["10%"]
    e3["increment_by_window"].pop(sorted(e3["increment_by_window"])[0])
    r3 = reconcile_cell(bad3, "composed_lgbm", "10%")
    ok(r3["P3_window_count"]["holds"] is False,
       "KNOWN-BAD: a dropped window FAILS the exact window-count predicate")

    # --- the delivered rate is REPORTED, never judged
    ok(r["delivered_rate_REPORTED"]["tolerance"] is None
       and r["delivered_rate_REPORTED"]["delivered"] is not None,
       "the delivered cancellation rate is COMPUTED and carries no tolerance "
       "-- reported beside the nominal rate, never a pass/fail of this path")

    # --- the receipt must name what it does NOT reconcile
    full = reconcile()
    ok(isinstance(full.get("NOT_RECONCILED_HERE"), dict)
       and "reduce_window" in full["NOT_RECONCILED_HERE"]["what"],
       "the receipt NAMES the half it does not reconcile rather than leaving "
       "a reader to infer the scope")
    ok(full["adjudicates"] is None,
       "the reconciler adjudicates nothing (rule 14)")
    pd_ = full["PAIRING_CONVENTION_DIVERGENCE"]
    ok(pd_["same_estimand"] is False
       and all(pd_["n_cancelled_identical_across_arms_per_budget"].values()),
       "the pairing divergence is COMPUTED from the artifact: one shared "
       "n_cancelled_actions per budget across both arms is what count-matching "
       "looks like, and be_forward_metric matches by THRESHOLD instead")
    ok("uninformative_for_the_forward_case" in
       full["cells"][0]["delivered_rate_REPORTED"],
       "the delivered-rate column carries its own warning: these cells are "
       "retrospective, so delivered equals nominal by construction")

    print(f"\n{checks} checks passed" if not fails
          else f"\n{len(fails)} FAILURES of {checks} checks")
    for f in fails:
        print(f"  - {f}")
    if checks != EXPECTED_CHECKS:
        print(f"FAIL: ran {checks} checks, EXPECTED_CHECKS={EXPECTED_CHECKS}.")
        return 1
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = list(sys.argv) if argv is None else list(argv)
    if "--selftest" in argv:
        return selftest()
    if "--reconcile" in argv:
        out = reconcile()
        print(json.dumps(out, indent=1, sort_keys=True, default=str))
        return 0 if out["summary"]["all_hold"] else 1
    print("usage: be_forward_recon.py --selftest | --reconcile")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
