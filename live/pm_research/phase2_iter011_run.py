#!/usr/bin/env python3
"""Iteration 011 DEVELOPMENT RUN — standalone pipeline, own receipt family.

Preregistration: ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md (3b71d3e, FROZEN
by user ruling R-232). Estimands and heads: phase2_iter011.py.

STANDALONE, AND THAT IS A DECLARED PROPERTY, NOT AN ACCIDENT. This module is not
in phase2_arms.CODE_IDENTITY_FILES and it MODIFIES no lattice file — it imports
the lattice's loading path so the guarded, proven route to the data is reused
rather than reimplemented. Consequence, stated in the receipt: no four-arm fit
is invalidated by this run, and the annotation-merge wiring (option (a)) waits
for the next four-arm cycle rather than being dragged in by 011.

TWO ARMS, and they differ in MODEL CLASS only (R-232 9.1):
    composed_linear : logistic heads + ridge magnitude heads
    composed_lgbm   : pinned LGBM classifier + pinned LGBM regressor
Both compose Q4 from the four heads. NEITHER fits Q4 (prereg §6).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import phase2_arms as PA
import phase2_declaration as D
import phase2_iter011 as I11

OUT = PA.DERIVED / "iter011_conditional_value_v1.json"
RECEIPT_FAMILY = "ITER011_CONDITIONAL_VALUE"


def build_design(block: dict, i: int) -> list:
    """x for row i: PM + fine + state. Both arms see the SAME features; they
    differ in model class, not in what they are shown (R-232 9.1)."""
    return block["PM"][i] + block["FN"][i] + block["ST"][i]


def head_targets(rows, latency_ms=None) -> dict:
    """The four heads' labels, each on its own population, indices carried.

    Indices travel so a head's rows can be traced back to the design matrix
    without re-deriving membership — re-deriving is how two definitions of a
    population diverge."""
    L = latency_ms
    y_fill, idx_prev = [], []
    for i, r in enumerate(rows):
        p = I11.preventable(r, L)
        y_fill.append(1 if p else 0)
        if p:
            idx_prev.append(i)
    v = {i: I11.signed_v_cancel(rows[i], L) for i in idx_prev}
    y_harm = [1 if v[i] > 0 else 0 for i in idx_prev]
    idx_h = [i for i in idx_prev if v[i] > 0]
    idx_g = [i for i in idx_prev if v[i] < 0]
    idx_z = [i for i in idx_prev if v[i] == 0.0]
    return {"y_fill": y_fill, "idx_prev": idx_prev, "y_harm": y_harm,
            "idx_harm": idx_h, "idx_good": idx_g, "idx_zero": idx_z,
            "m_harm_target": [v[i] for i in idx_h],
            "m_good_target": [-v[i] for i in idx_g],
            "counts": {"n_rows": len(rows), "n_preventable": len(idx_prev),
                       "n_v_positive": len(idx_h), "n_v_negative": len(idx_g),
                       "n_v_zero": len(idx_z)}}


def fit_arm(arm: str, X, tg: dict, seed_note: str = "") -> dict:
    """Fit the FOUR heads for one arm. Q4 is composed, never fitted."""
    import harmful_fast_compute as fc
    if arm not in I11.ARMS_011:
        raise RuntimeError(f"REFUSED: unknown arm {arm!r}; the frozen "
                           f"preregistration declares {I11.ARMS_011}")
    Xp = [X[i] for i in tg["idx_prev"]]
    Xh = [X[i] for i in tg["idx_harm"]]
    Xg = [X[i] for i in tg["idx_good"]]

    if arm == "composed_linear":
        Z, mu, sd = fc.fast_zscale(X, X)
        w_fill = fc.fast_fit_logistic_w(Z, tg["y_fill"],
                                        [1.0] * len(tg["y_fill"]))
        p_fill = [fc.fast_predict_p(w_fill, z) for z in Z]
        Zp = [Z[i] for i in tg["idx_prev"]]
        w_harm = (fc.fast_fit_logistic_w(Zp, tg["y_harm"], [1.0] * len(Zp))
                  if Zp else None)
        p_harm = [fc.fast_predict_p(w_harm, z) for z in Zp] if w_harm else []
        Zh = [Z[i] for i in tg["idx_harm"]]
        Zg = [Z[i] for i in tg["idx_good"]]
        wmh = (fc.fast_fit_ridge_w(Zh, tg["m_harm_target"],
                                   [1.0] * len(Zh), lam=10.0)
               if len(Zh) >= I11.UNDERPOWERED_MIN_N else None)
        wmg = (fc.fast_fit_ridge_w(Zg, tg["m_good_target"],
                                   [1.0] * len(Zg), lam=10.0)
               if len(Zg) >= I11.UNDERPOWERED_MIN_N else None)
        m_harm = [float(sum(a * b for a, b in zip(wmh, z))) for z in Zh] if wmh else []
        m_good = [float(sum(a * b for a, b in zip(wmg, z))) for z in Zg] if wmg else []
        return {"arm": arm, "p_fill": p_fill, "p_harm": p_harm,
                "m_harm": m_harm, "m_good": m_good,
                "m_harm_fitted": wmh is not None,
                "m_good_fitted": wmg is not None,
                "model": {"kind": "linear", "mu": mu, "sd": sd,
                          "w_fill": w_fill, "w_harm": w_harm,
                          "wmh": wmh, "wmg": wmg},
                "model_class": "logistic (hazard/sign) + ridge lam=10 (magnitudes)"}

    import lightgbm as lgb
    import numpy as np
    A = np.asarray(X, dtype=np.float64)
    clf = lgb.LGBMClassifier(**D.LGBM_PARAMS)
    clf.fit(A, np.asarray(tg["y_fill"]))
    p_fill = clf.predict_proba(A)[:, 1].tolist()
    Ap = A[tg["idx_prev"]] if tg["idx_prev"] else A[:0]
    p_harm = []
    if len(Ap) and len(set(tg["y_harm"])) > 1:
        c2 = lgb.LGBMClassifier(**D.LGBM_PARAMS)
        c2.fit(Ap, np.asarray(tg["y_harm"]))
        p_harm = c2.predict_proba(Ap)[:, 1].tolist()
    m_harm, m_good, fh, fg = [], [], False, False
    _reg_h = _reg_g = None
    for idx, tgt, out in ((tg["idx_harm"], tg["m_harm_target"], "h"),
                          (tg["idx_good"], tg["m_good_target"], "g")):
        if len(idx) >= I11.UNDERPOWERED_MIN_N:
            reg = lgb.LGBMRegressor(**D.LGBM_VALUE_PARAMS)
            reg.fit(A[idx], np.asarray(tgt))
            pred = reg.predict(A[idx]).tolist()
            if out == "h":
                m_harm, fh, _reg_h = pred, True, reg
            else:
                m_good, fg, _reg_g = pred, True, reg
    return {"arm": arm, "p_fill": p_fill, "p_harm": p_harm,
            "m_harm": m_harm, "m_good": m_good,
            "m_harm_fitted": fh, "m_good_fitted": fg,
            "model": {"kind": "lgbm", "clf": clf,
                      "c2": locals().get("c2"),
                      "reg_h": locals().get("_reg_h"),
                      "reg_g": locals().get("_reg_g")},
            "model_class": "LGBM classifier (hazard/sign) + LGBM regressor "
                           "(magnitudes), params PINNED in phase2_declaration"}


def apply_arm(fitres: dict, X, tg: dict) -> dict:
    """Apply FITTED heads to a NEW population. Out-of-sample by construction.

    In-sample metrics on a boosted model are inflated to the point of being
    misleading even when labelled as in-sample, so the heads are fitted on one
    development population and reported on another. This is still DEVELOPMENT
    evidence — both populations are development — but it is not self-graded."""
    import harmful_fast_compute as fc
    m = fitres["model"]
    Xp = [X[i] for i in tg["idx_prev"]]
    Xh = [X[i] for i in tg["idx_harm"]]
    Xg = [X[i] for i in tg["idx_good"]]
    if m["kind"] == "linear":
        mu, sd = m["mu"], m["sd"]
        def z(v):
            return [1.0] + [(v[i] - mu[i]) / sd[i] for i in range(len(mu))]
        p_fill = [fc.fast_predict_p(m["w_fill"], z(v)) for v in X]
        p_harm = ([fc.fast_predict_p(m["w_harm"], z(v)) for v in Xp]
                  if m["w_harm"] else [])
        mh = ([float(sum(a * b for a, b in zip(m["wmh"], z(v)))) for v in Xh]
              if m["wmh"] else [])
        mg = ([float(sum(a * b for a, b in zip(m["wmg"], z(v)))) for v in Xg]
              if m["wmg"] else [])
    else:
        import numpy as np
        A = np.asarray(X, dtype=np.float64)
        p_fill = m["clf"].predict_proba(A)[:, 1].tolist()
        p_harm = (m["c2"].predict_proba(A[tg["idx_prev"]])[:, 1].tolist()
                  if m["c2"] is not None and tg["idx_prev"] else [])
        mh = (m["reg_h"].predict(A[tg["idx_harm"]]).tolist()
              if m["reg_h"] is not None and tg["idx_harm"] else [])
        mg = (m["reg_g"].predict(A[tg["idx_good"]]).tolist()
              if m["reg_g"] is not None and tg["idx_good"] else [])
    return {"arm": fitres["arm"], "p_fill": p_fill, "p_harm": p_harm,
            "m_harm": mh, "m_good": mg,
            "m_harm_fitted": fitres["m_harm_fitted"],
            "m_good_fitted": fitres["m_good_fitted"],
            "model_class": fitres["model_class"],
            "evaluation": "OUT-OF-SAMPLE within development"}


def report_arm(fitres: dict, tg: dict) -> dict:
    """All four heads, always, including failures (prereg §3)."""
    q1 = I11.head_report("Q1_arrival", "probability",
                         fitres["p_fill"], tg["y_fill"])
    q2 = I11.head_report("Q2_sign", "probability",
                         fitres["p_harm"], tg["y_harm"])
    q3h = I11.head_report("Q3_m_harm", "magnitude",
                          fitres["m_harm"], tg["m_harm_target"])
    q3g = I11.head_report("Q3_m_good", "magnitude",
                          fitres["m_good"], tg["m_good_target"])
    rep = I11.four_head_report(q1, q2, q3h, q3g)
    rep["arm"] = fitres["arm"]
    rep["model_class"] = fitres["model_class"]
    rep["magnitude_heads_fitted"] = {"m_harm": fitres["m_harm_fitted"],
                                     "m_good": fitres["m_good_fitted"]}
    return rep


def selftest() -> int:
    """Known-bads BEFORE numbers. Drives both arms on synthetic data."""
    fails = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    L = str(D.TARGET_LATENCY_MS)

    def row(v, fill=True, shares=1.0):
        return {"any_fill_ahead": fill,
                "latency": {L: {"preventable_value_cents": v,
                                "preventable_shares": shares if fill else 0.0,
                                "stale_shares": 0.0}}}

    rows = ([row(5.0)] * 120 + [row(-4.0)] * 120 + [row(0.0)] * 10
            + [row(1.0, fill=False)] * 50)
    tg = head_targets(rows)
    c = tg["counts"]
    ok(c["n_rows"] == 300 and c["n_preventable"] == 250,
       "targets: the no-fill rows are excluded from the preventable base")
    ok(c["n_v_positive"] + c["n_v_negative"] + c["n_v_zero"] == c["n_preventable"],
       "targets: harm/good/zero PARTITION the base (no silent drops)")
    ok(len(tg["y_fill"]) == len(rows),
       "targets: the arrival label covers EVERY row, not just fills")
    ok(len(tg["m_harm_target"]) == c["n_v_positive"]
       and all(t > 0 for t in tg["m_harm_target"]),
       "targets: m_harm is on V>0 only and is positive")
    ok(len(tg["m_good_target"]) == c["n_v_negative"]
       and all(t > 0 for t in tg["m_good_target"]),
       "targets: m_good is on V<0 only and is POSITIVE by definition (-V)")

    import random as _r
    rr = _r.Random(11)
    X = [[1.0, rr.random(), rr.random(), rr.random()] for _ in rows]

    for arm in I11.ARMS_011:
        try:
            fr = fit_arm(arm, X, tg)
            rep = report_arm(fr, tg)
            ok(rep["all_heads_reported"], f"{arm}: ALL FOUR heads reported")
            ok(len(fr["p_fill"]) == len(rows),
               f"{arm}: p_fill covers every row")
            ok(len(fr["p_harm"]) == c["n_preventable"],
               f"{arm}: p_harm is on the PREVENTABLE base, not all rows")
            ok(len(fr["m_harm"]) in (0, c["n_v_positive"]),
               f"{arm}: m_harm is on V>0 only")
            ecv = I11.compose_expected_cancel_value(
                fr["p_fill"][0], 0.5,
                (fr["m_harm"] or [1.0])[0], (fr["m_good"] or [1.0])[0])
            ok(isinstance(ecv, float), f"{arm}: Q4 COMPOSES (never fitted)")
        except Exception as e:
            ok(False, f"{arm}: fits and reports ({type(e).__name__}: {e})")

    try:
        fit_arm("composed_magic", X, tg)
        ok(False, "an UNDECLARED arm is REFUSED")
    except RuntimeError as e:
        ok("frozen preregistration" in str(e), "an UNDECLARED arm is REFUSED")

    thin = [row(5.0)] * 3 + [row(-4.0)] * 3
    tgt = head_targets(thin)
    frt = fit_arm("composed_linear", [[1.0, 0.5]] * 6, tgt)
    rept = report_arm(frt, tgt)
    ok(not frt["m_harm_fitted"] and not frt["m_good_fitted"],
       "a THIN magnitude population is not fitted (min_n respected)")
    ok(set(rept["underpowered_heads"]) >= {"Q3_m_harm", "Q3_m_good"},
       "and the unfitted magnitude heads are REPORTED UNDERPOWERED, not dropped")

    ok("phase2_iter011_run.py" not in PA.CODE_IDENTITY_FILES,
       "this runner is NOT in the identity lattice — the standalone property "
       "is checked, not assumed")

    print(f"\n{'ITER011 RUN SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    return 1 if fails else 0


# ---- INCUMBENT COUNTERPARTS, and the two heads that have none --------------
# prereg §5(2) declares an incremental-over-incumbent null "per head". For two
# heads that null is NOT WELL-DEFINED, and saying so is better than computing
# something that looks like it:
#
#   Q1_arrival      the incumbent HAS a hazard head -> comparable
#   Q4_combined_ev  the incumbent HAS an expected_cancel_value -> comparable
#   Q2_sign         the incumbent has NO sign head. There is nothing to be
#   Q3_magnitudes   incremental TO. Reporting an increment against a head that
#                   does not exist would be inventing a baseline.
#
# So those cells carry NO_INCUMBENT_COUNTERPART as a STATUS, and the matched-
# random null (which IS defined for every head) still applies to all four. The
# gap is in the frozen document, not in the data; it is REPORTED, never patched
# by BE, because amending a frozen estimand after seeing the design is the thing
# the freeze exists to prevent.
INCUMBENT_COMPARABLE = {"Q1_arrival": True, "Q2_sign": False,
                        "Q3_magnitudes": False, "Q4_combined_ev": True}
NO_COUNTERPART = "NO_INCUMBENT_COUNTERPART"


def incumbent_null_applicability() -> dict:
    return {
        "comparable": {k: v for k, v in INCUMBENT_COMPARABLE.items()},
        "not_applicable_heads": sorted(k for k, v in INCUMBENT_COMPARABLE.items()
                                       if not v),
        "status_for_those_cells": NO_COUNTERPART,
        "why": "the incumbent (INCUMBENT_REWEIGHTED_ONLY) has a hazard head and "
               "a composed expected value, but NO sign head and NO magnitude "
               "heads. An increment against a head that does not exist would be "
               "an invented baseline (rule 9: a baseline must remove the "
               "tautology, not manufacture one).",
        "matched_random_still_applies_to_all_four": True,
        "handling": "REPORTED as a status in every affected cell. BE does not "
                    "amend a frozen estimand; if this gap should be closed the "
                    "USER amends the preregistration.",
    }


def main() -> int:
    """The 011 DEVELOPMENT run. Known-bads first: selftest gates the run."""
    if "--selftest" in sys.argv:
        return selftest()
    if selftest():
        raise SystemExit("REFUSED: selftest RED; no numbers from an instrument "
                         "that has not shown it can fire.")
    import phase2_embargo as EMB

    PA.assert_modules_under_root()
    PA.pin_data_root()
    PA.assert_tape_is_v5()
    _v = PA.assert_gate_passed()
    PA.assert_verdict_subject_is(PA.TAPE_PATH, _v)
    ident = PA._tape_identity()
    print(f"  identity: tape {ident['tape_sha256_prefix']} fragment "
          f"{ident['fragment_sha256_prefix']} topup {ident['topup_sha256_prefix']}",
          flush=True)

    print("  indexing train split...", flush=True)
    TP = PA.tape_index("train")
    FIT = PA._feature_pass(PA.FRAGMENT, "fragment", TAPE=TP)
    del TP
    print("  indexing score split for the embargo boundary...", flush=True)
    SP = PA.tape_index("score")
    probe = [{"t0": v["t0"], "t_start": v["t_start"]} for v in SP.values()]
    for coin in list(FIT):
        if not FIT[coin]["kept"]:
            continue
        before = len(FIT[coin]["kept"])
        kept, _ = EMB.purge_training(FIT[coin]["kept"], probe)
        keys = {(r["slug"], r["side"], r["gen"], r["t_start"]) for r in kept}
        keep = [n for n, r in enumerate(FIT[coin]["kept"])
                if (r["slug"], r["side"], r["gen"], r["t_start"]) in keys]
        for fam in ("PM", "FN", "ST"):
            FIT[coin][fam] = [FIT[coin][fam][n] for n in keep]
        FIT[coin]["kept"] = [FIT[coin]["kept"][n] for n in keep]
        print(f"  [purge/{coin}] {before:,} -> {len(FIT[coin]['kept']):,}",
              flush=True)
    EVAL = PA._feature_pass(PA.TOPUP, "topup", TAPE=SP)
    del SP, probe

    out = {"artifact": "iter011_conditional_value_v1",
           "receipt_family": RECEIPT_FAMILY,
           "preregistration": "ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md",
           "preregistration_commit": "3b71d3e",
           "preregistration_status": "FROZEN by user ruling R-232",
           "standalone": {
               "is_in_identity_lattice": "phase2_iter011_run.py" in
                                         PA.CODE_IDENTITY_FILES,
               "modifies_lattice_files": False,
               "consequence": "no four-arm fit is invalidated by this run; the "
                              "annotation-merge wiring (option (a)) waits for "
                              "the next four-arm cycle rather than being "
                              "dragged in by 011"},
           "arms": list(I11.ARMS_011), "heads": list(I11.HEADS_011),
           "budgets": list(I11.BUDGETS_011),
           "declared_family": I11.declared_family(),
           "incumbent_null_applicability": incumbent_null_applicability(),
           "identity": ident, "populations": {}, "results": {}}

    for coin in ("btc", "eth"):
        if coin not in FIT or not FIT[coin]["kept"] or coin not in EVAL:
            continue
        Xf = [build_design(FIT[coin], i) for i in range(len(FIT[coin]["kept"]))]
        Xe = [build_design(EVAL[coin], i) for i in range(len(EVAL[coin]["kept"]))]
        tgf = head_targets(FIT[coin]["kept"])
        tge = head_targets(EVAL[coin]["kept"])
        popsf = I11.head_populations(FIT[coin]["kept"])
        popse = I11.head_populations(EVAL[coin]["kept"])
        out["populations"][coin] = {
            "fit": tgf["counts"], "eval": tge["counts"],
            "fit_zero_mass": I11.zero_mass_diagnostic(popsf),
            "eval_zero_mass": I11.zero_mass_diagnostic(popse),
            "fit_empirical": I11.empirical_heads(popsf),
            "eval_empirical": I11.empirical_heads(popse),
            "eval_population_and_reach":
                PA.population_reach_disclosure(EVAL[coin]["kept"])}
        print(f"  [{coin}] fit {tgf['counts']} eval {tge['counts']}", flush=True)
        out["results"][coin] = {}
        for arm in I11.ARMS_011:
            t0 = time.time()
            fr = fit_arm(arm, Xf, tgf)
            ap = apply_arm(fr, Xe, tge)
            rep = report_arm(ap, tge)
            rep["fit_seconds"] = round(time.time() - t0, 1)
            rep["evaluation"] = "OUT-OF-SAMPLE within development"
            out["results"][coin][arm] = rep
            h = rep["heads"]
            print(f"  [{coin}/{arm}] Q1 auc {h['Q1_arrival'].get('auc')} | "
                  f"Q2 auc {h['Q2_sign'].get('auc')} | "
                  f"Q3h slope {h['Q3_m_harm'].get('calibration_slope')} | "
                  f"Q3g slope {h['Q3_m_good'].get('calibration_slope')} | "
                  f"underpowered {rep['underpowered_heads']}", flush=True)
            del fr, ap
        del Xf, Xe, FIT[coin]["PM"], FIT[coin]["FN"], FIT[coin]["ST"]

    out["development_evidence"] = {
        "is_a_validation": False,
        "computed_from": "per-coin eval_population_and_reach, which derives "
                         "is_a_validation from the declared population label "
                         "and the complete-UTC-day count against rule 11's bar",
        "statement": "DEVELOPMENT EVIDENCE. Both the fitting and the evaluation "
                     "populations are development; the evaluation is "
                     "out-of-sample relative to the fit but neither is a "
                     "validation set. Selection and validation require later "
                     "untouched complete UTC days (R-232 9.4, per coin)."}
    with OUT.open("w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
        fh.flush(); os.fsync(fh.fileno())
    print(f"\nWROTE {OUT.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
