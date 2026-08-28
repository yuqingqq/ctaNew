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
    y_pos = [1 if v[i] > 0 else 0 for i in idx_prev]
    y_neg = [1 if v[i] < 0 else 0 for i in idx_prev]   # A1.1 Option 1: ESTIMATED
    idx_h = [i for i in idx_prev if v[i] > 0]
    idx_g = [i for i in idx_prev if v[i] < 0]
    idx_z = [i for i in idx_prev if v[i] == 0.0]
    return {"y_fill": y_fill, "idx_prev": idx_prev,
            "y_pos": y_pos, "y_neg": y_neg,
            "idx_harm": idx_h, "idx_good": idx_g, "idx_zero": idx_z,
            "m_harm_target": [v[i] for i in idx_h],
            "m_good_target": [-v[i] for i in idx_g],
            "counts": {"n_rows": len(rows), "n_preventable": len(idx_prev),
                       "n_v_positive": len(idx_h), "n_v_negative": len(idx_g),
                       "n_v_zero": len(idx_z)}}


def fit_arm(arm: str, X, tg: dict, seed_note: str = "") -> dict:
    """Fit the heads for one arm. Q4 is composed, never fitted.

    HEADS ARE FITTED ON THEIR CONDITIONAL POPULATIONS -- that is what the
    estimands say -- but they are FUNCTIONS OF x and must PREDICT on every
    action. The previous version predicted only on the training subset, so the
    four vectors had different lengths (the reviewer measured 3/2/1/1) and
    action-time Q4 could not compose at all. Fitting domain and prediction
    domain are different things and are now kept separate."""
    import harmful_fast_compute as fc
    if arm not in I11.ARMS_011:
        raise RuntimeError(f"REFUSED: unknown arm {arm!r}; the frozen "
                           f"preregistration declares {I11.ARMS_011}")
    MIN = I11.UNDERPOWERED_MIN_N
    ip, ih, ig = tg["idx_prev"], tg["idx_harm"], tg["idx_good"]

    if arm == "composed_linear":
        Z, mu, sd = fc.fast_zscale(X, X)
        def sub(idx):
            return [Z[i] for i in idx]
        w_fill = fc.fast_fit_logistic_w(Z, tg["y_fill"], [1.0] * len(Z))
        w_pos = (fc.fast_fit_logistic_w(sub(ip), tg["y_pos"], [1.0] * len(ip))
                 if len(ip) >= MIN and len(set(tg["y_pos"])) > 1 else None)
        w_neg = (fc.fast_fit_logistic_w(sub(ip), tg["y_neg"], [1.0] * len(ip))
                 if len(ip) >= MIN and len(set(tg["y_neg"])) > 1 else None)
        wmh = (fc.fast_fit_ridge_w(sub(ih), tg["m_harm_target"],
                                   [1.0] * len(ih), lam=10.0)
               if len(ih) >= MIN else None)
        wmg = (fc.fast_fit_ridge_w(sub(ig), tg["m_good_target"],
                                   [1.0] * len(ig), lam=10.0)
               if len(ig) >= MIN else None)
        return {"arm": arm,
                "model": {"kind": "linear", "mu": mu, "sd": sd,
                          "w_fill": w_fill, "w_pos": w_pos, "w_neg": w_neg,
                          "wmh": wmh, "wmg": wmg},
                "fitted": {"p_fill": True, "p_pos": w_pos is not None,
                           "p_neg": w_neg is not None,
                           "m_harm": wmh is not None, "m_good": wmg is not None},
                "model_class": "logistic (arrival/sign) + ridge lam=10 "
                               "(magnitudes); A1.6 pinned"}

    import lightgbm as lgb
    import numpy as np
    A = np.asarray(X, dtype=np.float64)
    clf = lgb.LGBMClassifier(**D.LGBM_PARAMS)
    clf.fit(A, np.asarray(tg["y_fill"]))
    def cls(idx, y):
        if len(idx) < MIN or len(set(y)) < 2:
            return None
        m = lgb.LGBMClassifier(**D.LGBM_PARAMS)
        m.fit(A[idx], np.asarray(y))
        return m
    def reg(idx, t):
        if len(idx) < MIN:
            return None
        m = lgb.LGBMRegressor(**D.LGBM_VALUE_PARAMS)
        m.fit(A[idx], np.asarray(t))
        return m
    return {"arm": arm,
            "model": {"kind": "lgbm", "clf": clf,
                      "c_pos": cls(ip, tg["y_pos"]),
                      "c_neg": cls(ip, tg["y_neg"]),
                      "r_harm": reg(ih, tg["m_harm_target"]),
                      "r_good": reg(ig, tg["m_good_target"])},
            "fitted": {"p_fill": True,
                       "p_pos": cls(ip, tg["y_pos"]) is not None,
                       "p_neg": cls(ip, tg["y_neg"]) is not None,
                       "m_harm": reg(ih, tg["m_harm_target"]) is not None,
                       "m_good": reg(ig, tg["m_good_target"]) is not None},
            "model_class": "LGBM classifier (arrival/sign) + LGBM regressor "
                           "(magnitudes), params PINNED (A1.6)"}


def apply_arm(fitres: dict, X, tg: dict) -> dict:
    """Predict EVERY head on EVERY action. Row-aligned by construction.

    All five vectors have length len(X). A head that could not be fitted
    predicts a declared NEUTRAL value, recorded as such -- so composition is
    always possible and a missing head is visible in the receipt rather than
    silently shortening a vector."""
    import harmful_fast_compute as fc
    m = fitres["model"]
    n = len(X)
    if m["kind"] == "linear":
        mu, sd = m["mu"], m["sd"]
        Z = [[1.0] + [(v[i] - mu[i]) / sd[i] for i in range(len(mu))] for v in X]
        def pl(w):
            return [fc.fast_predict_p(w, z) for z in Z] if w else None
        def rl(w):
            return ([float(sum(a * b for a, b in zip(w, z))) for z in Z]
                    if w else None)
        p_fill, p_pos, p_neg = pl(m["w_fill"]), pl(m["w_pos"]), pl(m["w_neg"])
        m_harm, m_good = rl(m["wmh"]), rl(m["wmg"])
    else:
        import numpy as np
        A = np.asarray(X, dtype=np.float64)
        p_fill = m["clf"].predict_proba(A)[:, 1].tolist()
        p_pos = (m["c_pos"].predict_proba(A)[:, 1].tolist() if m["c_pos"] else None)
        p_neg = (m["c_neg"].predict_proba(A)[:, 1].tolist() if m["c_neg"] else None)
        m_harm = m["r_harm"].predict(A).tolist() if m["r_harm"] else None
        m_good = m["r_good"].predict(A).tolist() if m["r_good"] else None

    NEUTRAL = {"p_pos": 0.0, "p_neg": 0.0, "m_harm": 0.0, "m_good": 0.0}
    unfitted = []
    out = {"p_fill": p_fill}
    for k, v in (("p_pos", p_pos), ("p_neg", p_neg),
                 ("m_harm", m_harm), ("m_good", m_good)):
        if v is None:
            unfitted.append(k)
            out[k] = [NEUTRAL[k]] * n
        else:
            out[k] = v
    lens = {k: len(v) for k, v in out.items()}
    if len(set(lens.values())) != 1 or set(lens.values()) != {n}:
        raise RuntimeError(
            f"REFUSED: head vectors are not row-aligned: {lens} against "
            f"{n} actions. Unaligned heads cannot compose an action-time Q4 — "
            f"this is the defect the reviewer measured as lengths 3/2/1/1.")
    out["expected_cancel_value"] = [
        I11.compose_expected_cancel_value(out["p_fill"][i], out["p_pos"][i],
                                          out["p_neg"][i], out["m_harm"][i],
                                          out["m_good"][i]) for i in range(n)]
    out["p_zero_implied"] = [I11.implied_p_zero(out["p_pos"][i], out["p_neg"][i])
                             for i in range(n)]
    out.update({"arm": fitres["arm"], "n_actions": n,
                "unfitted_heads_neutralised": unfitted,
                "neutral_values": NEUTRAL,
                "model_class": fitres["model_class"],
                "fitted": fitres["fitted"],
                "evaluation": "OUT-OF-SAMPLE within development",
                "row_aligned": True})
    return out


def report_arm(pred: dict, tg: dict) -> dict:
    """All heads, always, including failures (prereg §3).

    Each head's METRIC is computed on ITS OWN population — where its labels
    exist — while its PREDICTIONS span every action so Q4 can compose. Those are
    different domains and conflating them is what produced the unaligned
    vectors."""
    ip, ih, ig = tg["idx_prev"], tg["idx_harm"], tg["idx_good"]
    q1 = I11.head_report("Q1_arrival", "probability",
                         pred["p_fill"], tg["y_fill"])
    q2p = I11.head_report("Q2_p_pos", "probability",
                          [pred["p_pos"][i] for i in ip], tg["y_pos"])
    q2n = I11.head_report("Q2_p_neg", "probability",
                          [pred["p_neg"][i] for i in ip], tg["y_neg"])
    q3h = I11.head_report("Q3_m_harm", "magnitude",
                          [pred["m_harm"][i] for i in ih], tg["m_harm_target"])
    q3g = I11.head_report("Q3_m_good", "magnitude",
                          [pred["m_good"][i] for i in ig], tg["m_good_target"])
    # A1.4: Q2's adjudicated statistic is AUC. With Option 1 there are two sign
    # heads; the cell takes the WORSE of the two, because a decomposition whose
    # negative side is uninformative has not established sign discrimination
    # even if its positive side has.
    aucs = [h["auc"] for h in (q2p, q2n) if h.get("auc") is not None]
    q2_cell = min(aucs) if aucs else None
    heads = {"Q1_arrival": q1, "Q2_p_pos": q2p, "Q2_p_neg": q2n,
             "Q3_m_harm": q3h, "Q3_m_good": q3g}
    under = sorted(k for k, v in heads.items()
                   if v.get("status") == I11.UNDERPOWERED)
    return {
        "arm": pred["arm"], "model_class": pred["model_class"],
        "heads": heads,
        "all_heads_reported": True,
        "underpowered_heads": under, "any_underpowered": bool(under),
        "unfitted_heads_neutralised": pred["unfitted_heads_neutralised"],
        "n_actions": pred["n_actions"], "row_aligned": pred["row_aligned"],
        "adjudicated_statistics": {
            "Q1_arrival": q1.get("auc"),
            "Q2_sign": q2_cell,
            "Q2_cell_rule": "min(AUC of p_pos, AUC of p_neg) — the WORSE side, "
                            "because a decomposition with an uninformative "
                            "negative side has not established sign "
                            "discrimination",
            "Q3_magnitudes": min([v for v in (q3h.get("calibration_slope"),
                                              q3g.get("calibration_slope"))
                                  if v is not None], default=None),
            "Q3_cell_rule": "min |calibration slope deviation| side reported; "
                            "both slopes carried",
        },
        "reporting_rule": "all heads reported whether or not they pass; a "
                          "strong arrival head does not establish toxicity "
                          "discrimination (parent §2.1)",
        "advancement_rule": "Q4 alone may NOT advance a candidate; explicit "
                            "user sign-off required (R-232 9.2)",
    }


def selftest() -> int:
    """Known-bads BEFORE numbers. Drives both arms on synthetic data."""
    fails = []

    def ok(c, label):
        print(f"  {'PASS' if c else 'FAIL'}  {label}")
        if not c:
            fails.append(label)

    L = str(D.TARGET_LATENCY_MS)

    def row(v, fill=True, shares=1.0):
        """A1.3: rows must be CONSISTENT. A no-fill row carries value 0 as well
        as zero shares — the previous helper zeroed shares while keeping a
        non-zero value, which is now correctly refused as VALUE_WITHOUT_SHARES."""
        return {"any_fill_ahead": fill,
                "latency": {L: {"preventable_value_cents": v if fill else 0.0,
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
            pr = apply_arm(fr, X, tg)
            rep = report_arm(pr, tg)
            n = len(rows)
            ok(all(len(pr[k]) == n for k in
                   ("p_fill", "p_pos", "p_neg", "m_harm", "m_good",
                    "expected_cancel_value")),
               f"{arm}: EVERY head predicts on EVERY action — row-aligned "
               f"(the reviewer measured 3/2/1/1 before this)")
            ok(rep["all_heads_reported"], f"{arm}: all heads reported")
            ok(rep["heads"]["Q2_p_pos"]["n"] == c["n_preventable"],
               f"{arm}: Q2 is SCORED on the preventable base while PREDICTING "
               f"on all actions — different domains, kept separate")
            ok(rep["heads"]["Q3_m_harm"]["n"] == c["n_v_positive"],
               f"{arm}: Q3 m_harm is scored on V>0 only")
            ok(isinstance(pr["expected_cancel_value"][0], float),
               f"{arm}: Q4 composes per action (never fitted)")
            ok(rep["adjudicated_statistics"]["Q2_cell_rule"].startswith("min("),
               f"{arm}: Q2's cell takes the WORSE sign head (A1.4)")
        except Exception as e:
            ok(False, f"{arm}: fits and reports ({type(e).__name__}: {e})")

    try:
        fit_arm("composed_magic", X, tg)
        ok(False, "an UNDECLARED arm is REFUSED")
    except RuntimeError as e:
        ok("frozen preregistration" in str(e), "an UNDECLARED arm is REFUSED")

    thin = [row(5.0)] * 3 + [row(-4.0)] * 3
    tgt = head_targets(thin)
    Xt = [[1.0, 0.5]] * 6
    frt = fit_arm("composed_linear", Xt, tgt)
    prt = apply_arm(frt, Xt, tgt)
    rept = report_arm(prt, tgt)
    ok(not frt["fitted"]["m_harm"] and not frt["fitted"]["m_good"],
       "a THIN magnitude population is not fitted (min_n respected)")
    ok(set(prt["unfitted_heads_neutralised"]) >= {"m_harm", "m_good"},
       "unfitted heads are NEUTRALISED and named, so composition still works "
       "and the absence is visible rather than shortening a vector")
    ok(all(len(prt[k]) == 6 for k in
           ("p_fill", "p_pos", "p_neg", "m_harm", "m_good",
            "expected_cancel_value")),
       "row alignment holds even when heads could not be fitted")
    ok(set(rept["underpowered_heads"]) >= {"Q3_m_harm", "Q3_m_good"},
       "and the unfitted magnitude heads are REPORTED UNDERPOWERED, not dropped")

    import tempfile as _tf
    _d = Path(_tf.mkdtemp())
    try:
        assert_outputs_written((_d / "never_written.json",), argv=["prog"])
        ok(False, "a run that writes NOTHING must not exit 0")
    except RuntimeError as e:
        ok("silent-success" in str(e),
           "a run that writes NOTHING must not exit 0 (the clean-exit-no-output "
           "shape is refused by name)")
    _z = _d / "z.json"; _z.write_text("")
    try:
        assert_outputs_written((_z,), argv=["prog"]); ok(False, "a ZERO-BYTE output is refused")
    except RuntimeError as e:
        ok("does not parse" in str(e), "a ZERO-BYTE output is refused")
    _e = _d / "e.json"; _e.write_text("{}")
    try:
        assert_outputs_written((_e,), argv=["prog"]); ok(False, "an EMPTY object is refused")
    except RuntimeError as e:
        ok("artifact identity" in str(e),
           "an EMPTY object is refused (existence alone is not a result)")
    ok(assert_outputs_written((_d / "nope2.json",),
                              argv=["prog", "--selftest"]).get("exempt")
       == "--selftest",
       "an EXPLICIT --selftest run is EXEMPT (a selftest writes no artifact "
       "by design; the guard must not punish it)")
    try:
        assert_outputs_written((_d / "nope3.json",), argv=["prog"])
        ok(False, "the exemption is NOT reachable without the declared flag")
    except RuntimeError:
        ok(True, "the exemption is NOT reachable without the declared flag — "
                 "it keys on the MODE, not on the outputs being missing")
    try:
        assert_outputs_written((_d / "nope4.json",),
                               argv=["prog", "--stage-fit", "--selftestish"])
        ok(False, "a LOOKALIKE flag does not grant the exemption")
    except RuntimeError:
        ok(True, "a LOOKALIKE flag (--selftestish) does not grant the exemption")
    _g = _d / "g.json"; _g.write_text(json.dumps({"artifact": "x", "a": 1}))
    ok(assert_outputs_written((_g,), argv=["prog"])[_g.name]["artifact"] == "x",
       "a well-formed output PASSES (the guard is not a wall). NOTE: every "
       "falsifier of the ENFORCING path passes an explicit non-selftest argv, "
       "because these run UNDER --selftest and would otherwise inherit the "
       "exemption and test nothing.")

    # ---------------------------------------------------------- STEP 5 ---
    ok(assert_frozen_constants()["verified"],
       "A1.6 the FROZEN constants match the code that will run")
    _sv = I11.UNDERPOWERED_MIN_N
    try:
        I11.UNDERPOWERED_MIN_N = 7
        assert_frozen_constants()
        ok(False, "a DRIFTED constant is REFUSED")
    except RuntimeError as _e:
        ok("drifted from the FROZEN" in str(_e),
           "a DRIFTED constant is REFUSED (frozen in a document is not the "
           "same as true in the process)")
    finally:
        I11.UNDERPOWERED_MIN_N = _sv
    _si = assert_standalone_identity()
    ok(_si["in_identity_lattice"] is False and _si["runner_sha256_prefix"],
       "A1.6 the standalone identity is MEASURED (both module shas) and "
       "recorded, not asserted in prose")
    _svl = PA.CODE_IDENTITY_FILES
    try:
        PA.CODE_IDENTITY_FILES = tuple(_svl) + ("phase2_iter011_run.py",)
        assert_standalone_identity()
        ok(False, "an 011 module INSIDE the lattice is REFUSED")
    except RuntimeError as _e:
        ok("standalone property" in str(_e),
           "an 011 module INSIDE the lattice is REFUSED — if that changes it "
           "must be a decision, not a drift")
    finally:
        PA.CODE_IDENTITY_FILES = _svl

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

DECLARED_OUTPUTS = (OUT,)
SELFTEST_FLAG = "--selftest"


def is_selftest_mode(argv=None) -> bool:
    """Declared selftest mode. A selftest CORRECTLY writes no artifact.

    The exemption is tied to the DECLARED MODE, never to 'the outputs happen to
    be missing' — the latter would be a bypass that excuses exactly the failure
    the guard exists to catch. In every other mode the guard applies in full,
    including when selftest() has been called internally as the run's gate."""
    return SELFTEST_FLAG in (sys.argv if argv is None else argv)


def assert_outputs_written(outputs=DECLARED_OUTPUTS, argv=None) -> dict:
    """A run that writes NOTHING must not exit 0.

    The silent-success shape: a clean exit obtained by not doing the work. It is
    indistinguishable from a completed run at the exit code, and the exit code
    is what an operator reads first. Every declared output must exist, parse,
    and carry its artifact identity — existence alone would pass a zero-byte
    file, and parsing alone would pass an empty object."""
    if is_selftest_mode(argv):
        return {"exempt": SELFTEST_FLAG,
                "why": "a selftest run writes no artifact BY DESIGN; the "
                       "exemption is on the DECLARED MODE, not on the outputs "
                       "being absent"}
    ev = {}
    for o in outputs:
        if not o.exists():
            raise RuntimeError(
                f"REFUSED: declared output {o.name} was never written, yet the "
                f"run reached its exit. A clean exit that produced nothing is "
                f"the silent-success shape: it looks identical to a completed "
                f"run at the exit code.")
        b = o.stat().st_size
        try:
            d = json.loads(o.read_text())
        except ValueError as e:
            raise RuntimeError(
                f"REFUSED: declared output {o.name} does not parse ({e}); a "
                f"file that exists is not a result.")
        if not isinstance(d, dict) or not d.get("artifact"):
            raise RuntimeError(
                f"REFUSED: {o.name} carries no artifact identity; an empty "
                f"object would otherwise pass an existence check.")
        ev[o.name] = {"bytes": b, "artifact": d["artifact"],
                      "top_level_keys": len(d)}
    return ev



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
    print(f"  train split indexed: {len(TP):,} rows", flush=True)
    FIT = PA._feature_pass(PA.FRAGMENT, "fragment", TAPE=TP)
    del TP
    print("  indexing score split for the embargo boundary...", flush=True)
    SP = PA.tape_index("score")
    print(f"  score split indexed: {len(SP):,} rows", flush=True)
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
    ev = assert_outputs_written()
    print(f"\nWROTE {OUT.name}: {ev[OUT.name]}", flush=True)
    return 0



# ---------------------------------------------------------------- STEP 5 ---
# Identity and run guards. A1.6, FROZEN.

FROZEN_CONSTANTS = {
    "UNDERPOWERED_MIN_N": 100,
    "ridge_lam": 10.0,
    "n_perm": 2000,
    "perm_seed": 20260828,
    "arms": ("composed_linear", "composed_lgbm"),
    "heads": ("Q1_arrival", "Q2_sign", "Q3_magnitudes", "Q4_combined_ev"),
    "family_size": 24,
    "declared_output": "iter011_conditional_value_v1.json",
}


def assert_frozen_constants() -> dict:
    """The A1.6 constants must match the code that will run.

    Frozen in a document is not the same as true in the process. Every constant
    below is read from the module that uses it, so a drift is a REFUSAL rather
    than a discrepancy nobody compares."""
    live = {
        "UNDERPOWERED_MIN_N": I11.UNDERPOWERED_MIN_N,
        "ridge_lam": 10.0,                      # asserted at its call sites
        "n_perm": I11.N_PERM_011,
        "perm_seed": I11.PERM_SEED_011,
        "arms": tuple(I11.ARMS_011),
        "heads": tuple(I11.HEADS_011),
        "family_size": I11.declared_family()["n_cells"],
        "declared_output": OUT.name,
    }
    bad = {k: (FROZEN_CONSTANTS[k], live[k]) for k in FROZEN_CONSTANTS
           if FROZEN_CONSTANTS[k] != live[k]}
    if bad:
        raise RuntimeError(
            f"REFUSED: code drifted from the FROZEN A1.6 constants "
            f"(frozen, live): {bad}. A constant frozen in a document and a "
            f"constant used by the process are different things until one is "
            f"compared to the other.")
    if "ridge_lam=10.0" not in inspect_ridge_calls():
        raise RuntimeError(
            "REFUSED: a ridge call does not use the frozen lam=10.0.")
    return {"frozen": dict(FROZEN_CONSTANTS), "verified": True}


def inspect_ridge_calls() -> str:
    """Every fast_fit_ridge_w call site, as text, so lam can be checked."""
    import inspect as _i
    src = _i.getsource(fit_arm)
    return "".join(
        f"ridge_lam={m}" for m in
        [t.split("lam=")[1].split(")")[0].strip()
         for t in src.split("fast_fit_ridge_w")[1:] if "lam=" in t])


def assert_standalone_identity() -> dict:
    """This runner is NOT in the identity lattice, and modifies none of it.

    Recorded in the receipt so the property is a claim a reader can check, not
    an assurance in a commit message."""
    import hashlib
    me = Path(__file__).resolve()
    lib = me.parent / "phase2_iter011.py"
    if me.name in PA.CODE_IDENTITY_FILES or lib.name in PA.CODE_IDENTITY_FILES:
        raise RuntimeError(
            f"REFUSED: an 011 module is inside CODE_IDENTITY_FILES. The "
            f"standalone property is what keeps a four-arm fit from being "
            f"invalidated by 011 work; if that changes it must be a decision, "
            f"not a drift.")
    def sha(f):
        return hashlib.sha256(f.read_bytes()).hexdigest()[:16]
    return {"runner": me.name, "runner_sha256_prefix": sha(me),
            "library": lib.name, "library_sha256_prefix": sha(lib),
            "in_identity_lattice": False,
            "modifies_lattice_files": False,
            "lattice_size": len(PA.CODE_IDENTITY_FILES),
            "consequence": "no four-arm fit is invalidated by this run; the "
                           "annotation-merge wiring waits for the next "
                           "four-arm cycle"}


if __name__ == "__main__":
    # BELT AND BRACES: exit 0 REQUIRES the declared outputs. main() asserts them
    # too; this catches an early `return 0` added later, which is exactly how a
    # silent success gets introduced by a well-meaning edit.
    _rc = main()
    if _rc == 0:
        assert_outputs_written(argv=sys.argv)
    raise SystemExit(_rc)
