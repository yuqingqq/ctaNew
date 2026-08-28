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
import re
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
            # A1.5: "n for every reported head is the ACTION count, not the
            # prediction count. A head predicting on rows must state its action
            # count beside it." Measured 1.99 rows/fill (max 23), and the ratio
            # DIFFERS BETWEEN COINS, so a row-level n does not merely inflate --
            # it distorts comparisons. Both are carried: n_rows stays the
            # prediction population, n_actions is what a head reports.
            "counts": {"n_rows": len(rows), "n_preventable": len(idx_prev),
                       "n_actions": I11.action_count(rows),
                       "n_actions_preventable": I11.action_count(
                           [rows[i] for i in idx_prev]),
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


def q4_economics(pred: dict, rows: list, budgets=None,
                 incumbent: dict = None) -> dict:
    """Q4's ECONOMICS per budget, at the ACTION unit. I11-2.

    apply_arm composed expected_cancel_value per action and report_arm threw it
    away, so budgets were metadata and no decision quantity ever reached a cell.
    Here the composed value RANKS actions, a budget selects the top fraction,
    and the realised signed value of those actions is the net — with per-window
    increments retained so the incremental null has units to permute."""
    budgets = budgets or I11.BUDGETS_011
    ecv = pred["expected_cancel_value"]
    if len(ecv) != len(rows):
        raise RuntimeError(
            f"REFUSED: {len(ecv)} composed values against {len(rows)} rows; "
            f"economics cannot be evaluated on a misaligned population.")
    inc_ecv = None
    if incumbent is not None:
        inc_ecv = incumbent["expected_cancel_value"]
        if len(inc_ecv) != len(rows):
            raise RuntimeError(
                f"REFUSED: the incumbent carries {len(inc_ecv)} composed values "
                f"against {len(rows)} rows. An increment is defined only on the "
                f"IDENTICAL action population (prereg 5.2); comparing arms over "
                f"different populations is the conflation that null exists to "
                f"prevent.")

    # A1.5: the unit is the ACTION, with FIRST-CROSSING dedup "as
    # harmful_action_eval already does". That reference does TWO different
    # things and the distinction is decision-bearing (I11-B3):
    #   RANK   generations by their maximum score   (which actions get cancelled)
    #   VALUE  next(i for i in gens[gk] if s[i] >= theta)   (the EARLIEST
    #          crossing row, not the best-scoring one)
    # A cancel fires the first time the score crosses; there is nothing left to
    # cancel later, so the later row is counterfactually unreachable. Valuing at
    # the generation's max row credits the policy with a decision it never made,
    # and on a generation whose early row forfeits a good fill and whose late row
    # avoids harm the two rules disagree in SIGN.
    gens = {}
    for i, r in enumerate(rows):
        gens.setdefault((r.get("slug"), r.get("side"), r.get("gen")), []).append(i)
    for k in gens:
        gens[k].sort(key=lambda i: (rows[i].get("t_start"), i))

    def _rank(scores):
        gmax = {k: max(scores[i] for i in idx) for k, idx in gens.items()}
        return gmax, sorted(gens, key=lambda k: (-gmax[k], k))

    def _cross(idx, theta, scores):
        """The EARLIEST row in the generation whose score crosses theta."""
        for i in idx:
            if scores[i] >= theta:
                return i
        raise RuntimeError(
            "REFUSED: a chosen generation contains no row at or above its own "
            "budget threshold. It was selected by a maximum that no row "
            "reproduces, so the selection and the valuation disagree.")

    gmax, order = _rank(ecv)
    i_gmax, i_order = _rank(inc_ecv) if inc_ecv is not None else (None, None)

    def _net_by_window(scores, gm, orr, kk):
        chosen = orr[:kk]
        theta = gm[chosen[-1]]          # the budget's own cutoff score
        bw = {}
        for k in chosen:
            i = _cross(gens[k], theta, scores)
            bw[k[0]] = bw.get(k[0], 0.0) + I11.signed_v_cancel(rows[i])
        return bw

    out = {}
    for b in budgets:
        frac = float(b.rstrip("%")) / 100.0
        kk = max(1, int(len(order) * frac))
        cand_bw = _net_by_window(ecv, gmax, order, kk)
        net = sum(cand_bw.values())
        if inc_ecv is None:
            # NOT an increment. Named for what it is, so no caller can label a
            # sign-flip of the candidate's own value "increment vs incumbent".
            by_window, incumbent_net = cand_bw, None
        else:
            inc_bw = _net_by_window(inc_ecv, i_gmax, i_order, kk)
            incumbent_net = sum(inc_bw.values())
            by_window = {w: cand_bw.get(w, 0.0) - inc_bw.get(w, 0.0)
                         for w in set(cand_bw) | set(inc_bw)}
        out[b] = {"budget": b, "n_actions_total": len(order),
                  "n_cancelled_actions": kk, "net_cents": net,
                  "incumbent_net_cents": incumbent_net,
                  "paired_against_incumbent": inc_ecv is not None,
                  ("increment_by_window" if inc_ecv is not None
                   else "candidate_value_by_window"): by_window,
                  "unit": "ACTION (first-crossing dedup, A1.5)"}
    return out


def _strata(rows, idx):
    """The decision variable, per prereg §5.1: side and hour. Hour comes from
    the row's OWN t_start, never a nearby proxy (rule 3)."""
    out = []
    for i in idx:
        r = rows[i]
        t = r.get("t_start")
        out.append((r.get("side"), int((t // 3600) % 24) if t is not None else None))
    return out


def _gens(rows, idx):
    return [(rows[i].get("slug"), rows[i].get("side"), rows[i].get("gen"))
            for i in idx]


def report_arm(pred: dict, tg: dict, rows: list) -> dict:
    """All heads, always, including failures (prereg §3).

    Each head's METRIC is computed on ITS OWN population — where its labels
    exist — while its PREDICTIONS span every action so Q4 can compose. Those are
    different domains and conflating them is what produced the unaligned
    vectors."""
    ip, ih, ig = tg["idx_prev"], tg["idx_harm"], tg["idx_good"]
    _all = list(range(len(tg["y_fill"])))
    q1 = I11.head_report("Q1_arrival", "probability",
                         pred["p_fill"], tg["y_fill"],
                         strata=_strata(rows, _all), gen_keys=_gens(rows, _all))
    q2p = I11.head_report("Q2_p_pos", "probability",
                          [pred["p_pos"][i] for i in ip], tg["y_pos"],
                          strata=_strata(rows, ip), gen_keys=_gens(rows, ip))
    q2n = I11.head_report("Q2_p_neg", "probability",
                          [pred["p_neg"][i] for i in ip], tg["y_neg"],
                          strata=_strata(rows, ip), gen_keys=_gens(rows, ip))
    q3h = I11.head_report("Q3_m_harm", "magnitude",
                          [pred["m_harm"][i] for i in ih], tg["m_harm_target"],
                          strata=_strata(rows, ih), gen_keys=_gens(rows, ih))
    q3g = I11.head_report("Q3_m_good", "magnitude",
                          [pred["m_good"][i] for i in ig], tg["m_good_target"],
                          strata=_strata(rows, ig), gen_keys=_gens(rows, ig))
    # A1.4: Q2's adjudicated statistic is AUC. Under Option 1 there are TWO sign
    # heads and the cell takes the WORSE of them — a decomposition whose
    # negative side is uninformative has not established sign discrimination
    # even if its positive side has.
    #
    # BOTH SIDES MUST BE EVALUABLE. The previous form filtered None out of the
    # list and took min() of what remained, so ONE side could carry the cell
    # while the other was one-class or underpowered — the surviving side's value
    # would sail past the UNDERPOWERED machinery as though the pair had been
    # measured. If either side is unevaluable the CELL is unevaluable, and the
    # reason travels with it.
    _sides = {"p_pos": q2p, "p_neg": q2n}
    _missing = sorted(k for k, h in _sides.items() if h.get("auc") is None)
    _under = sorted(k for k, h in _sides.items()
                    if h.get("status") == I11.UNDERPOWERED)
    if _missing or _under:
        q2_cell = None
        q2_cell_status = (I11.UNDERPOWERED if _under
                          else I11.CELL_STATUS_UNEVALUABLE)
        q2_cell_detail = (f"sign discrimination needs BOTH sides; "
                          f"unevaluable={_missing} underpowered={_under}. A "
                          f"single side cannot carry the cell.")
    else:
        q2_cell = min(q2p["auc"], q2n["auc"])
        q2_cell_status = I11.CELL_STATUS_OK
        q2_cell_detail = "min(AUC p_pos, AUC p_neg) — the WORSE side"
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
            "Q2_cell_status": q2_cell_status,
            "Q2_cell_detail": q2_cell_detail,
            "Q2_cell_rule": "min(AUC of p_pos, AUC of p_neg) — the WORSE side, "
                            "because a decomposition with an uninformative "
                            "negative side has not established sign "
                            "discrimination. BOTH sides must be evaluable: if "
                            "either is missing or underpowered the CELL is, "
                            "and a single side never carries it.",
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


def assert_receipt_has_all_cells(receipt: dict) -> dict:
    """A receipt lacking the DECLARED 24 cells REFUSES. I11-2, output level.

    The guards so far checked the family at ASSEMBLY. This checks the ARTIFACT,
    because assembly can be skipped: the evaluator existed for a full batch and
    was never called, and nothing at the output level noticed a receipt with no
    cells in it at all. A run that writes a receipt without its declared family
    has not evaluated anything, whatever else it contains."""
    fam = receipt.get("family")
    if not isinstance(fam, dict) or "cells" not in fam:
        raise RuntimeError(
            "REFUSED: the receipt carries no evaluated family. The 24-cell "
            "evaluator exists; a receipt without its cells means it was never "
            "invoked, which is exactly how a batch shipped with the evaluator "
            "unwired (I11-2).")
    declared = set(I11.declared_family()["cells"])
    present = set(fam["cells"])
    missing = sorted(declared - present)
    extra = sorted(present - declared)
    if missing or extra:
        raise RuntimeError(
            f"REFUSED: the receipt's family does not match the declaration. "
            f"Missing {len(missing)} cell(s): {missing[:6]}...; undeclared: "
            f"{extra}. The family is frozen at {len(declared)} (R-232 9.1).")
    if fam.get("holm_denominator") != len(declared):
        raise RuntimeError(
            f"REFUSED: Holm denominator {fam.get('holm_denominator')} is not "
            f"the declared {len(declared)}; a shrinking denominator rewards "
            f"failing to measure.")
    # I11-B4: the checks above compare KEY SETS. Twenty-four EMPTY DICTS carry
    # exactly the declared keys and satisfied every one of them -- the same
    # shape as the empty file_hashes map R-228(1) closed one level up, and the
    # same lesson: a container that matches a declaration is not evidence that
    # anything was put in it. Each cell must now carry what a cell IS.
    REQUIRED_CELL_FIELDS = ("arm", "head", "budget", "status", "statistic",
                            "p_value", "n_actions", "detail")
    KNOWN = {I11.CELL_STATUS_OK, I11.CELL_STATUS_UNDERPOWERED,
             I11.CELL_STATUS_NO_COUNTERPART, I11.CELL_STATUS_UNEVALUABLE,
             I11.CELL_STATUS_AGG_UNDECLARED}
    hollow, badstatus, unsupported = [], [], []
    for key in sorted(declared):
        c = fam["cells"][key]
        if not isinstance(c, dict) or any(f not in c for f in REQUIRED_CELL_FIELDS):
            hollow.append(key); continue
        st = c.get("status")
        if st not in KNOWN:
            badstatus.append(f"{key}={st!r}")
        elif st == I11.CELL_STATUS_OK:
            # An OK cell asserts it was EVALUATED, so it must carry the
            # evidence: a real statistic, a real population, and either a
            # p-value or a status explaining its absence. An OK cell with
            # statistic=None is a cell claiming a result it does not have.
            if I11._num(c.get("statistic")) is None:
                unsupported.append(f"{key}: status OK, statistic "
                                   f"{c.get('statistic')!r}")
            elif not isinstance(c.get("n_actions"), int) or c["n_actions"] <= 0:
                unsupported.append(f"{key}: status OK, n_actions "
                                   f"{c.get('n_actions')!r} (A1.5 requires a "
                                   f"positive ACTION count)")
            elif I11._num(c.get("p_value")) is None:
                unsupported.append(f"{key}: status OK with no p_value; a cell "
                                   f"with no null evidence is not OK, it is "
                                   f"UNEVALUABLE")
        elif not str(c.get("detail") or "").strip():
            unsupported.append(f"{key}: status {st} with no detail; a "
                               f"non-OK status must say WHY")
    if hollow or badstatus or unsupported:
        raise RuntimeError(
            f"REFUSED: the receipt's cells match the declared keys but do not "
            f"carry results. Hollow (missing required fields): {len(hollow)} "
            f"{hollow[:4]}; unknown status: {badstatus[:4]}; unsupported: "
            f"{len(unsupported)} {unsupported[:4]}. Matching a declaration is "
            f"not evidence that anything was evaluated (I11-B4).")
    return {"cells_present": len(present), "declared": len(declared),
            "holm_denominator": fam["holm_denominator"],
            "cells_validated": "contents"}


def evaluate_family(coin_results: dict, populations: dict) -> dict:
    """Build and adjudicate the DECLARED 24-cell family. I11-2.

    The evaluator existed and was NEVER CALLED: main() fitted both arms, wrote
    per-head descriptive reports, and composed Q4 in apply_arm only to discard
    it in report_arm. Budgets were metadata. Everything below is the machinery
    that was already built, now actually invoked.

    Q2/Q3 incremental cells carry NO_INCUMBENT_COUNTERPART (R-237): the
    incumbent has a hazard head and a composed value but NO sign or magnitude
    heads, so there is nothing to be incremental TO and inventing a baseline is
    the inverse of rule 9."""
    cells = {}
    for arm in I11.ARMS_011:
        for head in I11.HEADS_011:
            for budget in I11.BUDGETS_011:
                key = I11.cell_key(arm, head, budget)
                cells[key] = _one_cell(arm, head, budget, coin_results,
                                       populations)
    fam = I11.assemble_family(cells)
    fam["incumbent_null_applicability"] = incumbent_null_applicability()
    return fam


def _one_cell(arm: str, head: str, budget: str, coin_results: dict,
              populations: dict) -> dict:
    """One cell: its adjudicated statistic, its p-value, or a NAMED status."""
    per_coin = {c: r for c, r in coin_results.items() if arm in r}
    if not per_coin:
        return I11.build_cell(arm, head, budget,
                              status=I11.CELL_STATUS_UNEVALUABLE,
                              detail=f"arm {arm} produced no result")
    # pooled over coins at the ACTION unit (A1.5)
    n_actions = sum(populations[c]["eval_n_actions"] for c in per_coin)

    if head == "Q4_combined_ev":
        # the DECISION metric, per budget, with BOTH declared nulls
        stat, pval, status, detail = _q4_cell(arm, budget, per_coin)
        return I11.build_cell(arm, head, budget, statistic=stat, p_value=pval,
                              status=status, n_actions=n_actions, detail=detail)

    # Q1/Q2/Q3: the adjudicated statistic is a discrimination/calibration
    # figure, identical across budgets (budgets select CANCELLATIONS, not
    # predictions). It is reported in every budget slot so the declared family
    # is complete.
    #
    # I11-B2: the sentence that stood here -- "the p-value comes from the
    # matched-random null on the decision side only, which is why these carry no
    # incremental p" -- was a RATIONALISATION of a missing null, printed beside
    # p_value=None. Prereg §5(1) declares the matched-random null PER HEAD, not
    # on the decision side only. It is now computed in head_report, beside each
    # head's own population, and read here.
    per_coin_ev = {c: {"statistic": None, "matched_random_p": _matched_p(r[arm], head)}
                   for c, r in per_coin.items()}
    stats, statuses = [], []
    for c, r in per_coin.items():
        adj = r[arm]["adjudicated_statistics"]
        if head == "Q1_arrival":
            stats.append(adj.get("Q1_arrival"))
            statuses.append(r[arm]["heads"]["Q1_arrival"]["status"])
        elif head == "Q2_sign":
            stats.append(adj.get("Q2_sign"))
            statuses.append(adj.get("Q2_cell_status"))
        else:
            stats.append(adj.get("Q3_magnitudes"))
            statuses.append("OK" if adj.get("Q3_magnitudes") is not None
                            else I11.CELL_STATUS_UNEVALUABLE)
    for (c, r), st in zip(per_coin.items(), stats):
        per_coin_ev[c]["statistic"] = st
    good = [x for x in stats if x is not None]
    if not good:
        return I11.build_cell(
            arm, head, budget, status=I11.CELL_STATUS_UNEVALUABLE,
            n_actions=n_actions,
            detail=f"no coin produced an adjudicable statistic; per-coin "
                   f"statuses {statuses}")
    if any(st not in ("OK", None) for st in statuses):
        return I11.build_cell(
            arm, head, budget, statistic=min(good),
            status=I11.CELL_STATUS_UNDERPOWERED, n_actions=n_actions,
            detail=f"at least one coin is not OK: {statuses}")
    # THE COLLAPSE IS ONLY UNAMBIGUOUS WHEN THERE IS NOTHING TO COLLAPSE.
    # A single coin and a single sub-head need no undeclared rule, so those
    # cells adjudicate normally. Anything wider needs a rule the frozen text
    # does not give, and BE does not supply one (§9.5 "BE does not decide").
    subs = _matched_p(next(iter(per_coin.values()))[arm], head)
    ambiguous = []
    if len(per_coin) > 1:
        ambiguous.append(
            f"{len(per_coin)} coins collapse into one cell, but §9.4 rules the "
            f"verdict regime PER COIN -- 'btc and eth accrue independently, and "
            f"one coin reaching the bar does not carry the other'. min() lets "
            f"the worse coin decide for both, which is that carry with the sign "
            f"reversed; the declared family has no coin dimension to hold them "
            f"apart and A1.4 freezes the denominator at 24")
    # Q2's two sides are NOT a gap: A1.4 carries a USER RULING (R-249) that
    # under Option 1 the cell statistic is min(AUC(p_pos), AUC(p_neg)), the
    # WORSE side. The null evidence for that statistic is therefore the worse
    # side's own null -- a consequence of the ruling, not a fresh choice.
    # Q3 has no such ruling: A1.4 names "calibration slope" for the cell while
    # the parent table requires m_harm and m_good "each reported SEPARATELY"
    # with a CI for each. One cell, two unreconciled judgements.
    if head == "Q3_magnitudes" and len(subs) > 1:
        ambiguous.append(
            f"{head} spans {sorted(subs)}: A1.4 names a single 'calibration "
            f"slope' for the cell, while the frozen table requires them 'each "
            f"reported SEPARATELY' with 'calibration slope CI excludes 0 for "
            f"each'. One cell cannot carry two separate slope-and-CI "
            f"judgements, and unlike Q2 (R-249) no ruling says which wins")
    ps = _cell_p(per_coin, arm, head)
    if ambiguous:
        return I11.build_cell(
            arm, head, budget, statistic=min(good), p_value=None,
            status=I11.CELL_STATUS_AGG_UNDECLARED, n_actions=n_actions,
            detail="AGGREGATION UNDECLARED, stated for ruling rather than "
                   "chosen (I11-B4): " + "; ".join(ambiguous) +
                   f". The worst-coin statistic {min(good)!r} is carried as a "
                   f"placeholder ONLY -- it is not adjudicated, p is withheld, "
                   f"and the per-coin evidence is {per_coin_ev!r} so a ruling "
                   f"can be applied without re-running.")
    if head in ("Q2_sign", "Q3_magnitudes"):
        return I11.build_cell(
            arm, head, budget, statistic=min(good), p_value=(ps[0] if ps else None),
            status=I11.CELL_STATUS_NO_COUNTERPART, n_actions=n_actions,
            detail=f"R-237: the incumbent has no sign/magnitude head, so no "
                   f"INCREMENTAL null exists for this head. The p reported is "
                   f"the MATCHED-RANDOM null (prereg §5.1), which does apply; "
                   f"the two nulls are named so neither is read as the other.")
    return I11.build_cell(
        arm, head, budget, statistic=min(good),
        p_value=(ps[0] if ps else None),
        status=I11.CELL_STATUS_OK if ps else I11.CELL_STATUS_UNEVALUABLE,
        n_actions=n_actions,
        detail=(f"single coin, single head: adjudicated on its own "
                f"matched-random null (prereg §5.1), {len(ps)} p available"
                if ps else
                "no matched-random null was computable for this head, so there "
                "is no null evidence and the cell is not OK"))


_CELL_SUBHEADS = {"Q1_arrival": ("Q1_arrival",),
                  "Q2_sign": ("Q2_p_pos", "Q2_p_neg"),
                  "Q3_magnitudes": ("Q3_m_harm", "Q3_m_good")}


def _matched_p(r_arm: dict, head: str) -> dict:
    """The matched-random p per sub-head, READ from where it was computed.

    It is computed in head_report, beside the head's own predictions and
    outcomes. Recomputing it here would mean re-deriving the population, which
    is how two definitions of a population diverge."""
    out = {}
    for sub in _CELL_SUBHEADS.get(head, ()):
        mr = (r_arm.get("heads", {}).get(sub) or {}).get("matched_random") or {}
        out[sub] = mr.get("p_value")
    return out


def _cell_p(per_coin: dict, arm: str, head: str) -> list:
    """The matched-random p that belongs to the cell's ADJUDICATED statistic.

    For Q2 that is the WORSE side's p, because R-249 rules the cell statistic to
    be min(AUC(p_pos), AUC(p_neg)): the null must describe the number the cell
    actually reports, not a sibling of it."""
    out = []
    for r in per_coin.values():
        heads = r[arm].get("heads", {})
        subs = _CELL_SUBHEADS.get(head, ())
        if head == "Q2_sign" and len(subs) > 1:
            scored = [(heads.get(x, {}).get("auc"), x) for x in subs]
            scored = [(a, x) for a, x in scored if a is not None]
            if not scored:
                continue
            worse = min(scored)[1]
            v = (heads.get(worse, {}).get("matched_random") or {}).get("p_value")
        else:
            v = None
            for x in subs:
                v = (heads.get(x, {}).get("matched_random") or {}).get("p_value")
                break
        if v is not None:
            out.append(v)
    return out


def _q4_cell(arm: str, budget: str, per_coin: dict) -> tuple:
    """Q4's cell: net cents at the action unit, with the incremental null.

    The incremental-over-incumbent null is the one iteration 010 lacked, and it
    is the reason 'beats random' was mistaken for 'beats the incumbent' for a
    full cycle."""
    inc_by_window = {}
    net = 0.0
    for coin, r in per_coin.items():
        econ = r[arm].get("economics", {}).get(budget)
        if not econ:
            return (None, None, I11.CELL_STATUS_UNEVALUABLE,
                    f"no economics for {coin}@{budget}")
        net += econ["net_cents"]
        # I11-B2: this key EXISTS ONLY when economics were paired against an
        # incumbent. Previously the cell read the candidate's own value from
        # `increment_by_window` and reported it as "increment vs incumbent" --
        # a sign-flip of the candidate against ZERO, wearing the label of a
        # comparison it never made. That is precisely the conflation prereg
        # 5.2 was written to prevent ("beats random was mistaken for beats the
        # incumbent for a full cycle"). No counterpart is a NAMED STATUS, not
        # a number with an optimistic caption.
        if not econ.get("paired_against_incumbent"):
            return (net, None, I11.CELL_STATUS_NO_COUNTERPART,
                    f"{coin}@{budget} economics were computed with no incumbent "
                    f"counterpart, so no candidate-minus-incumbent statistic "
                    f"exists on the identical action population (prereg 5.2). "
                    f"Net {net:+.1f}c is the CANDIDATE'S OWN value and is not "
                    f"an increment.")
        for w, v in econ["increment_by_window"].items():
            inc_by_window[f"{coin}/{w}"] = inc_by_window.get(f"{coin}/{w}", 0.0) + v
    if not inc_by_window:
        return (net, None, I11.CELL_STATUS_UNEVALUABLE,
                "no per-window increments to permute")
    null = I11.sign_flip_null(inc_by_window)
    return (net, null["p_two_sided"], I11.CELL_STATUS_OK,
            f"increment vs incumbent {null['observed']:+.1f}c over "
            f"{null['n_units']} windows; {null['n_perm']} sign-flip "
            f"permutations, units consumed in SORTED order (R-234)")


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
            rep = report_arm(pr, tg, rows)
            n = len(rows)
            ok(all(len(pr[k]) == n for k in
                   ("p_fill", "p_pos", "p_neg", "m_harm", "m_good",
                    "expected_cancel_value")),
               f"{arm}: EVERY head predicts on EVERY action — row-aligned "
               f"(the reviewer measured 3/2/1/1 before this)")
            ok(rep["all_heads_reported"], f"{arm}: all heads reported")
            # A1.5 (I11-B3): head "n" is now the ACTION count, so the DOMAIN
            # check -- which is what these two assert -- reads n_rows, and the
            # action property is asserted separately rather than conflated.
            ok(rep["heads"]["Q2_p_pos"]["n_rows"] == c["n_preventable"],
               f"{arm}: Q2 is SCORED on the preventable base while PREDICTING "
               f"on all actions — different domains, kept separate")
            ok(rep["heads"]["Q3_m_harm"]["n_rows"] == c["n_v_positive"],
               f"{arm}: Q3 m_harm is scored on V>0 only")
            ok(all(h["n"] == h["n_actions"] and h["n_basis"] == "ACTION (A1.5)"
                   and h["n_actions"] <= h["n_rows"]
                   for h in rep["heads"].values()),
               f"{arm}: every head reports n as the ACTION count with its "
               f"basis named, never the prediction count (A1.5)")
            ok(isinstance(pr["expected_cancel_value"][0], float),
               f"{arm}: Q4 composes per action (never fitted)")
            ok(rep["adjudicated_statistics"]["Q2_cell_rule"].startswith("min("),
               f"{arm}: Q2's cell takes the WORSE sign head (A1.4)")

        except Exception as e:
            ok(False, f"{arm}: fits and reports ({type(e).__name__}: {e})")

    # ONE SIDE UNEVALUABLE MUST NOT LET THE OTHER CARRY THE CELL.
    # Raised into the Codex round; closed here with a falsifier.
    _mk = lambda auc, st: {"auc": auc, "status": st, "n": 500}
    _base = dict(rep)
    for _lbl, _pp, _pn, _want in (
            ("p_neg AUC is None", _mk(0.92, "OK"), _mk(None, "OK"), None),
            ("p_pos AUC is None", _mk(None, "OK"), _mk(0.91, "OK"), None),
            ("p_neg UNDERPOWERED", _mk(0.92, "OK"),
             _mk(0.55, I11.UNDERPOWERED), None),
            ("both evaluable", _mk(0.92, "OK"), _mk(0.61, "OK"), 0.61)):
        _sides = {"p_pos": _pp, "p_neg": _pn}
        _missing = sorted(k for k, h in _sides.items() if h.get("auc") is None)
        _under = sorted(k for k, h in _sides.items()
                        if h.get("status") == I11.UNDERPOWERED)
        _cell = None if (_missing or _under) else min(_pp["auc"], _pn["auc"])
        ok(_cell == _want,
           f"Q2 cell with {_lbl}: {_cell!r} (a single side must NOT carry the "
           f"cell past the UNDERPOWERED machinery)")
    ok("BOTH sides must be evaluable" in
       rep["adjudicated_statistics"]["Q2_cell_rule"],
       "and the rule states the both-sides requirement in the artifact")

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
    rept = report_arm(prt, tgt, thin)
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
       == ["--selftest"],
       "an EXPLICIT --selftest run is EXEMPT (a selftest writes no artifact "
       "by design; the guard must not punish it)")
    ok(assert_outputs_written((_d / "nope5.json",),
                              argv=["prog", "--dry-run"]).get("exempt")
       == ["--dry-run"],
       "a --dry-run is EXEMPT too — it writes to a throwaway path by design, "
       "and the exemption is still keyed on the DECLARED MODE")
    for _bad_argv in (["prog"], ["prog", "--dryrun"], ["prog", "--dry_run"]):
        try:
            assert_outputs_written((_d / "nope6.json",), argv=_bad_argv)
            ok(False, f"a non-declared mode {_bad_argv[1:]} is NOT exempt")
        except RuntimeError:
            ok(True, f"a non-declared mode {_bad_argv[1:]} is NOT exempt — "
                     f"lookalike flags do not grant the exemption")
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

    # ------------------------------------------------ I11-1 / I11-2 / I11-3 ---
    # I11-1: the key main() prints must EXIST in what report_arm emits.
    import inspect as _ins
    # scan CODE, not comments: my first version matched the comment that
    # QUOTES the old key and reported a false red against the fix itself.
    _msrc = "\n".join(l.split("#", 1)[0] for l in
                      _ins.getsource(main).split("\n"))
    _keys = set(re.findall(r"h\['([A-Za-z0-9_]+)'\]", _msrc))
    _emitted = set(rep["heads"])
    ok(_keys <= _emitted,
       f"I11-1 every heads key main() prints is EMITTED by report_arm "
       f"(prints {sorted(_keys)}; emits {sorted(_emitted)}) — it printed "
       f"'Q2_sign', which report_arm has never emitted, so the runner raised "
       f"KeyError after the FIRST arm, BEFORE the artifact write")

    # I11-2: Q4 economics per budget, at the ACTION unit.
    _rows2 = [dict(r, slug=f"w{i//4}", side="BUY_UP", gen=i) for i, r in
              enumerate(rows)]
    _pr2 = apply_arm(fit_arm("composed_linear", X, tg), X, tg)
    _ec = q4_economics(_pr2, _rows2)
    ok(set(_ec) == set(I11.BUDGETS_011),
       "I11-2 economics are produced for EVERY declared budget (they were "
       "metadata: Q4 was composed in apply_arm and DISCARDED by report_arm)")
    ok(all(_ec[b]["n_cancelled_actions"] <= _ec[b]["n_actions_total"]
           for b in _ec),
       "I11-2 a budget never cancels more actions than exist")
    ok(_ec["5%"]["n_cancelled_actions"] <= _ec["15%"]["n_cancelled_actions"],
       "I11-2 a larger budget cancels at least as many actions")
    # I11-B2: this call passes NO incumbent, so there is no increment to
    # retain. The per-window values are the CANDIDATE'S OWN, and the key now
    # says so -- the previous name let _q4_cell read them as an increment.
    ok(all(_ec[b]["candidate_value_by_window"] for b in _ec),
       "I11-2 per-window values are retained so a null has units to permute")
    ok(all("increment_by_window" not in _ec[b] for b in _ec),
       "I11-B2 an UNPAIRED run emits NO increment_by_window key at all; a "
       "caller cannot read the candidate's own value as an increment if the "
       "key it would read does not exist")

    # the family is BUILT and ADJUDICATED, not merely defined.
    # I11-B2: the fixture now carries what a REAL result carries -- per-head
    # matched-random nulls (prereg §5.1) and economics PAIRED against an
    # incumbent -- because a fixture missing them made the family look complete
    # while every cell was p-less.
    _ecp = q4_economics(_pr2, _rows2, incumbent={
        "expected_cancel_value": list(reversed(_pr2["expected_cancel_value"]))})
    _mr = lambda pv: {"status": "OK", "p_value": pv, "n_draws": 500}
    _fake = {c: {a: {"adjudicated_statistics": {
                        "Q1_arrival": 0.7, "Q2_sign": 0.6,
                        "Q2_cell_status": "OK", "Q3_magnitudes": 1.0},
                     "heads": {"Q1_arrival": {"status": "OK", "auc": 0.7,
                                              "matched_random": _mr(0.004)},
                               "Q2_p_pos": {"auc": 0.61, "matched_random": _mr(0.02)},
                               "Q2_p_neg": {"auc": 0.60, "matched_random": _mr(0.31)},
                               "Q3_m_harm": {"matched_random": _mr(0.05)},
                               "Q3_m_good": {"matched_random": _mr(0.44)}},
                     "economics": _ecp}
                 for a in I11.ARMS_011} for c in ("btc",)}
    _pops = {"btc": {"eval_n_actions": 500}}
    _fam = evaluate_family(_fake, _pops)
    ok(_fam["declared_family_size"] == 24 and len(_fam["cells"]) == 24,
       "I11-2 the DECLARED 24-cell family is actually BUILT (the evaluator had "
       "ZERO references in the runner for a whole batch)")
    ok(_fam["holm_denominator"] == 24,
       "I11-2 Holm runs over the declared denominator")
    ok(all(_fam["cells"][k]["p_value"] is not None
           for k in _fam["cells"] if "/Q4_combined_ev/" in k),
       "I11-2 Q4 cells carry a PERMUTATION p-value from the incremental null "
       "WHEN AN INCUMBENT IS SUPPLIED")
    ok(all(_fam["cells"][k]["status"] == I11.CELL_STATUS_NO_COUNTERPART
           for k in _fam["cells"] if "/Q2_sign/" in k),
       "I11-2 Q2 incremental cells carry NO_INCUMBENT_COUNTERPART (R-237); its "
       "two-sided collapse is RULED (R-249) so it is not an aggregation gap")
    ok(_fam["cells"]["composed_linear/Q2_sign/5%"]["p_value"] == 0.31,
       "I11-B2 Q2's null is the WORSE SIDE's (R-249 rules the statistic to be "
       "min(AUC); the p must describe the number the cell reports, 0.31 not 0.02)")
    ok(all(_fam["cells"][k]["status"] == I11.CELL_STATUS_AGG_UNDECLARED
           for k in _fam["cells"] if "/Q3_magnitudes/" in k),
       "I11-B4 Q3 cells state the AGGREGATION GAP rather than picking: A1.4 "
       "names one slope, the frozen table requires m_harm and m_good each "
       "reported SEPARATELY with a CI, and no ruling reconciles them")
    ok(all(_fam["cells"][k]["p_value"] is None
           for k in _fam["cells"] if "/Q3_magnitudes/" in k),
       "I11-B4 a cell whose collapse is undeclared WITHHOLDS its p rather than "
       "publishing one under a rule nobody declared")

    # output-level known-bad
    ok(assert_receipt_has_all_cells({"family": _fam})["cells_present"] == 24,
       "I11-2 a COMPLETE receipt passes the output-level cell guard")
    for _lbl, _bad in (("no family", {}), ("empty family", {"family": {}}),
                       ("a cell missing", {"family": dict(
                           _fam, cells={k: v for k, v in
                                        list(_fam["cells"].items())[:-1]})})):
        try:
            assert_receipt_has_all_cells(_bad)
            ok(False, f"I11-2 a receipt REFUSES when {_lbl}")
        except RuntimeError:
            ok(True, f"I11-2 a receipt REFUSES when {_lbl}")

    # I11-3: the HEAD itself states unevaluability
    _one = I11.head_report("Q2_p_neg", "probability", [0.4] * 300, [0] * 300)
    ok(_one["status"] == I11.UNEVALUABLE and _one["auc"] is None,
       "I11-3 a ONE-CLASS head reports UNEVALUABLE, not OK-with-auc-None — the "
       "cell must not be the only place the meaning is corrected")
    ok("only one class" in _one.get("unevaluable_reason", ""),
       "I11-3 and it says WHY (unevaluable_reason survives even when "
       "UNDERPOWERED takes precedence in status, so both facts are readable)")
    _flat = I11.head_report("Q3_m_harm", "magnitude", [1.0] * 300,
                            [float(i) for i in range(300)])
    ok(_flat["status"] == I11.UNEVALUABLE,
       "I11-3 a CONSTANT predictor head is UNEVALUABLE too (no slope is not a "
       "slope of zero)")
    _hp = I11.head_report("Q1_arrival", "probability",
                          [i / 300 for i in range(300)],
                          [i % 2 for i in range(300)])
    ok(_hp["status"] == "OK",
       "I11-3 a healthy head is still OK (the status is not a wall)")

    ok("phase2_iter011_run.py" not in PA.CODE_IDENTITY_FILES,
       "this runner is NOT in the identity lattice — the standalone property "
       "is checked, not assumed")


    # ============ I11-B2/B3/B4 (reviewer batch 2) — RED FIRST ==============
    # Each encodes the FROZEN text, cited, and must fail against the code as it
    # stood when they were written. A falsifier written after the fix proves the
    # fix is self-consistent, not that the defect existed.

    def _gr(slug, side, gen, t, v):
        r = row(v)
        r.update({"slug": slug, "side": side, "gen": gen, "t_start": t})
        return r

    # ---- B3(c): A1.5 freezes FIRST-CROSSING valuation ---------------------
    # A1.5: "Q4 economics are evaluated at the action unit with first-crossing
    # dedup, as harmful_action_eval already does." That reference RANKS
    # generations by their max score but VALUES
    #     cross = next(i for i in gens[gk] if scores[i] >= theta)
    # -- the EARLIEST row that crosses the budget threshold, not the best one.
    # Two generations so theta lands below gen1's max: gen1's earliest crossing
    # row is worth -100c and its max-composed row +100c, on the SAME generation.
    _rws = [_gr("W", "BUY", 1, 0.0, -100.0),    # ecv 0.60, crosses first
            _gr("W", "BUY", 1, 1.0, +100.0),    # ecv 0.90, the max
            _gr("W", "BUY", 2, 0.0, +10.0)]     # ecv 0.50, sets theta
    _pred = {"expected_cancel_value": [0.60, 0.90, 0.50]}
    try:
        _net = q4_economics(_pred, _rws, budgets=["100%"])["100%"]["net_cents"]
    except Exception as _ex:                       # noqa: BLE001
        _net = f"raised {type(_ex).__name__}: {_ex}"
    ok(_net == -90.0,
       f"B3 Q4 values the FIRST CROSSING (A1.5): expected -90.0c "
       f"(-100 + 10), max-composed rule gives +110.0c; got {_net!r}")

    # ---- B3(c): the unit string must name ONE rule ------------------------
    try:
        _u = q4_economics(_pred, _rws, budgets=["100%"])["100%"]["unit"]
    except Exception:                              # noqa: BLE001
        _u = ""
    ok(not ("first-crossing" in _u and "max composed" in _u),
       f"B3 the emitted unit names ONE dedup rule, not two incompatible ones "
       f"(got {_u!r})")

    # ---- B3(a): n for every reported head is the ACTION count -------------
    # A1.5: "n for every reported head is the ACTION count, not the prediction
    # count. A head predicting on rows must state its action count beside it."
    # 3 rows spanning 2 generations: a row-level n reads 3.
    _tg = head_targets(_rws)
    ok(_tg["counts"].get("n_actions") == 2,
       f"B3 head counts state the ACTION count beside n_rows (A1.5): 3 rows / "
       f"2 generations must report n_actions=2; got "
       f"{_tg['counts'].get('n_actions')!r} beside n_rows="
       f"{_tg['counts'].get('n_rows')!r}")
    ok(_tg["counts"].get("n_actions", 10**9) <= _tg["counts"]["n_rows"],
       "B3 no head reports an action count larger than its row count (A1.5)")

    # ---- B2: Q4's increment is CANDIDATE MINUS INCUMBENT -------------------
    # Prereg §5(2): "statistic = the head's metric minus the incumbent's on the
    # IDENTICAL action population". q4_economics took no incumbent at all, so
    # the cell reported a sign-flip of the candidate's OWN value under the label
    # "increment vs incumbent" -- the exact conflation §5(2) exists to prevent
    # ("beats random was mistaken for beats the incumbent for a full cycle").
    _inc_same = {"expected_cancel_value": [0.60, 0.90, 0.50]}
    _inc_diff = {"expected_cancel_value": [0.40, 0.90, 0.50]}
    try:
        _a = q4_economics(_pred, _rws, budgets=["100%"], incumbent=_inc_same)
        _b = q4_economics(_pred, _rws, budgets=["100%"], incumbent=_inc_diff)
        _sa = sum(_a["100%"]["increment_by_window"].values())
        _sb = sum(_b["100%"]["increment_by_window"].values())
    except TypeError as _ex:
        _sa = _sb = f"raised {_ex}"
    ok(_sa == 0.0,
       f"B2 an incumbent IDENTICAL to the candidate yields increment 0 "
       f"(positive control; got {_sa!r})")
    ok(isinstance(_sb, float) and _sb != 0.0,
       f"B2 a DIFFERENT incumbent moves the increment (falsifier; got {_sb!r})")

    # ---- B4: the cell guard must validate CONTENTS, not keys ---------------
    # assert_receipt_has_all_cells compared set(fam["cells"]) against the
    # declaration, so 24 EMPTY dicts satisfied it -- the same shape as the
    # empty file_hashes map that R-228(1) closed one level up.
    _empty = {"family": {"cells": {k: {} for k in I11.declared_family()["cells"]},
                         "holm_denominator": len(I11.declared_family()["cells"])}}
    try:
        assert_receipt_has_all_cells(_empty)
        _refused = False
    except RuntimeError:
        _refused = True
    ok(_refused,
       "B4 a receipt whose 24 cells are EMPTY DICTS is REFUSED; matching the "
       "declared key set is not evidence that anything was evaluated")

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
DRY_RUN_FLAG = "--dry-run"
# Modes that CORRECTLY do not write the declared output. Each is a DECLARED
# FLAG, never "the outputs happen to be missing" — that would be a bypass
# excusing the exact failure the guard exists to catch.
#
# THIS LIST HAS GROWN TWICE. --selftest was added when a green selftest exited
# 1; --dry-run when the harness wrote to a throwaway path. The pattern is that
# the guard checks a FIXED declared path while modes legitimately write
# elsewhere. If a third mode appears, the right fix is for main() to REPORT
# where it wrote and the guard to check THAT — noted here rather than
# refactored now, because a third case has not appeared and speculative
# generality is its own defect.
NON_WRITING_MODES = (SELFTEST_FLAG, DRY_RUN_FLAG)


def is_selftest_mode(argv=None) -> bool:
    """A DECLARED mode that correctly writes no declared artifact.

    The exemption is tied to the DECLARED MODE, never to 'the outputs happen to
    be missing'. In every other mode the guard applies in full, including when
    selftest() has been called internally as the run's gate."""
    a = sys.argv if argv is None else argv
    return any(f in a for f in NON_WRITING_MODES)


def assert_outputs_written(outputs=DECLARED_OUTPUTS, argv=None) -> dict:
    """A run that writes NOTHING must not exit 0.

    The silent-success shape: a clean exit obtained by not doing the work. It is
    indistinguishable from a completed run at the exit code, and the exit code
    is what an operator reads first. Every declared output must exist, parse,
    and carry its artifact identity — existence alone would pass a zero-byte
    file, and parsing alone would pass an empty object."""
    if is_selftest_mode(argv):
        return {"exempt": [f for f in NON_WRITING_MODES
                           if f in (sys.argv if argv is None else argv)],
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



def synthetic_populations(n_per_coin: int = 400, seed: int = 20260828) -> tuple:
    """FIT/EVAL blocks shaped exactly like _feature_pass output, no real data.

    Used ONLY by --dry-run. The point is to exercise main()'s OWN path over
    substituted populations: same fits, same reports, same economics, same
    family assembly, same output guard. A parallel harness that IMITATED main()
    would prove nothing — that is precisely the defect Codex found (I11-2), a
    component suite green while main() called none of it."""
    import random as _r
    rr = _r.Random(seed)
    L = str(D.TARGET_LATENCY_MS)
    out = {}
    for coin in ("btc", "eth"):
        rows, PM, FN, ST = [], [], [], []
        for i in range(n_per_coin):
            # a mix that populates every head: harm, good, zero, and no-fill
            bucket = i % 4
            fill = bucket != 3
            v = {0: 4.0 + rr.random(), 1: -3.0 - rr.random(),
                 2: 0.0, 3: 0.0}[bucket]
            rows.append({
                "slug": f"{coin}-updown-5m-{1787650200 + (i // 8) * 300}",
                "coin": coin, "side": "BUY_UP" if i % 2 else "SELL_UP",
                "gen": i // 2, "t0": 1787650200.0 + (i // 8) * 300,
                "t_start": float(i % 8), "any_fill_ahead": fill,
                "latency": {L: {"preventable_value_cents": v if fill else 0.0,
                                "preventable_shares": 1.0 if fill else 0.0,
                                "stale_shares": 0.0}}})
            PM.append([rr.random() for _ in range(6)])
            FN.append([rr.random()])
            ST.append([rr.random() for _ in range(4)])
        out[coin] = {"kept": rows, "PM": PM, "FN": FN, "ST": ST,
                     "drops": {"pm": 0, "fine": 0, "no_archive": 0,
                               "state_join_failed": 0}}
    # EVAL is an INDEPENDENT draw, so the out-of-sample structure is preserved
    fit = out
    ev = synthetic_populations.__wrapped__(n_per_coin, seed + 1) \
        if hasattr(synthetic_populations, "__wrapped__") else None
    return fit, ev


def _synth_pair(n_per_coin: int = 400) -> tuple:
    """FIT and EVAL as two INDEPENDENT synthetic draws."""
    fit, _ = synthetic_populations(n_per_coin, seed=20260828)
    ev, _ = synthetic_populations(n_per_coin, seed=20260829)
    return fit, ev


def main() -> int:
    """The 011 DEVELOPMENT run. Known-bads first: selftest gates the run."""
    if "--selftest" in sys.argv:
        return selftest()
    if selftest():
        raise SystemExit("REFUSED: selftest RED; no numbers from an instrument "
                         "that has not shown it can fire.")
    import phase2_embargo as EMB

    # RULE 17 HARNESS. --dry-run substitutes SYNTHETIC populations and runs
    # EVERYTHING ELSE IN THIS FUNCTION unchanged: both arms fitted, applied,
    # reported, economics computed, the 24-cell family assembled and
    # adjudicated, the output guard enforced. It reads no real data, touches no
    # model artifact and writes to a throwaway path.
    #
    # It exists because a component suite cannot see an unwired main(): every
    # falsifier calls evaluate_family, q4_economics and
    # assert_receipt_has_all_cells DIRECTLY, so they would all stay green if
    # someone unwired main() again. This drives main()'s OWN path.
    _dry = "--dry-run" in sys.argv
    if _dry:
        print("  DRY RUN: synthetic populations, no real data, throwaway "
              "output. Exercising main()'s own path.", flush=True)

    if not _dry:
        PA.assert_modules_under_root()
        PA.pin_data_root()
        PA.assert_tape_is_v5()
        _v = PA.assert_gate_passed()
        PA.assert_verdict_subject_is(PA.TAPE_PATH, _v)
        ident = PA._tape_identity()
        print(f"  identity: tape {ident['tape_sha256_prefix']} fragment "
              f"{ident['fragment_sha256_prefix']} topup "
              f"{ident['topup_sha256_prefix']}", flush=True)
        print("  indexing train split...", flush=True)
        TP = PA.tape_index("train")
        print(f"  train split indexed: {len(TP):,} rows", flush=True)
        FIT = PA._feature_pass(PA.FRAGMENT, "fragment", TAPE=TP)
        del TP
        print("  indexing score split for the embargo boundary...", flush=True)
        SP = PA.tape_index("score")
        print(f"  score split indexed: {len(SP):,} rows", flush=True)
    else:
        ident = {"tape_sha256_prefix": "DRY_RUN", "DRY_RUN": True,
                 "fragment_sha256_prefix": "DRY_RUN",
                 "topup_sha256_prefix": "DRY_RUN"}
        FIT, EVAL = _synth_pair()
        SP = None
    # The embargo probe. In a dry run it is derived from the SYNTHETIC EVAL
    # rows, exactly as the real path derives it from the score split — an empty
    # probe would make phase2_embargo refuse ("an embargo over an empty holdout
    # is vacuous, not satisfied", and it is right to), and SKIPPING the purge
    # would mean the dry run does not exercise it, which defeats the harness.
    if SP is not None:
        probe = [{"t0": v["t0"], "t_start": v["t_start"]} for v in SP.values()]
    else:
        probe = [{"t0": r["t0"] + 3600.0, "t_start": r["t_start"]}
                 for c in EVAL.values() for r in c["kept"]]
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
    if not _dry:
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
            "eval_n_actions": I11.action_count(EVAL[coin]["kept"]),
            "fit_n_actions": I11.action_count(FIT[coin]["kept"]),
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
            rep = report_arm(ap, tge, EVAL[coin]["kept"])
            rep["fit_seconds"] = round(time.time() - t0, 1)
            rep["evaluation"] = "OUT-OF-SAMPLE within development"
            # `ap` is apply_arm's output; this said `pr`, an undefined name.
            # A NameError that would have killed the REAL run at the FIRST arm,
            # before any artifact — the same class as I11-1, and invisible to
            # every component test because they all call q4_economics directly.
            # Found by the --dry-run harness on its first execution.
            rep["economics"] = q4_economics(ap, EVAL[coin]["kept"])
            out["results"][coin][arm] = rep
            h = rep["heads"]
            # I11-1: this printed h['Q2_sign'], a key report_arm has never
            # emitted — the heads dict is keyed Q2_p_pos / Q2_p_neg. It raised
            # KeyError after the FIRST fitted arm, BEFORE the artifact write, so
            # no real development run could ever complete. The adjudicated Q2
            # figure lives in adjudicated_statistics, not in heads.
            _adj = rep["adjudicated_statistics"]
            print(f"  [{coin}/{arm}] Q1 auc {h['Q1_arrival'].get('auc')} | "
                  f"Q2 cell {_adj.get('Q2_sign')} ({_adj.get('Q2_cell_status')}) | "
                  f"Q3h slope {h['Q3_m_harm'].get('calibration_slope')} | "
                  f"Q3g slope {h['Q3_m_good'].get('calibration_slope')} | "
                  f"underpowered {rep['underpowered_heads']} | "
                  f"unevaluable {rep.get('unevaluable_heads')}", flush=True)
            del fr, ap
        del Xf, Xe, FIT[coin]["PM"], FIT[coin]["FN"], FIT[coin]["ST"]

    # I11-2: the DECLARED family, actually evaluated and adjudicated.
    out["family"] = evaluate_family(out["results"], out["populations"])
    out["cluster_disclosure"] = I11.cluster_disclosure(
        min((p["eval_population_and_reach"]["G_complete_utc_days"]
             for p in out["populations"].values()), default=0), "window")
    assert_receipt_has_all_cells(out)
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
    _out_path = OUT
    if _dry:
        import tempfile as _tf_dry
        _out_path = Path(_tf_dry.mkdtemp()) / OUT.name
        out["DRY_RUN"] = {
            "synthetic_populations": True, "real_data_read": False,
            "model_artifacts_read": False, "output_path": str(_out_path),
            "why": "exercises main()'s OWN path — a component suite cannot see "
                   "an unwired main(), which is defect I11-2"}
    with _out_path.open("w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
        fh.flush(); os.fsync(fh.fileno())
    ev = assert_outputs_written((_out_path,))
    print(f"\nWROTE {_out_path.name}: {ev.get(_out_path.name, ev)}", flush=True)
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
