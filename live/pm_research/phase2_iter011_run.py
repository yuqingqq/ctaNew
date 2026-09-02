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

import inspect
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
#: The FROZEN preregistration and the commit it was frozen at. RR3-1: the
#: coverage floor's expected set must terminate at a document the USER froze,
#: not at a constant a seat can edit, so this pair is read by the emission AND
#: by the check that anchors the constants to it -- one value, not two.
PREREG_DOC = "ITER011_CONDITIONAL_VALUE_PREREGISTRATION.md"
PREREG_COMMIT = "3b71d3e"
RECEIPT_FAMILY = "ITER011_CONDITIONAL_VALUE"


#: Coins this runner knows. A `--coin` outside it REFUSES rather than silently
#: producing an empty population, which is the shape that makes "no rows" and
#: "wrong name" indistinguishable.
COINS_011 = ("btc", "eth")


def _coin_drop(blocks: dict, coin: str | None, what: str) -> dict:
    """Drop the non-selected coin's feature block the moment the pass returns.

    `_feature_pass` builds BOTH coins whatever we ask for -- it is bound code
    and cannot be told otherwise -- so the saving is in not CARRYING the other
    coin through everything downstream."""
    if coin is None:
        return blocks
    if coin not in blocks:
        raise RuntimeError(
            f"REFUSED: --coin {coin!r} absent from the {what} pass, which "
            f"produced {sorted(blocks)}. A missing coin is not an empty one.")
    for other in [c for c in list(blocks) if c != coin]:
        blocks.pop(other, None)
    return blocks


def restore_valuation_gate(block: dict) -> dict:
    """Re-attach `any_fill_ahead` to kept rows, using THE canonical function.

    `phase2_arms._feature_pass` projects each kept row to a FIXED field list --
    ("slug","day","t0","t_start","side","gen","latency","coin") -- which does
    NOT include `any_fill_ahead`. `phase2_iter011.validate_row` (A1.3, FROZEN)
    REQUIRES it and refuses MISSING_GATE without it. The two had never run
    together, because iteration 011 had never been fitted; the first real fit
    died here.

    NOT reimplemented, and not invented: `harmful_exposure_rows.any_fill_ahead`
    is declared as "THE single definition" precisely because two definitions of
    a valuation gate is one too many, and it states that the keptrow derivation
    OVERWRITES the builder's stored value at fit/score time -- so recomputing
    it from `latency` IS the rule that governed every committed result.

    VERIFIED EXACT on the whole population before relying on it: stored vs
    derived agree on 1,125,289 fragment rows and 638,917 topup rows, ZERO
    disagreements. `latency` is in the projection, so this needs no join and
    no extra population in memory.

    Fixed in the RUNNER rather than in the projection because
    `phase2_arms.py` is in CODE_IDENTITY_FILES and the frozen candidate binds
    its hash; this runner is declared outside the lattice.
    """
    import harmful_exposure_rows as _HER
    n = 0
    for r in block.get("kept") or ():
        if "any_fill_ahead" not in r:
            r["any_fill_ahead"] = _HER.any_fill_ahead(r.get("latency"))
            n += 1
    if n:
        print(f"  [gate] re-attached any_fill_ahead to {n:,} kept rows "
              f"(canonical derivation from `latency`)", flush=True)
    return block


def compact_design(block: dict) -> dict:
    """Pack PM+FN+ST into ONE float64 array and RELEASE the Python lists.

    MEASURED, and this is where the memory actually was. `_feature_pass`
    returns three parallel lists-of-lists per coin; at 105 features across
    1.08M rows (both coins) that is ~8 GB of Python float OBJECTS -- 24 bytes
    each, plus per-list headers -- against 0.45 GB for the identical numbers in
    one packed array. Two runs were OOM-killed carrying that into the topup
    pass, which is the peak.

    NUMERICALLY IDENTICAL, not an approximation: a Python float IS a float64,
    so `X[i].tolist()` returns exactly the values `PM[i] + FN[i] + ST[i]` did,
    in the same order. `build_design` keeps returning a `list`, so every
    downstream consumer sees the type it always saw.

    Called AFTER the embargo purge, which reindexes the three families and
    therefore needs them still to be lists.
    """
    import numpy as np
    n = len(block.get("kept") or ())
    if n == 0 or block.get("X") is not None:
        return block
    PM, FN, ST = block["PM"], block["FN"], block["ST"]
    w_pm, w_fn, w_st = len(PM[0]), len(FN[0]), len(ST[0])
    w = w_pm + w_fn + w_st
    X = np.empty((n, w), dtype=np.float64)
    for i in range(n):
        X[i] = PM[i] + FN[i] + ST[i]
    block["X"] = X
    # RECORD THE FAMILY WIDTHS, because packing DESTROYS the only thing that
    # said where one family ends and the next begins. The incumbent scores
    # PM+fine with NO state (arm D), so it needs the PM+FN PREFIX of this
    # array -- and a prefix length that is inferred rather than recorded is a
    # guess wearing an index. `_pm_fn_row` REFUSES a packed block without it.
    block["w"] = {"PM": w_pm, "FN": w_fn, "ST": w_st}
    # Drop the references so the arenas are reusable by the NEXT pass. RSS is
    # a high-water mark and CPython does not return arenas to the OS, so the
    # win is not a smaller peak here -- it is that the topup pass allocates
    # into space already held instead of growing past the cap.
    block["PM"] = block["FN"] = block["ST"] = None
    return block


def build_design(block: dict, i: int) -> list:
    """x for row i: PM + fine + state. Both arms see the SAME features; they
    differ in model class, not in what they are shown (R-232 9.1)."""
    X = block.get("X")
    if X is not None:
        return X[i].tolist()
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
            # A1.5 (I11-B3b): "Head fitting weights each row by
            # 1 / rows_in_generation, so a generation contributes once
            # regardless of how many decision rows it spans." Carried WITH the
            # targets so a head cannot be fitted without them -- the previous
            # arrangement had generation_weights written, unit-tested, and
            # reachable only from the test, while every fit passed unit weights.
            "w_rows": I11.generation_weights(rows),
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
        # A1.5: 1/rows_in_generation, so one generation contributes ONE unit of
        # mass however many decision rows it spans. Refused rather than
        # defaulted: a head fitted on unit weights answers a different question
        # from the one A1.5 froze, and it would look identical.
        W = tg.get("w_rows")
        if W is None or len(W) != len(Z):
            raise RuntimeError(
                f"REFUSED: fitting weights absent or misaligned "
                f"({None if W is None else len(W)} vs {len(Z)} rows). A1.5 "
                f"freezes 1/rows_in_generation; silently substituting unit "
                f"weights fits a different estimator under the same name.")

        def wsub(idx):
            return [W[i] for i in idx]
        w_fill = fc.fast_fit_logistic_w(Z, tg["y_fill"], W)
        w_pos = (fc.fast_fit_logistic_w(sub(ip), tg["y_pos"], wsub(ip))
                 if len(ip) >= MIN and len(set(tg["y_pos"])) > 1 else None)
        w_neg = (fc.fast_fit_logistic_w(sub(ip), tg["y_neg"], wsub(ip))
                 if len(ip) >= MIN and len(set(tg["y_neg"])) > 1 else None)
        wmh = (fc.fast_fit_ridge_w(sub(ih), tg["m_harm_target"],
                                   wsub(ih), lam=10.0)
               if len(ih) >= MIN else None)
        wmg = (fc.fast_fit_ridge_w(sub(ig), tg["m_good_target"],
                                   wsub(ig), lam=10.0)
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
    # A1.5 applies to EVERY head, not only the linear arm. LGBM took no
    # sample_weight at all, so the two arms were weighted differently while
    # being compared as though they differed only in model class (R-232 9.1).
    Wl = tg.get("w_rows")
    if Wl is None or len(Wl) != len(A):
        raise RuntimeError(
            f"REFUSED: fitting weights absent or misaligned "
            f"({None if Wl is None else len(Wl)} vs {len(A)} rows). A1.5 "
            f"freezes 1/rows_in_generation for head fitting.")
    Wa = np.asarray(Wl, dtype=np.float64)
    clf = lgb.LGBMClassifier(**D.LGBM_PARAMS)
    clf.fit(A, np.asarray(tg["y_fill"]), sample_weight=Wa)
    def cls(idx, y):
        if len(idx) < MIN or len(set(y)) < 2:
            return None
        m = lgb.LGBMClassifier(**D.LGBM_PARAMS)
        m.fit(A[idx], np.asarray(y), sample_weight=Wa[idx])
        return m
    def reg(idx, t):
        if len(idx) < MIN:
            return None
        m = lgb.LGBMRegressor(**D.LGBM_VALUE_PARAMS)
        m.fit(A[idx], np.asarray(t), sample_weight=Wa[idx])
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
    # THE GOVERNING INSTANT IS t0 + t_start. t_start is an offset WITHIN a
    # five-minute window, so t_start // 3600 is 0 for essentially every row and
    # the side x hour match collapsed every real UTC hour into one bucket --
    # matching on a constant is not matching (rule 7). Measured: two rows an
    # hour apart in real time both landed in hour 0.
    out = []
    for i in idx:
        r = rows[i]
        t, t0 = r.get("t_start"), r.get("t0")
        if t is None or t0 is None:
            out.append((r.get("side"), None))
            continue
        out.append((r.get("side"), int(((float(t0) + float(t)) // 3600) % 24)))
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
    # RR4-3: the SAME statistic at the ACTION unit, beside the row-level one.
    # Each head keeps its own population and index set, because collapsing a
    # head over rows it never scored would measure a different thing.
    for _nm, _kind, _pv, _av, _ix in (
            ("Q1_arrival", "probability", pred["p_fill"], tg["y_fill"], _all),
            ("Q2_p_pos", "probability", [pred["p_pos"][i] for i in ip],
             tg["y_pos"], ip),
            ("Q2_p_neg", "probability", [pred["p_neg"][i] for i in ip],
             tg["y_neg"], ip),
            ("Q3_m_harm", "magnitude", [pred["m_harm"][i] for i in ih],
             tg["m_harm_target"], ih),
            ("Q3_m_good", "magnitude", [pred["m_good"][i] for i in ig],
             tg["m_good_target"], ig)):
        heads[_nm]["action_unit"] = action_unit_metrics(
            _pv, _av, rows, _kind, idx=_ix)
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
            # RULE 10: this string said "min |calibration slope deviation|
            # side reported" beside code computing `min(slope)`. Those are
            # different sides whenever the slopes straddle 1 -- measured on
            # the 08-25 btc run, m_harm 0.6888 and m_good 0.9098 give
            # min-slope m_harm and min-|deviation| m_good, so the label named
            # the OTHER head's number. A conclusion printed beside a table
            # that contradicts it, for the fourth time in this programme.
            "Q3_cell_rule": "min(calibration slope of m_harm, m_good) — the "
                            "WORSE side under a gate that reads distance "
                            "from 0 (R-306's Q2-min logic); both slopes are "
                            "carried in `heads`. Deviation from 1 is the "
                            "REPORTED calibration diagnostic (R-286) and "
                            "selects nothing.",
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
    # RR2-1 adds GATE_PARTIALLY_EVALUATED. Registering it here is required
    # and is not a formality: this guard REFUSED the first dry emission over
    # an unknown status, which is the guard working -- a new status must be
    # DECLARED, never absorbed.
    KNOWN = {I11.CELL_STATUS_OK, I11.CELL_STATUS_UNDERPOWERED,
             I11.CELL_STATUS_NO_COUNTERPART, I11.CELL_STATUS_UNEVALUABLE,
             I11.CELL_STATUS_AGG_UNDECLARED, I11.CELL_STATUS_GATE_PARTIAL}
    hollow, badstatus, unsupported = [], [], []
    for key in sorted(declared):
        c = fam["cells"][key]
        if not isinstance(c, dict) or any(f not in c for f in REQUIRED_CELL_FIELDS):
            hollow.append(key); continue
        # I11-B4(2): the guard checked PRESENCE and one NaN. A cell carrying
        # every required key with arm/head/budget that disagree with its own
        # key, p=0.0 and n_actions=-999 sailed through as "cells_validated".
        # Identity first: a cell that misreports which cell it IS makes every
        # value in it unattributable.
        # EVERY problem in a cell is reported, not just the first. Stopping at
        # the earliest defect hides the rest, so a reader fixes one thing,
        # re-runs, and meets the next -- and a guard that names one fault in a
        # cell with three understates how wrong the cell is.
        _a, _h, _b = key.split("/")
        _bad: list = []
        if (c.get("arm"), c.get("head"), c.get("budget")) != (_a, _h, _b):
            _bad.append(f"identity disagrees with its own key -- carries "
                        f"arm={c.get('arm')!r} head={c.get('head')!r} "
                        f"budget={c.get('budget')!r}")
        _p = c.get("p_value")
        if _p is not None and (I11._num(_p) is None or not 0.0 < _p <= 1.0):
            _bad.append(f"p_value {_p!r} is outside (0, 1]; a permutation p is "
                        f"(1+k)/(1+n) and can never be 0, so a zero p is a "
                        f"computation that did not happen")
        _na = c.get("n_actions")
        if _na is not None and (not isinstance(_na, int) or isinstance(_na, bool)
                                or _na < 0):
            _bad.append(f"n_actions {_na!r} is not a non-negative count")
        if _bad:
            unsupported.append(f"{key}: " + "; ".join(_bad))
            continue
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
        stat, pval, status, sn, snb, detail, p2 = _q4_cell(arm, budget,
                                                            per_coin)
        return I11.build_cell(arm, head, budget, statistic=stat, p_value=pval,
                              status=status, n_actions=n_actions, detail=detail,
                              arrival_n=n_actions, statistic_n=sn,
                              statistic_n_basis=snb, p_two_sided=p2,
                              statistic_n_unit="windows")

    if head == "Q3_magnitudes" and len(per_coin) == 1:
        # R-306 RULED THIS AND THE CODE HAD NEVER IMPLEMENTED IT. The cell
        # below used to return AGGREGATION_UNDECLARED, correctly, because no
        # ruling reconciled A1.4's single "calibration slope" with the parent
        # table's two separately-gated slopes. R-306 (USER, 2026-08-29) is
        # that ruling and it lives in the frozen A1.4 amendment; a gap that a
        # ruling has closed is no longer a gap. The multi-coin case is a
        # SEPARATE gap and still falls through to the generic path.
        stat, pval, status, sn, snb, detail = _q3_compose(arm, per_coin)
        return I11.build_cell(arm, head, budget, statistic=stat, p_value=pval,
                              status=status, n_actions=n_actions, detail=detail,
                              arrival_n=n_actions, statistic_n=sn,
                              statistic_n_basis=snb,
                              statistic_n_unit="rows")

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
    # F1: the n behind THIS cell's statistic. Q1 scores every action; Q2's
    # ruled statistic is the WORSE SIDE's AUC (R-249) and both sides are
    # scored on the PREVENTABLE base, which is a different population from
    # the arrival one the cell was selected from.
    _sn, _snb, _su = None, "", "actions"
    if len(per_coin) == 1:
        _c, _r = next(iter(per_coin.items()))
        if head == "Q1_arrival":
            _m = _head_metric_n(_r[arm], "Q1_arrival")
            _sn, _su = _m["metric_n"], _m["metric_unit"]
            _snb = (f"Q1_arrival — the ROW count the AUC was computed on "
                    f"(RR4-3). ACTION-UNIT auc {_m['action_unit_value']!r} "
                    f"over {_m['action_n']!r} actions "
                    f"({_m['rows_per_action']!r} rows/action)")
        elif head == "Q2_sign":
            _hs = _r[arm].get("heads", {}) or {}
            _sc = [(_hs.get(x, {}).get("auc"), x) for x in ("Q2_p_pos", "Q2_p_neg")]
            _sc = [(a, x) for a, x in _sc if a is not None]
            if _sc:
                _ws = min(_sc)[1]
                _m = _head_metric_n(_r[arm], _ws)
                _sn, _su = _m["metric_n"], _m["metric_unit"]
                _snb = (f"Q2 worse side {_ws} (R-249) — the ROW count the AUC "
                        f"was computed on, over the PREVENTABLE base, not the "
                        f"arrival population (RR4-3). ACTION-UNIT auc "
                        f"{_m['action_unit_value']!r} over {_m['action_n']!r} "
                        f"actions ({_m['rows_per_action']!r} rows/action)")
    good = [x for x in stats if x is not None]
    if not good:
        return I11.build_cell(
            arm, head, budget, status=I11.CELL_STATUS_UNEVALUABLE,
            n_actions=n_actions, arrival_n=n_actions,
            detail=f"no coin produced an adjudicable statistic; per-coin "
                   f"statuses {statuses}")
    if any(st not in ("OK", None) for st in statuses):
        return I11.build_cell(
            arm, head, budget, statistic=min(good),
            status=I11.CELL_STATUS_UNDERPOWERED, n_actions=n_actions,
            arrival_n=n_actions, statistic_n=_sn, statistic_n_basis=_snb,
            statistic_n_unit="actions",
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
    # Q3's two sides are NOT a gap either, as of R-306 (USER, 2026-08-29):
    # CONJUNCTION + WORSE SIDE, implemented in `_q3_compose` and reached
    # above. What remains here is only the COIN axis, whose R-306 clause
    # (btc-only adjudication) this runner does not implement -- so a
    # multi-coin Q3 still says so rather than collapsing.
    ps = _cell_p(per_coin, arm, head)
    if ambiguous:
        return I11.build_cell(
            arm, head, budget, statistic=min(good), p_value=None,
            status=I11.CELL_STATUS_AGG_UNDECLARED, n_actions=n_actions,
            arrival_n=n_actions, statistic_n=_sn, statistic_n_basis=_snb,
            statistic_n_unit="actions",
            detail="AGGREGATION UNDECLARED, stated for ruling rather than "
                   "chosen (I11-B4): " + "; ".join(ambiguous) +
                   f". The worst-coin statistic {min(good)!r} is carried as a "
                   f"placeholder ONLY -- it is not adjudicated, p is withheld, "
                   f"and the per-coin evidence is {per_coin_ev!r} so a ruling "
                   f"can be applied without re-running.")
    if head == "Q2_sign":
        # Q2's frozen gate DOES name an incumbent and the incumbent has no
        # sign head, so R-237's no-counterpart status is correct here. R-397
        # ruling 2 removes it from Q3 ONLY, whose gate never asked.
        return I11.build_cell(
            arm, head, budget, statistic=min(good), p_value=(ps[0] if ps else None),
            status=I11.CELL_STATUS_NO_COUNTERPART, n_actions=n_actions,
            arrival_n=n_actions, statistic_n=_sn, statistic_n_basis=_snb,
            statistic_n_unit="actions",
            detail=f"R-237: the incumbent has no sign/magnitude head, so no "
                   f"INCREMENTAL null exists for this head. The p reported is "
                   f"the MATCHED-RANDOM null (prereg §5.1), which does apply; "
                   f"the two nulls are named so neither is read as the other.")
    return I11.build_cell(
        arm, head, budget, statistic=min(good),
        p_value=(ps[0] if ps else None),
        status=I11.CELL_STATUS_OK if ps else I11.CELL_STATUS_UNEVALUABLE,
        n_actions=n_actions, arrival_n=n_actions,
        statistic_n=_sn, statistic_n_basis=_snb, statistic_n_unit=_su,
        detail=(f"single coin, single head: adjudicated on its own "
                f"matched-random null (prereg §5.1), {len(ps)} p available"
                if ps else
                "no matched-random null was computable for this head, so there "
                "is no null evidence and the cell is not OK"))


# ---------------------------------------------------------------------------
# Q4's INCUMBENT: LOAD, VERIFY, APPLY  (R-280 ruling)
# ---------------------------------------------------------------------------
# prereg §5(2) needs "the head's metric minus THE INCUMBENT'S on the IDENTICAL
# action population". The definite article names the COMMITTED incumbent, so:
#   - never RE-FIT inside 011 -- a re-fit incumbent is a different incumbent and
#     would silently answer a different question;
#   - never load STALE SCORES -- scores over different rows are not the
#     identical population, whatever their column names say;
#   - LOAD the four-arm stack's artifact, VERIFY it by hash, APPLY it here.
#
# IDENTITY, CHECKED NOT ASSUMED: INCUMBENT_REWEIGHTED_ONLY is ARM D, whose model
# is FITDIR/linear_d_<coin>.json. It is NOT harmful_reduced_fine_candidate_v1
# .json -- that file is arm A's frozen linear (the manifest's
# frozen_incumbent_sha256_prefix), and applying it here would be a different arm
# wearing the incumbent's name.
INCUMBENT_ARM = "INCUMBENT_REWEIGHTED_ONLY"


def _sha16(p) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def load_verified_incumbent(coin: str, fitdir=None, manifest: dict = None) -> dict:
    """The COMMITTED incumbent for `coin`, or a REFUSAL naming what failed."""
    fitdir = Path(fitdir) if fitdir is not None else PA.FITDIR
    art = fitdir / f"linear_d_{coin}.json"
    if not art.exists():
        raise RuntimeError(
            f"REFUSED: no incumbent artifact at {art}. Q4's increment is "
            f"defined against the COMMITTED incumbent; fitting one here would "
            f"answer a different question (R-280).")
    if manifest is None:
        mf = fitdir / PA.FIT_MANIFEST
        if not mf.exists():
            raise RuntimeError(
                f"REFUSED: no {PA.FIT_MANIFEST} beside {art.name}, so its "
                f"identity cannot be verified. An UNVERIFIED incumbent is not "
                f"the incumbent, it is a file with a familiar name.")
        manifest = json.loads(mf.read_text())
    want = (manifest.get("file_hashes") or {}).get(art.name)
    if want is None:
        raise RuntimeError(
            f"REFUSED: the fit manifest does not bind {art.name}. A hash the "
            f"manifest never recorded cannot be checked against it, and an "
            f"absent binding must never read as a matching one (R-225(1b)).")
    got = _sha16(art)
    if got != want:
        raise RuntimeError(
            f"REFUSED: {art.name} is not the committed incumbent "
            f"(manifest={want!r} now={got!r}). A DIFFERENT incumbent gives a "
            f"different increment, and the numbers cannot show which was used.")
    d = json.loads(art.read_text())
    if d.get("arm") != INCUMBENT_ARM:
        raise RuntimeError(
            f"REFUSED: {art.name} identifies as arm {d.get('arm')!r}, not "
            f"{INCUMBENT_ARM!r}. The prereg's increment names ONE arm.")
    for k in ("norm_mu", "norm_sd", "hazard_weights", "value_weights"):
        if k not in d:
            raise RuntimeError(f"REFUSED: incumbent artifact has no {k!r}.")
    nmu = len(d["norm_mu"])
    if len(d["hazard_weights"]) != nmu + 1:
        raise RuntimeError(
            f"REFUSED: {len(d['hazard_weights'])} hazard weights against "
            f"{nmu} scalers. The design carries ONE intercept, so those widths "
            f"must differ by exactly one; anything else means the artifact and "
            f"the design disagree about what a row is.")
    # State what was verified AND what was NOT. This checks the ARTIFACT's
    # identity, not the whole fit chain, and those are different claims.
    d["_verified"] = {
        "artifact": art.name, "sha256_prefix": got,
        "bound_by": "fit_manifest.file_hashes",
        "not_verified": "the manifest's OWN bindings (tape, gate, fit code) are "
                        "checked by assert_fit_complete_and_matching, not here; "
                        "this proves only that this file is the one the "
                        "manifest recorded"}
    return d


def action_unit_metrics(pred: list, actual: list, rows: list, kind: str,
                        idx=None) -> dict:
    """The SAME statistic, measured once per ACTION instead of once per row.

    RR4-3, and CLAUDE.md rule 2 is the reason: "rows are actions... if several
    rows can share one outcome, the evaluator must de-duplicate to actions or
    the result is inflated (measured: 1.99 rows/fill, max 23)". `head_report`
    uses `gen_keys` to decide UNDERPOWERED and computes the METRIC over the
    full row vector, so every surviving statistic is a row-level number
    reported under an action-level n — measured here at 1.754 rows/action for
    Q1 and up to 1.992 for Q3_m_harm.

    NO RE-FIT AND NO NEW ESTIMAND. The model's predictions are untouched; this
    collapses them to one value per (slug, side, gen) and recomputes the same
    metric. It is reported BESIDE the row-level number, never instead of it —
    which of the two adjudicates is the USER's question, not a seat's.

    THE COLLAPSE RULE IS A CHOICE, SO ALL THREE ARE REPORTED. `max` is not
    invented here: `q4_economics` already ranks generations by their maximum
    score, and AUC is a ranking metric, so the ranking convention already in
    the codebase is the primary. `mean` and `first` (earliest decision
    instant) are computed beside it, so a reader sees whether the answer
    depends on the rule rather than taking one on trust.

    THE LABEL IS THE SHARED OUTCOME. A generation's rows share one fill, so
    the action's label is 1 if ANY of its rows carries the outcome. Generations
    whose rows DISAGREE are counted and reported: if that count is large the
    "shared outcome" premise is itself wrong, and that is a finding rather
    than a detail."""
    idx = list(range(len(rows))) if idx is None else list(idx)
    if not (len(pred) == len(actual) == len(idx)):
        raise RuntimeError(
            f"REFUSED: action-unit collapse needs aligned vectors; got pred "
            f"{len(pred)}, actual {len(actual)}, idx {len(idx)}.")
    gens: dict = {}
    for j, i in enumerate(idx):
        r = rows[i]
        gens.setdefault((r.get("slug"), r.get("side"), r.get("gen")),
                        []).append(j)
    mixed = 0
    order_max, order_mean, order_first, labels = [], [], [], []
    for k, js in sorted(gens.items()):
        ys = [actual[j] for j in js]
        if len(set(ys)) > 1:
            mixed += 1
        ps = [pred[j] for j in js]
        order_max.append(max(ps))
        order_mean.append(math.fsum(ps) / len(ps))
        first = min(js, key=lambda j: (rows[idx[j]].get("t_start"), j))
        order_first.append(pred[first])
        if kind == "probability":
            labels.append(1 if any(ys) else 0)
        else:
            labels.append(math.fsum(ys) / len(ys))
    def metric(scores):
        if kind == "probability":
            return I11.auc(scores, labels)
        return I11.calibration_slope(scores, labels)
    return {
        "unit": "ACTION (distinct slug/side/gen)",
        "n_actions": len(gens), "n_rows": len(idx),
        "rows_per_action": (round(len(idx) / len(gens), 4) if gens else None),
        "collapse_primary": "max",
        "value": metric(order_max),
        "by_collapse_rule": {"max": metric(order_max),
                             "mean": metric(order_mean),
                             "first": metric(order_first)},
        "generations_with_disagreeing_row_labels": mixed,
        "label_rule": ("any row carries the outcome (probability heads); "
                       "mean of the rows' targets (magnitude heads)"),
        "why": "rule 2: several rows share one outcome, so a row-level metric "
               "lets a generation with more rows count more than once. "
               "Reported BESIDE the row-level number; which adjudicates is a "
               "USER question (RR4-3)."}


def q1_incremental(cand_p: list, inc_haz: dict, rows: list,
                   y_fill: list) -> dict:
    """Q1's SECOND declared conjunct: does the candidate beat the incumbent
    hazard head on the identical action population? R-397 ruling 1.

    THE GATE IS A CONJUNCTION OF TWO DIFFERENT KINDS OF THING, and reading it
    literally is what keeps this honest. Frozen prereg 3, Q1:

        "beats the matched-random null AND beats the incumbent hazard head"

    The first conjunct is a NULL TEST and carries a p. The second is a
    COMPARISON on the identical population and is a BOOLEAN. Composing them
    into a single p would require a rule nobody has ruled, so nothing is
    composed: the adjudicated p stays the matched-random one, and the
    comparison is reported as the boolean the gate asks for.

    THE 5(2) INCREMENT NULL IS ALSO COMPUTED, AND ITS LIMIT IS STATED.
    Prereg 5(2) declares "window-level sign-flip permutation of per-window
    paired differences" for the increment. That design is additive: for Q4 the
    per-window values SUM to the statistic. **AUC does not sum.** The
    population AUC difference is NOT the mean of per-window AUC differences,
    so this null describes the MEAN PER-WINDOW difference, not the declared
    population statistic. It is therefore REPORTED and never adjudicated, and
    the mismatch is named here rather than resolved by picking one — that
    would be exactly the "p that does not describe the number beside it"
    defect (I11-B5) the programme already fixed once.

    Windows where either side's AUC is undefined (one class present) are
    EXCLUDED BY NAME and counted (rule 4), never silently dropped."""
    # THE INCUMBENT MUST IDENTIFY ITSELF, not merely supply numbers.
    # Measured: replacing `apply_incumbent_hazard(...)` with a constant
    # `{"p_fill": [0.0] * n}` produced a perfectly well-formed comparison
    # (all-tied scores give AUC 0.5), so the leg READ as computed, the seam
    # passed, and Q1's gate reported a beat against nothing. A number cannot
    # say where it came from; the verified artifact hash can.
    _arm = inc_haz.get("arm") if isinstance(inc_haz, dict) else None
    if _arm != INCUMBENT_ARM:
        raise RuntimeError(
            f"REFUSED: Q1's incumbent leg was handed "
            f"{type(inc_haz).__name__} declaring arm {_arm!r}, not "
            f"{INCUMBENT_ARM!r}. A vector of numbers is not the incumbent "
            f"hazard head.")
    prov = inc_haz.get("provenance") or {}
    if not prov.get("sha256_prefix"):
        raise RuntimeError(
            "REFUSED: Q1's incumbent leg carries no verified provenance. The "
            "comparison is only meaningful against the COMMITTED incumbent "
            "(R-280), and an unidentified predictor would let a constant "
            "vector pass as one.")
    if inc_haz.get("head") != "Q1_arrival":
        raise RuntimeError(
            f"REFUSED: Q1's leg was handed the {inc_haz.get('head')!r} head; "
            f"the gate names the incumbent HAZARD head specifically.")
    inc_p = inc_haz["p_fill"]
    n = len(rows)
    if not (len(cand_p) == len(inc_p) == len(y_fill) == n):
        raise RuntimeError(
            f"REFUSED: Q1's incremental leg needs the IDENTICAL action "
            f"population (prereg 5.2); got candidate {len(cand_p)}, incumbent "
            f"{len(inc_p)}, labels {len(y_fill)}, rows {n}.")
    a_c, a_i = I11.auc(cand_p, y_fill), I11.auc(inc_p, y_fill)
    increment = None if (a_c is None or a_i is None) else a_c - a_i
    by_w: dict = {}
    for i, r in enumerate(rows):
        by_w.setdefault(r.get("slug"), []).append(i)
    diffs, excl = {}, {"single_class_window": 0, "too_few_rows": 0}
    for w, idx in sorted(by_w.items()):
        ys = [y_fill[i] for i in idx]
        if len(idx) < 2:
            excl["too_few_rows"] += 1
            continue
        wc = I11.auc([cand_p[i] for i in idx], ys)
        wi = I11.auc([inc_p[i] for i in idx], ys)
        if wc is None or wi is None:
            excl["single_class_window"] += 1
            continue
        diffs[w] = wc - wi
    null = I11.sign_flip_null(diffs) if diffs else None
    _au_c = action_unit_metrics(cand_p, y_fill, rows, "probability")
    _au_i = action_unit_metrics(inc_p, y_fill, rows, "probability")
    _au_inc = (None if (_au_c["value"] is None or _au_i["value"] is None)
               else _au_c["value"] - _au_i["value"])
    return {
        "head": "Q1_arrival",
        "candidate_auc": a_c, "incumbent_auc": a_i,
        "increment_auc": increment,
        "beats_incumbent_hazard_head": (None if increment is None
                                        else increment > 0.0),
        # RR4-3: this was labelled `n_actions` while holding the ROW count,
        # which A1.5 forbids outright ("no head may report an action count
        # larger than its population's distinct (slug, side, gen) count").
        # Both are carried, each under its own name.
        "n_rows": n,
        "n_actions": len({(r.get("slug"), r.get("side"), r.get("gen"))
                          for r in rows}),
        "auc_unit": "ROW — candidate_auc/incumbent_auc/increment_auc above "
                    "are computed over ROWS; the action-unit pair is below",
        # RR4-3: THE SAME COMPARISON AT THE ACTION UNIT. Q1's gate conjunct is
        # "beats the incumbent hazard head", and whether it beats can depend
        # on the unit, so both are computed and neither is hidden. Both sides
        # are collapsed by the SAME rule on the SAME generations, so the
        # comparison stays paired.
        "action_unit": {
            "candidate": _au_c, "incumbent": _au_i,
            "increment_auc": _au_inc,
            "beats_incumbent_hazard_head": (None if _au_inc is None
                                            else _au_inc > 0.0),
            "agrees_with_row_level": (
                None if (_au_inc is None or increment is None)
                else (_au_inc > 0.0) == (increment > 0.0)),
            "why": "RR4-3: the gate conjunct is a comparison, and a "
                   "comparison can depend on the unit it is measured at. "
                   "Both units are computed on the same generations with the "
                   "same collapse rule; whether they AGREE is the fact a "
                   "reader needs, so it is computed rather than assumed."},
        "incumbent_provenance": prov,
        "windows_total": len(by_w), "windows_used": len(diffs),
        "windows_excluded": excl,
        "increment_null_REPORTED_NOT_ADJUDICATED": (
            None if null is None else {
                "p_value": null["p_value"],
                "p_two_sided": null.get("p_two_sided"),
                "n_units": null["n_units"], "n_perm": null["n_perm"],
                "observed_mean_per_window_auc_difference": null["observed"]}),
        "why_the_null_is_not_adjudicated":
            "prereg 5(2)'s sign-flip design is ADDITIVE — for Q4 the "
            "per-window values sum to the statistic. AUC does not sum, so the "
            "population AUC difference is not the mean of per-window AUC "
            "differences and this p describes the mean per-window difference, "
            "not the declared statistic. Adjudicating it would put a p beside "
            "a number it does not describe (I11-B5). The GATE's second "
            "conjunct is a comparison, and that is what is adjudicated.",
        "arm_note": "the incumbent is INCUMBENT_REWEIGHTED_ONLY's HAZARD head "
                    "(probability before the value head), applied to the same "
                    "rows by apply_incumbent_hazard (R-280, R-397 ruling 1)"}


def _pm_fn_row(block: dict, j: int, n_expected: int) -> list:
    """Row j's PM+fine features, from whichever representation the block holds.

    THE PACK BROKE THIS PATH AND NOTHING SAW IT. `compact_design` sets
    block["PM"] = block["FN"] = block["ST"] = None after packing, so the
    arm-D arithmetic -- `block["PM"][j] + block["FN"][j]` -- raises
    TypeError on any real block by the time the results loop runs. Every
    falsifier for the incumbent path builds its own UNPACKED fixture, so the
    suite could not see it: rule 17's shape again, one layer below the wiring.

    The prefix is CHECKED, never assumed. `compact_design` records the family
    widths; a packed block without them REFUSES rather than slicing by a
    length inferred from the model, which would make a width mismatch look
    like a correct read of a differently-shaped row."""
    X = block.get("X")
    if X is None:
        return block["PM"][j] + block["FN"][j]          # NO state features
    w = block.get("w")
    if not w:
        raise RuntimeError(
            "REFUSED: a PACKED design block carries no family widths, so the "
            "PM+fine prefix cannot be located. Slicing to the model's own "
            "width would make a shape disagreement read as a correct row.")
    k = w["PM"] + w["FN"]
    if k != n_expected:
        raise RuntimeError(
            f"REFUSED: the packed block's PM+fine width is {k} "
            f"({w['PM']}+{w['FN']}) but the incumbent was fitted on "
            f"{n_expected}. The prefix and the model disagree about what a "
            f"row is; a slice would still return {n_expected} numbers.")
    return X[j, :k].tolist()


def apply_incumbent(model: dict, block: dict, idx) -> dict:
    """The incumbent's composed value on the SAME rows the candidate scored.

    The arithmetic is COPIED from the four-arm apply path (phase2_arms.py, the
    INCUMBENT_REWEIGHTED_ONLY branch) rather than reinvented: a second
    implementation of a scoring rule IS a second rule, and the disagreement
    would surface as an increment instead of as an error."""
    import harmful_fast_compute as fc
    mu, sd = model["norm_mu"], model["norm_sd"]
    W, WM = model["hazard_weights"], model["value_weights"]
    out = []
    for j in idx:
        raw = _pm_fn_row(block, j, len(mu))            # NO state features
        if len(raw) != len(mu):
            raise RuntimeError(
                f"REFUSED: row {j} has {len(raw)} PM+fine features but the "
                f"incumbent was fitted on {len(mu)}. Applying weights to a "
                f"differently-shaped vector yields a number, not a prediction.")
        x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
        ph = fc.fast_predict_p(W, x)
        vh = float(sum(a * b for a, b in zip(WM, x))) if WM else 0.0
        out.append(ph * vh)
    return {"expected_cancel_value": out, "arm": INCUMBENT_ARM, "n": len(out),
            "provenance": model.get("_verified")}


def apply_incumbent_hazard(model: dict, block: dict, idx) -> dict:
    """The incumbent's HAZARD head on the same rows -- Q1's counterpart.

    Q1's frozen gate is "beats the matched-random null AND beats the incumbent
    hazard head". Unlike Q2/Q3, the incumbent HAS this head
    (INCUMBENT_COMPARABLE["Q1_arrival"] is True), so R-237's
    no-counterpart excuse does not reach Q1 and its incremental leg is
    required rather than optional.

    Same artifact, same verification, same arithmetic as apply_incumbent --
    this returns the probability BEFORE it is multiplied by the value head,
    because Q1 is about ARRIVAL, not about composed value."""
    import harmful_fast_compute as fc
    mu, sd = model["norm_mu"], model["norm_sd"]
    W = model["hazard_weights"]
    out = []
    for j in idx:
        raw = _pm_fn_row(block, j, len(mu))            # NO state features
        if len(raw) != len(mu):
            raise RuntimeError(
                f"REFUSED: row {j} has {len(raw)} PM+fine features but the "
                f"incumbent hazard head was fitted on {len(mu)}.")
        x = [1.0] + [(raw[i] - mu[i]) / sd[i] for i in range(len(mu))]
        out.append(fc.fast_predict_p(W, x))
    return {"p_fill": out, "arm": INCUMBENT_ARM, "n": len(out),
            "head": "Q1_arrival",
            "provenance": model.get("_verified")}


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
    actually reports, not a sibling of it.

    For Q3 it is `max(p)` -- the WORSE of the two, ruled by R-306. This branch
    took the FIRST sub-head's p (Q3_m_harm, by dict order), which is neither
    side's ruled role: it happened to be right whenever m_harm was the weaker
    side and silently wrong whenever it was not. `_q3_compose` is the
    single-coin path and computes this itself; this keeps the two in step for
    the multi-coin path, which still reports its own gap."""
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
        elif head == "Q3_magnitudes" and len(subs) > 1:
            ps = [(heads.get(x, {}).get("matched_random") or {}).get("p_value")
                  for x in subs]
            ps = [x for x in ps if x is not None]
            # CONJUNCTION: both sides or nothing. A cell whose second side has
            # no null has not met the pass condition; taking the one p present
            # would let half a head carry the cell.
            v = max(ps) if len(ps) == len(subs) else None
        else:
            v = None
            for x in subs:
                v = (heads.get(x, {}).get("matched_random") or {}).get("p_value")
                break
        if v is not None:
            out.append(v)
    return out


def _head_metric_n(r_arm: dict, sub: str):
    """The n the sub-head's METRIC was computed on, with its UNIT. RR4-3.

    `head_report` decides UNDERPOWERED from the action count and computes the
    METRIC over the full row vector, so the two are different numbers and the
    cell was reporting the wrong one: measured 1.754 rows/action on Q1, up to
    1.992 on Q3_m_harm. The row count is what the statistic was computed on;
    the action count is what the action-unit statistic beside it uses."""
    h = (r_arm.get("heads", {}) or {}).get(sub) or {}
    n_rows, n_act = h.get("n_rows"), h.get("n_actions")
    au = (h.get("action_unit") or {})
    return {"metric_n": n_rows if isinstance(n_rows, int) else n_act,
            "metric_unit": "rows" if isinstance(n_rows, int) else "actions",
            "action_n": n_act,
            "action_unit_value": au.get("value"),
            "rows_per_action": au.get("rows_per_action")}


def _head_n(r_arm: dict, sub: str):
    """The ACTION count the sub-head `sub` was scored on, or None.

    READ from where it was computed (head_report), never re-derived: two
    derivations of a population is how the two n diverge in the first place.
    """
    h = (r_arm.get("heads", {}) or {}).get(sub) or {}
    n = h.get("n_actions")
    return n if isinstance(n, int) else None


def _cell_n_draws(receipt: dict, cell: dict):
    """The draw count behind THIS cell's adjudicated p, read from the heads.

    Read, never assumed: Q4's p comes from the sign-flip null (n_perm) and
    Q1/Q2/Q3's from the matched-random null (n_draws), and the two counts
    DIFFER in this artifact -- which is itself part of what F-3 exposes."""
    head, arm = cell.get("head"), cell.get("arm")
    if head == "Q4_combined_ev":
        return I11.N_PERM_011
    subs = _CELL_SUBHEADS.get(head, ())
    for arms in (receipt.get("results") or {}).values():
        hs = ((arms.get(arm) or {}).get("heads")) or {}
        for sub in subs:
            n = ((hs.get(sub) or {}).get("matched_random") or {}).get("n_draws")
            if n:
                return n
    return None


def _holm_over(pvals: dict, m: int) -> dict:
    """Holm step-down over `pvals` with denominator `m`. RR4-2.

    The same arithmetic `assemble_family` runs, applied to a HYPOTHETICAL p
    vector so the one-draw consequence can be COMPUTED rather than multiplied."""
    order = sorted(pvals, key=lambda k: pvals[k])
    adj, prev = {}, 0.0
    for i, k in enumerate(order):
        a = max(min(1.0, pvals[k] * (m - i)), prev)
        adj[k] = a
        prev = a
    return adj


def attach_floor_disclosure(receipt: dict) -> dict:
    """Every cell at its null's floor SAYS SO -- above all a surviving one."""
    cells = ((receipt.get("family") or {}).get("cells")) or {}
    m = (receipt.get("family") or {}).get("holm_denominator") or 24
    at, seen = [], 0
    for key, c in cells.items():
        nd = _cell_n_draws(receipt, c)
        d = permutation_floor_disclosure(c.get("p_value"), nd, m)
        c["permutation_floor"] = d
        seen += 1
        if d.get("at_permutation_floor"):
            at.append(key)
    surv_at = sorted(k for k in at
                     if cells[k].get("survives_joint_reading_at_0_05"))
    # RR4-2: BOTH one-draw numbers, COMPUTED. `permutation_floor_disclosure`
    # multiplies by the full denominator, which is the WHOLE-FAMILY case: if
    # every at-floor cell moved together the multiplier stays m. If only ONE
    # cell moves it sorts BEHIND the still-tied cells and Holm's step-down
    # gives it a SMALLER multiplier, so the single-cell consequence is milder
    # than the multiplied figure implies. The reviewer's round-1 framing (and
    # mine, taken from it) overstated it; both are now run through the real
    # step-down instead of being reasoned about.
    base_p = {k: c["p_value"] for k, c in cells.items()
              if c.get("p_value") is not None}
    for key, c in cells.items():
        d = c.get("permutation_floor") or {}
        if not d.get("at_permutation_floor"):
            continue
        nd = d.get("n_draws")
        nxt = 2.0 / (nd + 1) if nd else None
        if nxt is None:
            continue
        one = dict(base_p); one[key] = nxt
        allf = dict(base_p)
        for k2 in at:
            allf[k2] = 2.0 / ((cells[k2]["permutation_floor"]["n_draws"]) + 1)
        h1 = _holm_over(one, m).get(key)
        h2 = _holm_over(allf, m).get(key)
        d["holm_if_ONE_draw_beat_it_in_THIS_CELL_ONLY"] = h1
        d["survives_if_THIS_CELL_ONLY_moved"] = (h1 is not None and h1 < 0.05)
        d["holm_if_ONE_draw_beat_it_in_EVERY_at_floor_CELL"] = h2
        d["survives_if_EVERY_at_floor_cell_moved"] = (h2 is not None
                                                      and h2 < 0.05)
        d["one_draw_numbers_are_COMPUTED"] = (
            "both run through the real Holm step-down on the perturbed p "
            "vector, not obtained by multiplying. A single cell moving sorts "
            "BEHIND the still-tied cells and gets a smaller multiplier, so "
            "the single-cell consequence is milder than the whole-family one "
            "(RR4-2).")
    receipt["family"]["permutation_floor_summary"] = {
        "cells_read": seen, "cells_at_floor": len(at),
        "SURVIVING_cells_at_floor": surv_at,
        "draw_counts_are_not_uniform": sorted(
            {_cell_n_draws(receipt, c) for c in cells.values()}
            - {None}),
        "why": "F-3: a surviving cell sitting on its null's floor survives by "
               "ONE DRAW. The draw counts differ by head in this artifact "
               "(the increment null runs at the A1.6-pinned n_perm; the "
               "matched-random null runs at its own declared constant), so "
               "the head that survives and the head that fails were not "
               "measured at the same resolution -- stated, not corrected: "
               "changing a declared constant after seeing the result is "
               "rule 11's territory and is ESCALATED, not done here."}
    return receipt["family"]["permutation_floor_summary"]


def permutation_floor_disclosure(p_value, n_draws, m: int = 24) -> dict:
    """Is this cell's p the SMALLEST its null can produce, and by what margin?

    F-3. Every Q1/Q2/Q3 cell sits at p = 1/501 EXACTLY -- 0 of 500 draws beat
    the observed -- and Q1, the head that SURVIVES, carried no floor language
    at all while Q3, which decides nothing, carried it on all six cells. The
    disclosure was present exactly where it did not matter.

    The margin is the point and it is arithmetic, not rhetoric: at the floor
    the family clears the bar at holm = m/(n+1); had ONE draw of the n beaten
    the observed, p = 2/(n+1) and holm doubles. Stated per cell so a reader of
    a SURVIVING cell sees how wide its survival is."""
    if p_value is None or not n_draws:
        return {"at_permutation_floor": False,
                "why": "no p or no draw count, so the floor is undefined here"}
    floor = 1.0 / (n_draws + 1)
    at = abs(p_value - floor) < 1e-15
    nxt = 2.0 / (n_draws + 1)
    return {
        "at_permutation_floor": at,
        "n_draws": n_draws,
        "floor_p": floor,
        "holm_at_floor": min(1.0, m * floor),
        "next_attainable_p": nxt,
        # MULTIPLIED, and named as the whole-family case it actually is.
        # `attach_floor_disclosure` adds the two COMPUTED numbers beside it.
        "holm_if_EVERY_at_floor_cell_moved_together_multiplied": min(1.0, m * nxt),
        "margin_in_draws": 1 if at else None,
        "why": ("AT THE FLOOR: 0 of {n} draws reached the observed, so this p "
                "is the smallest this null can express and the result is "
                "RESOLUTION-LIMITED. One draw the other way moves holm from "
                "{a:.4f} to {b:.4f}. A p at the floor is a bound, not a "
                "measurement.").format(n=n_draws, a=min(1.0, m * floor),
                                       b=min(1.0, m * nxt))
        if at else "not at the floor; the null had room to express this p"}


def _q3_compose(arm: str, per_coin: dict) -> tuple:
    """Q3's cell under R-306: CONJUNCTION + WORSE SIDE.

    Returns `(statistic, p_value, status, statistic_n, statistic_n_basis,
    detail)`.

    THE RULING, quoted from the frozen A1.4 amendment (USER, 2026-08-29):
    "Q3's TWO ruled slope gates compose into its single cell as CONJUNCTION +
    WORSE SIDE -- the cell PASSES only if BOTH slope CIs (m_harm, m_good)
    exclude 0, and the cell's adjudicated p is the WORSE of the two (the
    Q2-min logic: half a working head cannot carry a cell). Family stays 24."

    Implemented as the ruling states it, in three separable parts:

    CONJUNCTION.  Both sides must be EVALUABLE, or the CELL is -- the same
    rule Q2 already carries ("a single side never carries it"). With both
    evaluable, adjudicating on the WORSE p at a common alpha IS the
    conjunction: it is the intersection-union p, which rejects only when both
    sides do. So the p and the pass condition are one mechanism, not two.

    WORSE SIDE.  Two things are named and they are computed separately,
    because they need not come from the same side:
      * the adjudicated p is `max(p_harm, p_good)` -- "the WORSE of the two",
        the ruling's literal words, worse meaning weaker evidence;
      * the reported statistic is `min(slope)` -- the Q2-min logic the
        parenthetical invokes, under a gate that reads DISTANCE FROM 0, where
        the smaller slope is the binding one for "both exclude 0".
    When the two disagree about WHICH side is worse, the disagreement is
    DISCLOSED in the detail rather than resolved by preference.

    "CI EXCLUDES 0" IS TESTED IN ITS IMPLEMENTED FORM, AND THAT FORM IS
    CHECKED, NOT ASSUMED.  R-286 ruled that the adjudicated null is the GATE's
    text and that "the implementation's `no_skill_value` in-band declaration
    is the visibility mechanism of record" -- the matched-random null with
    `no_skill_value = 0.0`, one-sided `greater`. This function REFUSES a
    sub-head whose null was declared against any other no-skill value or
    alternative, because a p computed against 1.0 (the REPORTED calibration
    diagnostic, never adjudicated per R-286) would be a different question
    wearing the gate's name.

    A LITERAL INTERVAL IS NOT CLAIMED AND MUST NOT BE READ IN. Rule 8 forbids
    an interval below G=5 complete UTC days and this population is at G=0; the
    disclosure travels in the cell's detail so "CI excludes 0" is never read
    as an interval that was computed."""
    if len(per_coin) != 1:
        raise RuntimeError(
            f"REFUSED: _q3_compose adjudicates ONE coin ({len(per_coin)} "
            f"supplied). R-306's coin clause (btc-only adjudication) is not "
            f"implemented in this runner; the multi-coin gap is reported by "
            f"the generic path, never collapsed here.")
    coin, r = next(iter(per_coin.items()))
    heads = r[arm].get("heads", {})
    subs = _CELL_SUBHEADS["Q3_magnitudes"]
    ev, missing, under, wrongnull = {}, [], [], []
    for sub in subs:
        h = heads.get(sub) or {}
        mr = h.get("matched_random") or {}
        slope, pv = h.get("calibration_slope"), mr.get("p_value")
        if h.get("status") == I11.UNDERPOWERED or mr.get("status") == I11.UNDERPOWERED:
            under.append(sub)
        if slope is None or pv is None:
            missing.append(sub)
            continue
        nsv, alt = mr.get("no_skill_value"), mr.get("alternative")
        if nsv != 0.0 or alt != "greater":
            wrongnull.append(f"{sub}(no_skill_value={nsv!r}, "
                             f"alternative={alt!r})")
            continue
        ev[sub] = {"calibration_slope": slope, "matched_random_p": pv}
    # CONJUNCTION: a cell whose second side is absent has not been measured.
    if under:
        return (None, None, I11.CELL_STATUS_UNDERPOWERED, None, "",
                f"R-306 CONJUNCTION: Q3 needs BOTH slope gates and "
                f"{sorted(under)} is underpowered on {coin}. Half a working "
                f"head cannot carry a cell, so the CELL is underpowered and "
                f"its p is withheld; the per-side evidence is {ev!r}.")
    if wrongnull:
        return (None, None, I11.CELL_STATUS_UNEVALUABLE, None, "",
                f"R-306/R-286: the adjudicated gate is 'calibration slope CI "
                f"excludes 0', whose implemented form is the matched-random "
                f"null at no_skill_value=0.0, one-sided 'greater'. These "
                f"sub-heads declared a different null: {sorted(wrongnull)}. A "
                f"p against another no-skill value answers another question "
                f"and must not be adjudicated under this gate's name.")
    if missing:
        return (None, None, I11.CELL_STATUS_UNEVALUABLE, None, "",
                f"R-306 CONJUNCTION: {sorted(missing)} carry no slope or no "
                f"matched-random p on {coin}, so the conjunction cannot be "
                f"evaluated and a single side never carries the cell. "
                f"Per-side evidence: {ev!r}.")
    sh, sg = ev[subs[0]], ev[subs[1]]
    # WORSE SIDE, computed twice under the ruling's two names.
    worse_p_side = max(subs, key=lambda k: ev[k]["matched_random_p"])
    worse_stat_side = min(subs, key=lambda k: ev[k]["calibration_slope"])
    pval = ev[worse_p_side]["matched_random_p"]
    stat = ev[worse_stat_side]["calibration_slope"]
    # A TIE IS NOT A CONCURRENCE, and on this population it is the normal
    # case: both sub-heads sit at the 1/(n_draws+1) permutation floor, so
    # "the WORSE of the two" has no worse to name and `max` returns whichever
    # side argument order puts first. Reporting that as AGREE would be a label
    # that cannot distinguish two different situations -- rule 16's shape in
    # a receipt field. The p VALUE is unaffected (both sides carry it), and
    # saying so is the honest form of the ruling rather than a weaker one.
    tied = ev[subs[0]]["matched_random_p"] == ev[subs[1]]["matched_random_p"]
    floors = {k: (1.0 / (nd + 1) if (nd := (heads.get(k, {}).get(
        "matched_random") or {}).get("n_draws")) else None) for k in subs}
    at_floor = sorted(k for k in subs
                      if floors[k] is not None
                      and ev[k]["matched_random_p"] == floors[k])
    if tied:
        reading = (f"TIED ON p: both sides carry {pval!r}, so R-306's "
                   f"worse-of-the-two does not discriminate here and the "
                   f"adjudicated p is the same whichever side is named. The "
                   f"side reported is argument order's and nothing turns on "
                   f"it. This is NOT a concurrence")
    elif worse_p_side == worse_stat_side:
        reading = "AGREE"
    else:
        reading = "DISAGREE (disclosed, not resolved by preference)"
    if at_floor:
        reading += (f". At the PERMUTATION FLOOR on {at_floor}: p equals "
                    f"1/(n_draws+1) exactly, so the null is resolution-"
                    f"limited and more permutations are needed before this "
                    f"cell's p can separate anything")
    # F1: the cell's n is the BINDING side's -- the side whose slope the cell
    # reports. Both sides' n travel in the detail so the pair stays visible.
    _mn = _head_metric_n(r[arm], worse_stat_side)
    _n_stat = _mn["metric_n"]
    _n_both = {k: _head_metric_n(r[arm], k)["metric_n"] for k in subs}
    _au = {k: _head_metric_n(r[arm], k) for k in subs}
    # R-397 RULING 2: Q3 adjudicates on ITS OWN gate. Its frozen gate is
    # "calibration slope CI excludes 0 for each, reported separately" and
    # carries NO incumbent term, so NO_INCUMBENT_COUNTERPART -- a status
    # meaning "the incumbent cannot answer this head" -- was blocking a head
    # that never asked the incumbent anything. Both slope conjuncts are
    # evaluated (R-306), so the cell is OK.
    return (stat, pval, I11.CELL_STATUS_OK, _n_stat,
            f"Q3 binding side {worse_stat_side} — the ROW count the "
            f"calibration slope was computed on (RR4-3); both sides "
            f"{_n_both}. ACTION-UNIT slope beside it: "
            f"{ {k: v['action_unit_value'] for k, v in _au.items()} } over "
            f"{ {k: v['action_n'] for k, v in _au.items()} } actions "
            f"({ {k: v['rows_per_action'] for k, v in _au.items()} } "
            f"rows/action)",
            f"n behind this statistic: {_n_stat!r} actions on "
            f"{worse_stat_side} (both sides {_n_both!r}). "
            f"R-306 (USER, 2026-08-29, frozen A1.4): Q3's two slope gates "
            f"compose as CONJUNCTION + WORSE SIDE. Both sides are evaluable, "
            f"so the cell is adjudicated: statistic = min slope "
            f"{stat!r} from {worse_stat_side}, p = the WORSE of the two "
            f"{pval!r} from {worse_p_side}; the two worse-side readings "
            f"{reading}"
            f". Adjudicating the worse p at a common alpha IS the conjunction "
            f"(intersection-union), so the cell passes only if BOTH sides do. "
            f"Per-side evidence on {coin}: "
            f"Q3_m_harm slope {sh['calibration_slope']!r} p "
            f"{sh['matched_random_p']!r}; Q3_m_good slope "
            f"{sg['calibration_slope']!r} p {sg['matched_random_p']!r}. "
            f"The gate's 'CI excludes 0' is tested in its R-286 implemented "
            f"form -- matched-random null at no_skill_value=0.0, one-sided "
            f"'greater', VERIFIED on both sides -- and NO literal interval is "
            f"claimed: rule 8 forbids one at G=0 complete UTC days. Status is "
            f"OK and adjudicated on ITS OWN declared gate (R-397 ruling 2, "
            f"USER): this head's frozen gate carries NO incumbent term, so "
            f"NO_INCUMBENT_COUNTERPART -- which means 'the incumbent cannot "
            f"answer this head' -- was blocking a head that never asked it "
            f"anything. The p is the MATCHED-RANDOM null, which IS this "
            f"gate's declared null (R-286); no incremental null is owed.")


def _q4_cell(arm: str, budget: str, per_coin: dict) -> tuple:
    """Q4's cell: net cents at the action unit, with the incremental null.

    The incremental-over-incumbent null is the one iteration 010 lacked, and it
    is the reason 'beats random' was mistaken for 'beats the incumbent' for a
    full cycle."""
    inc_by_window = {}
    net = 0.0
    inc_net = 0.0
    incumbent_net = 0.0
    _n_ranked = 0
    for coin, r in per_coin.items():
        econ = r[arm].get("economics", {}).get(budget)
        if not econ:
            return (None, None, I11.CELL_STATUS_UNEVALUABLE, None, "",
                    f"no economics for {coin}@{budget}", None)
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
                    econ.get("n_actions_total"),
                    "ranked generations (ACTION unit, first-crossing dedup, "
                    "A1.5); no increment exists so no null unit count does",
                    f"{coin}@{budget} economics were computed with no incumbent "
                    f"counterpart, so no candidate-minus-incumbent statistic "
                    f"exists on the identical action population (prereg 5.2). "
                    f"Net {net:+.1f}c is the CANDIDATE'S OWN value and is not "
                    f"an increment.", None)
        incumbent_net += econ.get("incumbent_net_cents") or 0.0
        _n_ranked += econ.get("n_actions_total") or 0
        for w, v in econ["increment_by_window"].items():
            inc_net += v
            inc_by_window[f"{coin}/{w}"] = inc_by_window.get(f"{coin}/{w}", 0.0) + v
    if not inc_by_window:
        return (net, None, I11.CELL_STATUS_UNEVALUABLE, None, "",
                "no per-window increments to permute", None)
    null = I11.sign_flip_null(inc_by_window)
    # I11-B5: the STATISTIC is the INCREMENT, because the p describes the
    # increment. The cell previously returned `net` -- the candidate's own
    # realised value -- beside a p computed by sign-flipping per-window
    # INCREMENTS, so Holm would have ranked the increment's evidence against
    # the raw net's magnitude. A p that does not describe the number beside it
    # is not evidence about that number. The raw nets are REPORTED, never
    # adjudicated (A1.4: every other metric is reported and never adjudicated).
    # ADJUDICATE the one-sided p. Q4's gate says the candidate must BEAT the
    # incumbent, and the two-sided form gave a candidate LOSING by 120c the
    # same p as one winning by 120c. p_two_sided remains in the null's output
    # as a reported diagnostic; it is not what decides a cell.
    # F1: Q4's adjudicated statistic is a SUM OVER WINDOWS and its p is a
    # sign-flip over those same windows, so the n behind the number is the
    # WINDOW count, not the action count -- naming the action count here
    # would overstate the unit the interval-free claim rests on (rule 8).
    # F-2: R-288 promised p_two_sided "stays as a REPORTED diagnostic so
    # nothing citing it breaks" and it occurred ZERO times in the artifact,
    # so the FROZEN form's p could not be recovered from the emission at all.
    # The frozen prereg 5(2) says two-sided; the adjudicated p is one-sided
    # (R-286/R-288) and only the USER amends a frozen design (rule 4), so the
    # frozen form travels beside the adjudicated one until A2 is ruled.
    _two = null.get("p_two_sided")
    return (inc_net, null["p_value"], I11.CELL_STATUS_OK, null["n_units"],
            f"windows carrying a candidate-minus-incumbent increment "
            f"(sign-flip units, R-234 sorted order); the action population "
            f"ranked at this budget is {_n_ranked!r}",
            f"increment vs incumbent {null['observed']:+.1f}c over "
            f"{null['n_units']} windows; {null['n_perm']} sign-flip "
            f"permutations, units consumed in SORTED order (R-234). "
            f"REPORTED not adjudicated: candidate_net_cents {net:+.1f}, "
            f"incumbent_net_cents {incumbent_net:+.1f}; the adjudicated "
            f"statistic is the INCREMENT {inc_net:+.1f}c, which is what this "
            f"p describes. FROZEN-FORM DIAGNOSTIC (F-2): prereg 5(2) declares "
            f"a TWO-SIDED p and the adjudicated p here is ONE-SIDED per "
            f"R-286/R-288 (a two-sided test scores |sum|, so a candidate "
            f"LOSING by 120c earns the p of one WINNING by 120c). The frozen "
            f"form is p_two_sided={_two!r}; it is REPORTED and never "
            f"adjudicated, and amendment A2 is DRAFT-FOR-USER-FREEZE because "
            f"only the USER amends a frozen design (rule 4).",
            _two)


def _selftest_verdict(fails: list) -> int:
    """Print the verdict and return the exit code, together and last."""
    print(f"\n{'ITER011 RUN SELFTEST GREEN' if not fails else 'RED'}: "
          f"{len(fails)} failing")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


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
    # R-306: a Q3 sub-head carries a SLOPE and a null declared against
    # no_skill_value 0.0 one-sided 'greater' (R-286's implemented form of
    # "CI excludes 0"). The fixture must carry what a real head carries --
    # a fixture missing them made the Q3 cells look merely undeclared.
    _mr3 = lambda pv: dict(_mr(pv), no_skill_value=0.0, alternative="greater")
    _fake = {c: {a: {"adjudicated_statistics": {
                        "Q1_arrival": 0.7, "Q2_sign": 0.6,
                        "Q2_cell_status": "OK", "Q3_magnitudes": 1.0},
                     "heads": {"Q1_arrival": {"status": "OK", "auc": 0.7,
                                              "matched_random": _mr(0.004)},
                               "Q2_p_pos": {"auc": 0.61, "matched_random": _mr(0.02)},
                               "Q2_p_neg": {"auc": 0.60, "matched_random": _mr(0.31)},
                               "Q3_m_harm": {"status": "OK",
                                             "calibration_slope": 0.70,
                                             "matched_random": _mr3(0.05)},
                               "Q3_m_good": {"status": "OK",
                                             "calibration_slope": 0.91,
                                             "matched_random": _mr3(0.44)}},
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
    # R-306 INVERTS BOTH OF THESE, and they are kept as the record of what
    # changed: the AGGREGATION_UNDECLARED refusal was correct until the ruling
    # existed, and wrong after it. A gap a ruling has closed is no longer a gap.
    # R-397 RULING 2 SUPERSEDES the status half of this control, and it is
    # kept as the record: Q3 adjudicates (R-306) AND is now OK, because its
    # frozen gate carries no incumbent term and NO_INCUMBENT_COUNTERPART was
    # blocking a head that never asked the incumbent anything.
    ok(all(_fam["cells"][k]["status"] == I11.CELL_STATUS_OK
           for k in _fam["cells"] if "/Q3_magnitudes/" in k),
       "R-397(2) Q3 cells are OK and adjudicated on their OWN declared gate; "
       "the no-counterpart status stays on Q2, whose gate DOES name an "
       "incumbent the incumbent cannot supply")
    ok(all(_fam["cells"][k]["p_value"] == 0.44
           for k in _fam["cells"] if "/Q3_magnitudes/" in k),
       "R-306 Q3's adjudicated p is the WORSE of the two (0.44, m_good), NOT "
       "the first sub-head's 0.05 -- the old `_cell_p` took m_harm by dict "
       "order, which is right only when m_harm happens to be the weaker side")
    ok(all(_fam["cells"][k]["statistic"] == 0.70
           for k in _fam["cells"] if "/Q3_magnitudes/" in k),
       "R-306 Q3's statistic is the min SLOPE (0.70) -- the binding side "
       "under a gate that reads distance from 0")

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



    # ---- R-280: the Q4 incumbent LOAD-VERIFY-APPLY path -------------------
    # Rule 15 on a path that has not run yet: it must be shown able to REFUSE
    # before it is ever trusted to accept. Executes post-HOLD-RELEASE; these
    # run now so the refusals are proven, not promised.
    import tempfile as _tf, shutil as _sh
    _fd = Path(_tf.mkdtemp())
    _real = PA.FITDIR / "linear_d_btc.json"
    _have_real = _real.exists() and (PA.FITDIR / PA.FIT_MANIFEST).exists()
    ok(_have_real,
       "R-280 the committed incumbent artifact and its manifest are present "
       "(if this fails the checks below prove nothing)")
    if _have_real:
        _m = json.loads((PA.FITDIR / PA.FIT_MANIFEST).read_text())
        _mod = load_verified_incumbent("btc")
        ok(_mod["arm"] == INCUMBENT_ARM and
           _mod["_verified"]["sha256_prefix"] == _m["file_hashes"]["linear_d_btc.json"],
           "POSITIVE CONTROL: the COMMITTED incumbent loads and its hash equals "
           "the one the fit manifest recorded")
        ok("not_verified" in _mod["_verified"],
           "R-280 the loader states what it did NOT verify; checking one "
           "artifact's identity is not checking the fit chain, and a loader "
           "that blurs those invites the second to be assumed from the first")

        _sh.copy(_real, _fd / "linear_d_btc.json")
        (_fd / PA.FIT_MANIFEST).write_text(json.dumps(_m))
        # KNOWN-BAD 1: content changed, manifest unchanged.
        _t = json.loads(_real.read_text())
        _t["hazard_weights"] = [w + 1e-9 for w in _t["hazard_weights"]]
        (_fd / "linear_d_btc.json").write_text(json.dumps(_t))
        try:
            load_verified_incumbent("btc", fitdir=_fd)
            _r1 = ""
        except RuntimeError as e:
            _r1 = str(e)
        ok("not the committed incumbent" in _r1,
           f"KNOWN-BAD: a TAMPERED incumbent is refused by hash — a 1e-9 weight "
           f"nudge is invisible in every number it produces (got {_r1[:60]!r})")
        # KNOWN-BAD 2: right bytes, wrong arm.
        _sh.copy(_real, _fd / "linear_d_btc.json")
        _t2 = json.loads(_real.read_text()); _t2["arm"] = "LGBM_PINNED"
        (_fd / "linear_d_btc.json").write_text(json.dumps(_t2))
        (_fd / PA.FIT_MANIFEST).write_text(json.dumps(
            {**_m, "file_hashes": {**_m["file_hashes"],
                                   "linear_d_btc.json": _sha16(_fd / "linear_d_btc.json")}}))
        try:
            load_verified_incumbent("btc", fitdir=_fd)
            _r2 = ""
        except RuntimeError as e:
            _r2 = str(e)
        ok("names ONE arm" in _r2,
           f"KNOWN-BAD: an artifact that hashes correctly but identifies as a "
           f"DIFFERENT ARM is refused (got {_r2[:60]!r})")
        # KNOWN-BAD 3: the manifest does not bind it at all.
        _sh.copy(_real, _fd / "linear_d_btc.json")
        (_fd / PA.FIT_MANIFEST).write_text(json.dumps({**_m, "file_hashes": {}}))
        try:
            load_verified_incumbent("btc", fitdir=_fd)
            _r3 = ""
        except RuntimeError as e:
            _r3 = str(e)
        ok("does not bind" in _r3,
           f"KNOWN-BAD: an UNBOUND artifact is refused; an absent binding must "
           f"never read as a matching one (got {_r3[:60]!r})")
        # KNOWN-BAD 4: missing entirely.
        try:
            load_verified_incumbent("nosuchcoin", fitdir=_fd)
            _r4 = ""
        except RuntimeError as e:
            _r4 = str(e)
        ok("no incumbent artifact" in _r4,
           "KNOWN-BAD: a missing incumbent REFUSES rather than falling back to "
           "fitting one (R-280: a re-fit incumbent is a different incumbent)")

        # APPLY: arm D is PM+fine with NO STATE FEATURES. That is its defining
        # property and the reason (PLUS_PRED_STATE_V1 - INCUMBENT) isolates
        # state, so it is asserted behaviourally: changing ST must not move a
        # single value.
        _n = len(_mod["norm_mu"])
        _blk = {"PM": [[0.10] * (_n - 3) for _ in range(4)],
                "FN": [[0.20] * 3 for _ in range(4)],
                "ST": [[0.30] * 5 for _ in range(4)]}
        _a1 = apply_incumbent(_mod, _blk, range(4))
        _blk2 = dict(_blk, ST=[[99.0] * 5 for _ in range(4)])
        _a2 = apply_incumbent(_mod, _blk2, range(4))
        ok(_a1["expected_cancel_value"] == _a2["expected_cancel_value"],
           "R-280 apply_incumbent ignores STATE features entirely: arm D is "
           "PM+fine only, and that is exactly what makes "
           "(PLUS_PRED_STATE_V1 - INCUMBENT) isolate state")
        ok(len(_a1["expected_cancel_value"]) == 4 and
           all(isinstance(v, float) for v in _a1["expected_cancel_value"]),
           "R-280 apply_incumbent returns one composed value per requested row")
        try:
            apply_incumbent(_mod, {"PM": [[1.0]], "FN": [[1.0]],
                                   "ST": [[0.0]]}, range(1))
            _r5 = ""
        except RuntimeError as e:
            _r5 = str(e)
        ok("differently-shaped vector" in _r5,
           f"KNOWN-BAD: a WIDTH MISMATCH refuses rather than scoring a "
           f"truncated row (got {_r5[:60]!r})")
        # The increment is only defined on the IDENTICAL population, so the
        # composed values must align row-for-row with the candidate's.
        _cand = {"expected_cancel_value": [0.5] * 4}
        ok(len(_a1["expected_cancel_value"]) == len(_cand["expected_cancel_value"]),
           "R-280 the incumbent scores the SAME rows as the candidate, which is "
           "what makes prereg §5.2's 'identical action population' true rather "
           "than merely claimed")
    _sh.rmtree(_fd, ignore_errors=True)

    # THE VERDICT IS PRINTED BY THE RETURN, not by a line above it. Twice now a
    # block appended before `return` has landed BELOW the summary, so the suite
    # printed GREEN with failures listed underneath it -- rule 10 in the
    # instrument itself. Binding the print to the return makes that
    # unrepresentable: anything inserted before the return runs before the
    # verdict, because the verdict IS the return.

    # ============ Codex round-3 findings (red-first) ======================
    # (1) RULE 17 INSIDE BATCH-4, and it is mine. A1.5 freezes "Head fitting
    # weights each row by 1 / rows_in_generation". generation_weights was
    # WRITTEN and its only caller was a unit test; the fits passed [1.0]*len.
    # So n was relabelled ACTION while the estimator stayed row-weighted --
    # exactly the suite-green-is-not-pipeline-wired class I have been filing
    # against others.
    _wr = [_gr("W", "BUY", 1, 0.0, 5.0), _gr("W", "BUY", 1, 1.0, 5.0),
           _gr("W", "BUY", 2, 0.0, -4.0)]
    _tw = head_targets(_wr)
    ok(_tw.get("w_rows") == [0.5, 0.5, 1.0],
       f"B3(b) head_targets carries A1.5 fitting weights 1/rows_in_generation "
       f"(2-row gen -> 0.5 each, 1-row gen -> 1.0); got {_tw.get('w_rows')!r}")
    _fsrc = inspect.getsource(fit_arm)
    ok("[1.0] * len" not in _fsrc and "[1.0]*len" not in _fsrc,
       "B3(b) fit_arm no longer passes UNIT weights; A1.5's weighting reaches "
       "the estimator, not just the report")
    ok("w_rows" in _fsrc,
       "B3(b) fit_arm consumes the declared weights")
    ok(_tw["counts"]["n_actions"] == 2 and _tw["counts"]["n_rows"] == 3,
       "the action count and the row count are BOTH stated, never conflated")

    # (6) The stratum hour was WINDOW-RELATIVE. t_start is an offset within a
    # 5-minute window, so t_start//3600 is 0 for essentially every row and
    # side x hour matching collapsed every real UTC hour into one bucket. The
    # governing instant is t0 + t_start.
    _h = [{"slug": "s", "side": "BUY_UP", "gen": 1, "t_start": 0.0, "t0": 0},
          {"slug": "s", "side": "BUY_UP", "gen": 2, "t_start": 0.0, "t0": 3600}]
    ok(_strata(_h, [0, 1]) == [("BUY_UP", 0), ("BUY_UP", 1)],
       f"B6 the stratum hour is the GOVERNING UTC hour (t0 + t_start), not the "
       f"window-relative one; two rows an hour apart must not share a stratum "
       f"(got {_strata(_h, [0, 1])})")

    # (5) The Q4 cell returned `net` as its statistic while its p came from a
    # sign-flip over INCREMENTS. A cell whose p describes a different quantity
    # than its statistic cannot be read: Holm would rank the increment's
    # evidence against the raw net's magnitude.
    _q4rows = [_gr("W", "BUY", g, 0.0, 10.0 if g % 2 else -6.0) for g in range(1, 9)]
    _q4p = {"expected_cancel_value": [0.9 - 0.1 * i for i in range(8)]}
    _q4i = {"expected_cancel_value": [0.1 + 0.1 * i for i in range(8)]}
    _pc = {"btc": {"composed_linear": {
        "economics": q4_economics(_q4p, _q4rows, incumbent=_q4i),
        "heads": {}, "adjudicated_statistics": {}}}}
    _st, _pv, _stat, _sn4, _snb4, _det, _p2a = _q4_cell(
        "composed_linear", I11.BUDGETS_011[0], _pc)
    _ec0 = _pc["btc"]["composed_linear"]["economics"][I11.BUDGETS_011[0]]
    ok(_st == sum(_ec0["increment_by_window"].values()),
       f"B5 the Q4 cell's STATISTIC is the INCREMENT its p describes, not the "
       f"raw candidate net ({_st!r} vs increment "
       f"{sum(_ec0['increment_by_window'].values())!r}, candidate net "
       f"{_ec0['net_cents']!r})")
    ok("candidate_net_cents" in _det and "incumbent_net_cents" in _det,
       "B5 the raw candidate and incumbent nets are REPORTED beside the "
       "adjudicated increment, never adjudicated in its place")

    # (4) The receipt guard accepted substantive garbage: correct keys, but
    # arm/head/budget disagreeing with the cell's own identity, a NaN
    # statistic, p=0 and a negative action count.
    _bad = {}
    for k in I11.declared_family()["cells"]:
        a, h, b = k.split("/")
        _bad[k] = {"cell": k, "arm": "WRONG", "head": "WRONG", "budget": "WRONG",
                   "statistic": float("nan"), "p_value": 0.0, "status": "OK",
                   "n_actions": -999, "detail": "x",
                   "adjudicated_statistic_name": None}
    try:
        assert_receipt_has_all_cells({"family": {
            "cells": _bad, "holm_denominator": len(_bad)}})
        _g4 = ""
    except RuntimeError as e:
        _g4 = str(e)
    ok(_g4 != "",
       "B4 a family whose cells carry the right KEYS but wrong arm/head/budget, "
       "a NaN statistic, p=0 and n_actions=-999 is REFUSED; matching a "
       "declaration is not carrying a result")
    for token in ("arm", "n_actions", "p_value"):
        ok(token in _g4,
           f"B4 the refusal NAMES {token} rather than failing vaguely")

    # (2) Q1's frozen gate is "beats the matched-random null AND beats the
    # incumbent hazard head". The incumbent HAS a hazard head
    # (INCUMBENT_COMPARABLE["Q1_arrival"] is True), so unlike Q2/Q3 there is a
    # counterpart and R-237 does not excuse its absence.
    ok(INCUMBENT_COMPARABLE["Q1_arrival"] is True,
       "Q1 HAS an incumbent counterpart, so its incremental leg is required")
    ok(callable(globals().get("apply_incumbent_hazard")),
       "B2 a Q1 incumbent HAZARD comparator exists (the R-280 load-verify-"
       "apply path extends to the hazard head, not only to composed value)")


    # ---- the Q4 cell adjudicates the DIRECTIONAL p (sibling of (3)) -------
    _lose = [_gr("W", "BUY", g, 0.0, -50.0) for g in range(1, 9)]
    _lp = {"expected_cancel_value": [0.9 - 0.1 * i for i in range(8)]}
    _li = {"expected_cancel_value": [0.1 + 0.1 * i for i in range(8)]}
    _lpc = {"btc": {"composed_linear": {
        "economics": q4_economics(_lp, _lose, incumbent=_li),
        "heads": {}, "adjudicated_statistics": {}}}}
    _ls, _lpv, _lst, _lsn, _lsnb, _ld, _lp2 = _q4_cell(
        "composed_linear", I11.BUDGETS_011[0], _lpc)
    _inc = sum(_lpc["btc"]["composed_linear"]["economics"]
               [I11.BUDGETS_011[0]]["increment_by_window"].values())
    if _inc < 0:
        ok(_lpv is not None and _lpv > 0.5,
           f"a Q4 cell whose increment is NEGATIVE ({_inc:+.1f}c: the candidate "
           f"LOST to the incumbent) must not carry small-p evidence; got "
           f"p={_lpv}")
    ok(_lpv is None or 0.0 < _lpv <= 1.0,
       "the Q4 cell's p is a real permutation p in (0, 1]")

    _selftest_coin_slice(ok)
    _selftest_q3_r306(ok)
    _selftest_incumbent_wiring(ok)
    _selftest_survivor_and_provenance(ok)
    _selftest_artifact_wiring_guard(ok)
    _selftest_review_batch_f2_f7(ok)
    _selftest_gate_evaluation_rr2(ok)
    _selftest_frozen_prereg_anchor(ok)
    _selftest_r397_rulings(ok)
    _selftest_action_unit_rr4(ok)

    return _selftest_verdict(fails)


def _selftest_action_unit_rr4(ok):
    """RR4-3: the same statistic, deduplicated to actions. Red-first."""
    # Two generations. The FIRST has three rows and the SECOND one, so a
    # row-level metric lets generation A count three times. That is the
    # inflation rule 2 names, constructed so it MUST show.
    rws = ([{"slug": "w", "side": "BUY_UP", "gen": 0, "t_start": float(i)}
            for i in range(3)]
           + [{"slug": "w", "side": "BUY_UP", "gen": 1, "t_start": 0.0}])
    ok(action_unit_metrics([0.9, 0.9, 0.9, 0.1], [1, 1, 1, 0], rws,
                           "probability")["n_actions"] == 2,
       "RR4-3 the collapse counts ACTIONS, not rows (2 generations from 4 rows)")
    _m = action_unit_metrics([0.9, 0.9, 0.9, 0.1], [1, 1, 1, 0], rws,
                             "probability")
    ok(_m["n_rows"] == 4 and _m["rows_per_action"] == 2.0,
       f"RR4-3 both populations travel with their ratio "
       f"({_m['n_rows']} rows / {_m['n_actions']} actions = "
       f"{_m['rows_per_action']})")

    # THE INFLATION IS REAL AND MEASURABLE. Row-level AUC is dominated by the
    # generation with more rows; the action unit gives each one vote.
    # Constructed so the two units MUST disagree: generation A (3 rows) is
    # mixed and generation B (1 row) is negative. At the row unit A's three
    # rows each vote; at the action unit A votes once. Row AUC 0.75, action
    # AUC 1.0 — my first fixture gave 0.0 at BOTH units and proved nothing.
    _p = [0.9, 0.8, 0.7, 0.85]
    _y = [1, 1, 0, 0]
    _row_auc = I11.auc(_p, _y)
    _act = action_unit_metrics(_p, _y, rws, "probability")
    ok(_row_auc != _act["value"] and _row_auc == 0.75
       and _act["value"] == 1.0,
       f"RR4-3 KNOWN-BAD: a generation with THREE rows and one with ONE give "
       f"different answers at the two units (row {_row_auc}, action "
       f"{_act['value']}) — which is the inflation rule 2 names")

    # ...and the control must ADMIT the case where they agree, or it is a
    # check that fires on everything.
    _even = [{"slug": "w", "side": "BUY_UP", "gen": g, "t_start": 0.0}
             for g in range(4)]
    _pe, _ye = [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]
    ok(action_unit_metrics(_pe, _ye, _even, "probability")["value"]
       == I11.auc(_pe, _ye),
       "RR4-3 POSITIVE CONTROL: with ONE row per action the two units agree "
       "exactly — the collapse changes nothing when there is nothing to "
       "collapse")

    # ALL THREE COLLAPSE RULES ARE REPORTED, and they can differ.
    _mix = [{"slug": "w", "side": "BUY_UP", "gen": 0, "t_start": 0.0},
            {"slug": "w", "side": "BUY_UP", "gen": 0, "t_start": 1.0},
            {"slug": "w", "side": "BUY_UP", "gen": 1, "t_start": 0.0}]
    _r3 = action_unit_metrics([0.1, 0.9, 0.5], [1, 1, 0], _mix,
                              "probability")["by_collapse_rule"]
    ok(set(_r3) == {"max", "mean", "first"},
       f"RR4-3 all three collapse rules are reported, so a reader sees "
       f"whether the answer depends on the choice (got {_r3})")

    # DISAGREEING ROW LABELS ARE COUNTED — if large, the shared-outcome
    # premise is itself wrong, and that is a finding not a detail.
    _d = action_unit_metrics([0.1, 0.9, 0.5], [1, 0, 0], _mix, "probability")
    ok(_d["generations_with_disagreeing_row_labels"] == 1,
       "RR4-3 generations whose rows DISAGREE on the outcome are counted; a "
       "large count would refute the shared-outcome premise the collapse "
       "rests on")
    ok(action_unit_metrics([0.1, 0.9, 0.5], [1, 1, 0], _mix,
                           "probability")[
           "generations_with_disagreeing_row_labels"] == 0,
       "RR4-3 and it reads zero when they agree — the counter discriminates")

    # magnitude heads collapse too, and refuse a misaligned call
    ok(action_unit_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 9.0], _mix,
                           "magnitude")["value"] is not None,
       "RR4-3 magnitude heads collapse as well, so Q3's surviving slope has "
       "an action-unit twin")
    try:
        action_unit_metrics([0.1], [1, 0], _mix, "probability")
        ok(False, "RR4-3 a misaligned collapse must REFUSE")
    except RuntimeError as e:
        ok("aligned vectors" in str(e),
           "RR4-3 KNOWN-BAD: misaligned vectors REFUSE rather than zipping to "
           "the shorter one")

    # M79: the CELL must quote the metric's OWN population. The action count
    # is the twin's n, not this statistic's, and reporting it here is F1's
    # defect one layer down — the thing RR4-3 is about.
    _ra = {"heads": {"Q1_arrival": {
        "n_rows": 311640, "n_actions": 177674,
        "action_unit": {"value": 0.8, "rows_per_action": 1.754}}}}
    _mn = _head_metric_n(_ra, "Q1_arrival")
    ok(_mn["metric_n"] == 311640 and _mn["metric_unit"] == "rows"
       and _mn["action_n"] == 177674,
       f"RR4-3 the cell quotes the ROW count the metric was computed on with "
       f"its unit, and carries the action count beside it (got "
       f"{_mn['metric_n']} {_mn['metric_unit']}, action {_mn['action_n']})")
    _mn2 = _head_metric_n({"heads": {"H": {"n_actions": 5}}}, "H")
    ok(_mn2["metric_n"] == 5 and _mn2["metric_unit"] == "actions",
       "RR4-3 and a head with no row count falls back to actions and SAYS so "
       "— the unit is never assumed")

    # ---- RR4-1: the conjunct-derived status, driven ------------------
    def cellset(q4_null=True, q2_status=I11.CELL_STATUS_NO_COUNTERPART):
        cs = {}
        for a in I11.ARMS_011:
            for h in I11.HEADS_011:
                for b in I11.BUDGETS_011:
                    cs[I11.cell_key(a, h, b)] = I11.build_cell(
                        a, h, b, statistic=1.0, p_value=0.001,
                        status=(q2_status if h == "Q2_sign"
                                else I11.CELL_STATUS_OK), n_actions=10)
        r = {"family": I11.assemble_family(cs),
             "incumbent_null_applicability": {"comparable": dict(
                 INCUMBENT_COMPARABLE)},
             "results": {"btc": {a: {"q1_incremental": {
                 "beats_incumbent_hazard_head": True, "incumbent_auc": 0.7,
                 "incumbent_provenance": {"sha256_prefix": "x"}}}
                 for a in I11.ARMS_011}}}
        r["incumbent_legs_evaluated"] = {
            h: {"incumbent_counterpart_computed": True}
            for h, v in INCUMBENT_COMPARABLE.items() if v}
        if not q4_null:
            for c in r["family"]["cells"].values():
                if c["head"] == "Q4_combined_ev":
                    c["declared_gate_outcome"] = {"conjuncts": {
                        "increment_beats_incumbent": True,
                        "matched_random": True}}
        return r

    _cs = cellset()
    attach_declared_gate_outcomes(_cs)
    _g = apply_gate_evaluation_status(_cs)
    _q4 = [k for k, c in _cs["family"]["cells"].items()
           if c["head"] == "Q4_combined_ev"]
    ok(all(_cs["family"]["cells"][k]["status"]
           == I11.CELL_STATUS_GATE_PARTIAL for k in _q4)
       and all(_cs["family"]["cells"][k]["gate_conjuncts_evaluated"] is False
               for k in _q4),
       "RR4-1 KNOWN-BAD: an OK cell carrying a NULL conjunct (Q4's uncomputed "
       "matched-random) is re-statused GATE_PARTIALLY_EVALUATED — twelve "
       "cells previously asserted gate_conjuncts_evaluated true while "
       "carrying one")
    ok(all("conjuncts never evaluated" in
           _cs["family"]["cells"][k]["gate_partial_reason"] for k in _q4),
       "RR4-1 and the reason names the GENERIC cause, distinct from the "
       "incumbent-leg one")
    _q2 = [k for k, c in _cs["family"]["cells"].items()
           if c["head"] == "Q2_sign"]
    ok(all(_cs["family"]["cells"][k]["status"]
           == I11.CELL_STATUS_NO_COUNTERPART for k in _q2),
       "RR4-1 KNOWN-BAD (the other way): a cell already carrying a MORE "
       "SPECIFIC status keeps it — NO_INCUMBENT_COUNTERPART says WHY the "
       "conjunct is unanswerable and the generic status would lose that")
    _q1 = [k for k, c in _cs["family"]["cells"].items()
           if c["head"] in ("Q1_arrival", "Q3_magnitudes")]
    ok(all(_cs["family"]["cells"][k]["status"] == I11.CELL_STATUS_OK
           and _cs["family"]["cells"][k]["gate_conjuncts_evaluated"] is True
           for k in _q1),
       "RR4-1 POSITIVE CONTROL: Q1 and Q3, whose conjuncts are all evaluated, "
       "stay OK — the generalisation narrows nothing that was whole")
    _cs2 = cellset(q4_null=False)
    _g2 = apply_gate_evaluation_status(_cs2)
    ok(not any(_cs2["family"]["cells"][k]["head"] == "Q4_combined_ev"
               for k in _g2["cells_gate_partially_evaluated"]),
       "RR4-1 and with Q4's conjuncts BOTH evaluated it is not re-statused — "
       "the rule follows the conjuncts, not the head")

    # ---- RR4-2: the per-cell numbers must be COMPUTED -----------------
    _fl = {"family": {"cells": {}, "holm_denominator": 24},
           "results": {"btc": {"composed_linear": {"heads": {
               "Q1_arrival": {"matched_random": {"n_draws": 500}}}}}}}
    for i in range(24):
        k = I11.cell_key("composed_linear", "Q1_arrival", f"{i}%")
        c = I11.build_cell("composed_linear", "Q1_arrival", f"{i}%",
                           statistic=0.8, p_value=1 / 501, n_actions=10)
        c["holm_p"] = 24 / 501
        _fl["family"]["cells"][k] = c
    attach_floor_disclosure(_fl)
    _d0 = list(_fl["family"]["cells"].values())[0]["permutation_floor"]
    ok(_d0["holm_if_ONE_draw_beat_it_in_THIS_CELL_ONLY"]
       < _d0["holm_if_ONE_draw_beat_it_in_EVERY_at_floor_CELL"],
       f"RR4-2 both one-draw numbers are attached per cell and the "
       f"single-cell one is MILDER "
       f"({_d0['holm_if_ONE_draw_beat_it_in_THIS_CELL_ONLY']:.4f} vs "
       f"{_d0['holm_if_ONE_draw_beat_it_in_EVERY_at_floor_CELL']:.4f})")
    ok(_d0["survives_if_THIS_CELL_ONLY_moved"] is True
       and _d0["survives_if_EVERY_at_floor_cell_moved"] is False,
       "RR4-2 and they answer DIFFERENTLY at the bar, which is why the "
       "multiplied figure alone overstated the single-cell consequence")


def _selftest_r397_rulings(ok):
    """R-397 rulings 1 and 2, red-first in both directions."""
    L = str(D.TARGET_LATENCY_MS)

    def rows(n=40):
        return [{"slug": f"w{i // 8}", "side": "BUY_UP", "gen": i,
                 "t0": 1787650200.0 + (i // 8) * 300, "t_start": float(i % 8),
                 "coin": "btc"} for i in range(n)]

    # ---- RULING 1: the second conjunct is a COMPARISON, and it discriminates
    _r, _y = rows(), [i % 2 for i in range(40)]
    _better = [0.9 if y else 0.1 for y in _y]          # perfect
    _worse = [0.1 if y else 0.9 for y in _y]           # inverted

    def inc(p):
        """A well-formed incumbent hazard block, as apply_incumbent_hazard
        returns it — arm, head and VERIFIED provenance, not bare numbers."""
        return {"p_fill": list(p), "arm": INCUMBENT_ARM, "head": "Q1_arrival",
                "n": len(p), "provenance": {"sha256_prefix": "deadbeef"}}

    _win = q1_incremental(_better, inc(_worse), _r, _y)
    ok(_win["beats_incumbent_hazard_head"] is True
       and _win["candidate_auc"] == 1.0 and _win["incumbent_auc"] == 0.0,
       f"R-397(1) POSITIVE CONTROL: a candidate that outranks the incumbent "
       f"BEATS its hazard head (auc {_win['candidate_auc']} vs "
       f"{_win['incumbent_auc']})")
    _lose = q1_incremental(_worse, inc(_better), _r, _y)
    ok(_lose["beats_incumbent_hazard_head"] is False,
       "R-397(1) KNOWN-BAD: the SAME code returns False when the candidate "
       "loses — the conjunct discriminates rather than always passing, which "
       "is what makes 'pass or fail, either is the result of record' true")
    _tie = q1_incremental(_better, inc(_better), _r, _y)
    ok(_tie["beats_incumbent_hazard_head"] is False
       and _tie["increment_auc"] == 0.0,
       "R-397(1) a TIE does not beat: 'beats' is strict, so an identical "
       "hazard head fails the conjunct rather than passing it")

    # the identical-population requirement is CHECKED, not assumed
    try:
        q1_incremental(_better, inc(_worse[:-1]), _r, _y)
        ok(False, "R-397(1) a misaligned incumbent must REFUSE")
    except RuntimeError as e:
        ok("IDENTICAL action population" in str(e),
           "R-397(1) KNOWN-BAD: a mismatched incumbent vector REFUSES — "
           "prereg 5.2's increment is defined only on the identical "
           "population")

    # M60, THE ONE THAT SURVIVED: a CONSTANT vector produces a perfectly
    # well-formed comparison (all-tied scores give AUC 0.5), so the leg read
    # as computed and the gate reported a beat against nothing. A number
    # cannot say where it came from.
    for _lbl, _bad in (
            ("a bare list of numbers", [0.0] * 40),
            ("a dict with only p_fill", {"p_fill": [0.0] * 40}),
            ("the right arm but NO provenance",
             {"p_fill": [0.0] * 40, "arm": INCUMBENT_ARM,
              "head": "Q1_arrival"}),
            ("provenance but the WRONG head",
             {"p_fill": [0.0] * 40, "arm": INCUMBENT_ARM, "head": "Q4",
              "provenance": {"sha256_prefix": "deadbeef"}})):
        try:
            q1_incremental(_better, _bad, _r, _y)
            ok(False, f"R-397(1) KNOWN-BAD ({_lbl}) must REFUSE")
        except RuntimeError as e:
            ok("REFUSED" in str(e),
               f"R-397(1) KNOWN-BAD ({_lbl}): Q1's leg REFUSES a predictor "
               f"that cannot identify itself — unwiring the call and handing "
               f"back a constant produced a well-formed AUC and passed")

    # windows that cannot carry an AUC are EXCLUDED BY NAME and counted
    _one = [1] * 40
    _ex = q1_incremental(_better, inc(_worse), _r, _one)
    ok(_ex["windows_used"] == 0
       and _ex["windows_excluded"]["single_class_window"] == 5
       and _ex["increment_null_REPORTED_NOT_ADJUDICATED"] is None,
       f"R-397(1) single-class windows are EXCLUDED BY NAME and COUNTED "
       f"(rule 4), and with none usable the null is absent rather than "
       f"fabricated (got {_ex['windows_excluded']})")
    ok("does not sum" in _win["why_the_null_is_not_adjudicated"],
       "R-397(1) the artifact STATES why the 5(2) null is reported and not "
       "adjudicated: AUC does not sum over windows, so its p would not "
       "describe the declared statistic")

    # ---- RULING 2: each head answers ITS OWN gate ---------------------
    def fam(q1_beats=True):
        cells = {}
        for a in I11.ARMS_011:
            for h in I11.HEADS_011:
                for b in I11.BUDGETS_011:
                    cells[I11.cell_key(a, h, b)] = I11.build_cell(
                        a, h, b, statistic=1.0, p_value=0.001,
                        status=I11.CELL_STATUS_OK, n_actions=10)
        f = I11.assemble_family(cells)
        # Q4's leg must be present too, or the RR2-1 predicate fires on Q4
        # and the control would report Q1 passing for the wrong reason.
        return {"family": f, "results": {"btc": {
            a: {"q1_incremental": {"beats_incumbent_hazard_head": q1_beats,
                                   "incumbent_auc": 0.5,
                                   "incumbent_provenance": {
                                       "sha256_prefix": "deadbeef"}},
                "economics": {b: {"paired_against_incumbent": True,
                                  "incumbent_net_cents": 1.0}
                              for b in I11.BUDGETS_011}}
            for a in I11.ARMS_011}}}

    _f = fam(q1_beats=True)
    attach_declared_gate_outcomes(_f)
    _q3 = _f["family"]["cells"]["composed_lgbm/Q3_magnitudes/5%"]
    _q2 = _f["family"]["cells"]["composed_lgbm/Q2_sign/5%"]
    _q1 = _f["family"]["cells"]["composed_lgbm/Q1_arrival/5%"]
    ok(_q3["declared_gate_outcome"]["passed"] is True
       and "incumbent" not in str(_q3["declared_gate_outcome"]["conjuncts"]),
       "R-397(2) Q3 answers its OWN gate — two slope conjuncts, no incumbent "
       "term anywhere in it")
    ok(_q2["declared_gate_outcome"]["passed"] is None
       and _q2["declared_gate_outcome"]["conjuncts"]["incumbent"] is None,
       "R-397(2) Q2's incumbent conjunct reads NULL, not false: the incumbent "
       "has no sign head, so it is UNANSWERABLE, not answered-and-failed")
    ok(_q1["declared_gate_outcome"]["passed"] is True
       and _q1["declared_gate_outcome"]["conjuncts"]["incumbent_hazard"] is True,
       "R-397(1+2) Q1's gate is now FULLY evaluated: both conjuncts present")
    _fl = fam(q1_beats=False)
    attach_declared_gate_outcomes(_fl)
    ok(_fl["family"]["cells"]["composed_lgbm/Q1_arrival/5%"][
           "declared_gate_outcome"]["passed"] is False,
       "R-397(1) KNOWN-BAD: when the candidate does NOT beat the incumbent "
       "hazard head its gate FAILS — the wiring can produce a failure, which "
       "is what makes the pass meaningful")
    ok(_f["family"]["declared_gate_outcomes"]["counts"]["not_evaluable"] >= 6,
       "R-397(2) heads whose gate cannot be fully evaluated are COUNTED as "
       "not-evaluable rather than folded into 'failed'")

    # the joint reading stays a SEPARATE question
    ok("survives_joint_reading_at_0_05" in
       _f["family"]["declared_gate_outcomes"]["separate_from"],
       "R-397(2) the artifact says the gate outcome and the joint reading are "
       "different questions, answered in different fields")

    # ---- the RR2-1 predicate must now PASS ---------------------------
    _art = {"family": _f["family"],
            "incumbent_null_applicability": {"comparable": dict(
                INCUMBENT_COMPARABLE)},
            "results": _f["results"]}
    _art["incumbent_legs_evaluated"] = incumbent_legs_evaluated(_art)
    ok(_art["incumbent_legs_evaluated"]["Q1_arrival"][
           "incumbent_counterpart_computed"] is True,
       "R-397(1) incumbent_legs_evaluated reads TRUE for Q1 — read from the "
       "RESULT (a real incumbent AUC), never from the presence of a call site")
    _ge = apply_gate_evaluation_status(_art)
    _q1part = [k for k in _ge["cells_gate_partially_evaluated"]
               if _art["family"]["cells"][k]["head"] == "Q1_arrival"]
    ok(_q1part == [],
       f"R-397(1) the RR2-1 predicate now PASSES FOR Q1: none of its cells is "
       f"GATE_PARTIALLY_EVALUATED once the leg is wired (got {_q1part}). "
       f"Q2/Q4 still are, under RR4-1, because their conjuncts are genuinely "
       f"unevaluated — a different fact about different heads")
    # ...and it must still FIRE if the leg goes away, or the pass means nothing
    _art2 = json.loads(json.dumps(_art))
    for _rep in _art2["results"]["btc"].values():
        _rep["q1_incremental"] = {}
    _art2["incumbent_legs_evaluated"] = incumbent_legs_evaluated(_art2)
    _ge3 = apply_gate_evaluation_status(_art2)
    ok(len([k for k in _ge3["cells_gate_partially_evaluated"]
            if _art2["family"]["cells"][k]["head"] == "Q1_arrival"]) == 6,
       "R-397(1) KNOWN-BAD: remove the leg's RESULT and the predicate fires "
       "again on all six Q1 cells — the pass is evidence, not a disabled "
       "check")


def _selftest_frozen_prereg_anchor(ok):
    """RR3-1: the premise must terminate at bytes no seat can edit."""
    _g = _frozen_prereg_section3()
    ok(len(_g) == 4 and _g["Q1_arrival"]["carries_incumbent_term"] is True
       and _g["Q3_magnitudes"]["carries_incumbent_term"] is False,
       f"RR3-1 the FROZEN section-3 gate table is read from git at "
       f"{PREREG_COMMIT}, not from the working tree (4 rows; Q1 names an "
       f"incumbent, Q3 does not)")
    ok(_g["Q2_sign"]["inherited"] is True
       and _g["Q2_sign"]["carries_incumbent_term"] is True,
       "RR3-1 Q2's 'same' row is resolved as an INHERITANCE — read literally "
       "it names no incumbent, and a parser that did so would report a "
       "mismatch against a correct constant")

    # POSITIVE CONTROL, and it must ADMIT the real pair.
    _ev = assert_constants_match_frozen_prereg()
    ok(_ev["heads_where_they_DIVERGE"] == ["Q2_sign"],
       f"RR3-1 POSITIVE CONTROL: the real pair is ADMITTED and the REQUIRED "
       f"divergence is exactly Q2 (got {_ev['heads_where_they_DIVERGE']})")
    ok(_ev["incumbent_has_head"]["Q4_combined_ev"] is True
       and _ev["gate_carries_incumbent_term"]["Q2_sign"] is True
       and _ev["incumbent_has_head"]["Q2_sign"] is False,
       "RR3-1 the two constants terminate in DIFFERENT places: the document "
       "says Q2's gate names an incumbent, the hash-verified artifact says "
       "the incumbent has no sign head")

    # THE REVIEWER'S KNOWN-BAD: constant AND receipt edited together, which
    # RR2-2's floor admitted at 6 checks with coverage_is_complete true.
    _saved = dict(INCUMBENT_COMPARABLE)
    try:
        for _lbl, _mut, _want in (
                ("Q4 flipped false (the reviewer's pair)",
                 {"Q4_combined_ev": False}, "HASH-VERIFIED incumbent"),
                ("Q2 flipped true (a counterpart invented)",
                 {"Q2_sign": True}, "HASH-VERIFIED incumbent")):
            INCUMBENT_COMPARABLE.clear()
            INCUMBENT_COMPARABLE.update({**_saved, **_mut})
            try:
                assert_constants_match_frozen_prereg()
                ok(False, f"RR3-1 KNOWN-BAD ({_lbl}) must REFUSE")
            except RuntimeError as e:
                ok(_want in str(e),
                   f"RR3-1 KNOWN-BAD ({_lbl}): REFUSED — the expected set no "
                   f"longer terminates at a line a seat can edit")
    finally:
        INCUMBENT_COMPARABLE.clear()
        INCUMBENT_COMPARABLE.update(_saved)
    ok(dict(INCUMBENT_COMPARABLE) == _saved,
       "RR3-1 the control RESTORES the constant it mutated — a suite that "
       "leaves a constant edited poisons every check after it")

    # THE HARMONISATION GUARD, driven DIRECTLY. Inside the anchored function
    # it is unreachable — the artifact check fires first on every input that
    # would merge the two maps — so testing it through the constants would
    # have been a control that cannot fail.
    ok(assert_propositions_not_harmonised(
        {"a": True, "b": True}, {"a": True, "b": False}) == ["b"],
       "RR3-1 the harmonisation guard ADMITS maps that disagree, and names "
       "where")
    try:
        assert_propositions_not_harmonised({"a": True, "b": False},
                                           {"a": True, "b": False})
        ok(False, "RR3-1 identical maps must REFUSE as harmonised")
    except RuntimeError as e:
        ok("HARMONISED" in str(e),
           "RR3-1 KNOWN-BAD: two maps agreeing on every head are REFUSED as "
           "harmonised — the Q2 difference is what RR2-1 rests on and must "
           "not be tidied away")

    # THE GATE TEXT, not only the boolean. A mutation rewriting Q3's gate
    # string left the boolean untouched and passed everything, while the
    # artifact would publish a gate the frozen document does not contain.
    ok(assert_gate_text_matches({"a": "x  y"}, {"a": "x y"})["heads_checked"]
       == ["a"],
       "RR3-1 the gate-TEXT check ADMITS the frozen wording (whitespace "
       "normalised, because a table cell wraps)")
    for _lbl, _d, _c in (("a paraphrase", {"a": "beats the incumbent"},
                          {"a": "beats the incumbent hazard head"}),
                         ("a missing string", {"a": "x"}, {}),
                         ("an added conjunct", {"a": "beats X"},
                          {"a": "beats X AND beats Y"})):
        try:
            assert_gate_text_matches(_d, _c)
            ok(False, f"RR3-1 gate-text drift ({_lbl}) must REFUSE")
        except RuntimeError as e:
            ok("not the frozen section-3 wording" in str(e),
               f"RR3-1 KNOWN-BAD ({_lbl}): a published gate string that is "
               f"not the document's is REFUSED — the boolean check cannot "
               f"see it")

    # THE TRANSCRIPTION CHECK, driven DIRECTLY — on the real document the two
    # already agree, so a mutant deleting it changed nothing and survived.
    ok(assert_gate_terms_match({"a": True}, {"a": True})["heads_checked"] == ["a"],
       "RR3-1 the transcription check ADMITS a faithful transcription")
    for _lbl, _d, _c in (("a term the document does not have",
                          {"a": False}, {"a": True}),
                         ("a term the document has and the code drops",
                          {"a": True}, {"a": False}),
                         ("a head missing from the code", {"a": True}, {})):
        try:
            assert_gate_terms_match(_d, _c)
            ok(False, f"RR3-1 transcription drift ({_lbl}) must REFUSE")
        except RuntimeError as e:
            ok("disagrees with the FROZEN" in str(e),
               f"RR3-1 KNOWN-BAD ({_lbl}): DECLARED_GATES is REFUSED when it "
               f"stops transcribing the frozen table")

    # THE PARSE RULES, driven on synthetic bodies for the same reason.
    _good = ("\n## 3.\n| # | q | h | m | gate |\n"
             "| Q1 | a | b | c | beats the incumbent hazard head |\n"
             "| Q2 | a | b | c | same, on the fill-conditional population |\n"
             "| Q3 | a | b | c | calibration slope CI excludes 0 |\n"
             "| Q4 | a | b | c | beats the incumbent by a null |\n## 4.\n")
    _p = _parse_section3(_good, "fixture")
    ok(len(_p) == 4 and _p["Q2_sign"]["carries_incumbent_term"] is True
       and _p["Q3_magnitudes"]["carries_incumbent_term"] is False,
       "RR3-1 the parser ADMITS a well-formed table and resolves 'same'")
    for _lbl, _body, _want in (
            ("only three rows", _good.replace(
                "| Q4 | a | b | c | beats the incumbent by a null |\n", ""),
             "expected 4"),
            ("no section 3 at all", "## 9. something\n", "no section 3"),
            ("'same' with nothing to inherit", _good.replace(
                "| Q1 | a | b | c | beats the incumbent hazard head |\n", ""),
             "no preceding row")):
        try:
            _parse_section3(_body, "fixture")
            ok(False, f"RR3-1 the parser must REFUSE ({_lbl})")
        except RuntimeError as e:
            ok(_want in str(e),
               f"RR3-1 KNOWN-BAD ({_lbl}): a partial or malformed parse "
               f"REFUSES rather than anchoring the constants to a fragment")

    # A document it cannot read is a REFUSAL, never a quiet pass.
    for _lbl, _kw in (("an unknown commit", {"commit": "0000000"}),
                      ("a missing document", {"doc": "NO_SUCH_DOC.md"})):
        try:
            _frozen_prereg_section3(**_kw)
            ok(False, f"RR3-1 {_lbl} must REFUSE")
        except RuntimeError as e:
            # NAMED, not merely refused. Without the git check the empty
            # stdout falls through to "no section 3", which is a TRUE
            # statement about an unread file — a refusal for the wrong
            # reason, and the mutant that removed the git check survived on
            # a control that accepted any refusal at all.
            ok("git show" in str(e),
               f"RR3-1 KNOWN-BAD ({_lbl}): the GIT read failure is named as "
               f"itself, not absorbed by a later branch reporting an empty "
               f"parse of a file that was never read")


def _selftest_gate_evaluation_rr2(ok):
    """RR2-1 both directions: the CURRENT shape must fail, a wired one pass."""
    def artifact(leg_computed):
        """A receipt shaped like the shipped one, differing ONLY in whether
        Q1's incumbent leg was computed. One variable, so the control cannot
        pass for a second reason."""
        cells = {}
        for a in I11.ARMS_011:
            for h in I11.HEADS_011:
                for b in I11.BUDGETS_011:
                    cells[I11.cell_key(a, h, b)] = I11.build_cell(
                        a, h, b, statistic=0.83, p_value=1 / 501,
                        status=(I11.CELL_STATUS_OK
                                if h in ("Q1_arrival", "Q4_combined_ev")
                                else I11.CELL_STATUS_NO_COUNTERPART),
                        n_actions=10, detail="fixture")
        econ = {b: {"paired_against_incumbent": True, "net_cents": 5.0,
                    "incumbent_net_cents": 1.0, "n_actions_total": 10,
                    "increment_by_window": {f"w{i}": 1.0 for i in range(8)}}
                for b in I11.BUDGETS_011}
        r = {"family": I11.assemble_family(cells),
             "incumbent_null_applicability": {"comparable": dict(
                 INCUMBENT_COMPARABLE)},
             "results": {"btc": {a: {
                 "economics": econ,
                 # RR4-1 derives the conjunct from the RESULT, so a fixture
                 # that set `incumbent_legs_evaluated` directly without this
                 # no longer passes — correctly, and it is stricter: the leg
                 # fact and the evidence behind it can no longer disagree.
                 **({"q1_incremental": {
                     "beats_incumbent_hazard_head": True,
                     "incumbent_auc": 0.71,
                     "incumbent_provenance": {"sha256_prefix": "deadbeef"}}}
                    if leg_computed else {}),
                 # ENOUGH FOR evaluate_family TO REBUILD, because
                 # finalise_family rebuilds rather than reusing the cells --
                 # a fixture that only supplied cells would exercise the
                 # re-status and skip the rebuild it is meant to order.
                 "adjudicated_statistics": {
                     "Q1_arrival": 0.83, "Q2_sign": 0.60,
                     "Q2_cell_status": "OK", "Q3_magnitudes": 0.69},
                 "heads": {
                     "Q1_arrival": {"status": "OK", "auc": 0.83,
                                    "n_actions": 100,
                                    "matched_random": {"status": "OK",
                                                       "p_value": 1 / 501,
                                                       "n_draws": 500}},
                     "Q2_p_pos": {"auc": 0.61, "n_actions": 50,
                                  "matched_random": {"status": "OK",
                                                     "p_value": 1 / 501,
                                                     "n_draws": 500}},
                     "Q2_p_neg": {"auc": 0.60, "n_actions": 50,
                                  "matched_random": {"status": "OK",
                                                     "p_value": 1 / 501,
                                                     "n_draws": 500}},
                     "Q3_m_harm": {"status": "OK", "calibration_slope": 0.69,
                                   "n_actions": 40,
                                   "matched_random": {
                                       "status": "OK", "p_value": 1 / 501,
                                       "n_draws": 500, "no_skill_value": 0.0,
                                       "alternative": "greater"}},
                     "Q3_m_good": {"status": "OK", "calibration_slope": 0.91,
                                   "n_actions": 45,
                                   "matched_random": {
                                       "status": "OK", "p_value": 1 / 501,
                                       "n_draws": 500, "no_skill_value": 0.0,
                                       "alternative": "greater"}}}}
                 for a in I11.ARMS_011}}}
        r["incumbent_legs_evaluated"] = {
            h: {"declared_comparable": bool(v),
                "incumbent_counterpart_computed": (
                    leg_computed if h == "Q1_arrival" else bool(v)),
                "note": "fixture"}
            for h, v in INCUMBENT_COMPARABLE.items()}
        return r

    # DIRECTION 1 — THE SHIPPED SHAPE MUST FAIL. Q1's leg is NOT computed,
    # so its six cells stop being published as surviving. This is the
    # reviewer's "the current artifact must FAIL the new check".
    _a = artifact(leg_computed=False)
    _ge = apply_gate_evaluation_status(_a)
    _a["family"] = I11.assemble_family(_a["family"]["cells"])
    _q1 = [k for k, c in _a["family"]["cells"].items()
           if c["head"] == "Q1_arrival"]
    # RR4-1 GENERALISES THIS: Q4's matched-random conjunct is uncomputed, so
    # its six cells re-status too. Q1's six are the RR2-1 case (a declared
    # counterpart that exists and was not computed); both are the same rule
    # now — a conjunct nobody evaluated.
    ok("Q1_arrival" in _ge["heads_affected"]
       and all(_a["family"]["cells"][k]["status"]
               == I11.CELL_STATUS_GATE_PARTIAL for k in _q1),
       f"RR2-1 KNOWN-BAD (the SHIPPED shape): all six Q1 cells are re-statused "
       f"GATE_PARTIALLY_EVALUATED because their gate names a counterpart that "
       f"EXISTS and was not computed (heads affected "
       f"{_ge['heads_affected']})")
    ok(all("incumbent leg" in _a["family"]["cells"][k]["gate_partial_reason"]
           for k in _q1),
       "RR4-1 and Q1's cells keep the SPECIFIC reason (the incumbent leg), "
       "not the generic one — the two cases are one rule and still "
       "distinguishable in the receipt")
    ok(not any(_a["family"]["cells"][k]["survives_joint_reading_at_0_05"]
               for k in _q1),
       "RR2-1 and NONE of them is published as surviving — the flag stops "
       "asserting a joint reading the run did not complete")
    ok(all(_a["family"]["cells"][k]["statistic"] is not None
           and _a["family"]["cells"][k]["p_value"] is not None for k in _q1)
       and len(_a["family"]["cells"]) == 24
       and _a["family"]["holm_denominator"] == 24,
       "RR2-1 REPORTED, NEVER DROPPED: the statistic and p are unchanged and "
       "still carried, and the denominator stays 24 so a later ruling can be "
       "applied without re-running")

    # DIRECTION 2 — A WIRED ARTIFACT MUST PASS. Same fixture, one variable
    # flipped. Without this the check is a predicate that only ever refuses.
    _b = artifact(leg_computed=True)
    _ge2 = apply_gate_evaluation_status(_b)
    _b["family"] = I11.assemble_family(_b["family"]["cells"])
    _q1b = [k for k, c in _b["family"]["cells"].items()
            if c["head"] == "Q1_arrival"]
    ok(all(_b["family"]["cells"][k]["status"] == I11.CELL_STATUS_OK
           for k in _q1b)
       and not any(c["head"] == "Q1_arrival"
                   for c in (_b["family"]["cells"][k]
                             for k in _ge2["cells_gate_partially_evaluated"])),
       "RR2-1 POSITIVE CONTROL: with Q1's leg COMPUTED its cells are OK again "
       "— the conjunct narrows the claim, it does not disable the head")
    ok(all(_b["family"]["cells"][k]["survives_joint_reading_at_0_05"]
           for k in _q1b),
       "RR2-1 and the six Q1 cells DO survive again once their gate is whole")

    # The three inputs must each be load-bearing, or the predicate is passing
    # for a reason it does not name.
    for _lbl, _mut in (
            ("gate carries no incumbent term",
             lambda r: [c.update({"declared_gate": dict(
                 c["declared_gate"], carries_incumbent_term=False)})
                 for c in r["family"]["cells"].values()
                 if c["head"] == "Q1_arrival"]),
            ("the head is not declared comparable",
             lambda r: r["incumbent_null_applicability"]["comparable"].update(
                 {"Q1_arrival": False}))):
        _c = artifact(leg_computed=False)
        _mut(_c)
        attach_declared_gate_outcomes(_c)
        _aff = apply_gate_evaluation_status(_c)["cells_gate_partially_evaluated"]
        # Under RR4-1 Q1 still re-statuses here — but for the GENERIC reason
        # (a conjunct nobody evaluated), never the incumbent-leg one. That
        # distinction is what proves the leg predicate needs all three inputs
        # rather than firing whenever a leg is absent.
        _reasons = {_c["family"]["cells"][k]["gate_partial_reason"]
                    for k in _aff
                    if _c["family"]["cells"][k]["head"] == "Q1_arrival"}
        ok(not any("incumbent leg" in r for r in _reasons),
           f"RR2-1 the incumbent-leg predicate needs ALL THREE inputs: with "
           f"'{_lbl}' Q1 carries no incumbent-leg reason (got {_reasons or 'none'}), "
           f"so it is not simply refusing whenever a leg is missing")

    # THE ORDERING, THROUGH finalise_family ITSELF. The control below proves
    # the re-assembly matters; this one proves finalise_family DOES it. A
    # mutant deleting the re-assembly survived until this existed, because
    # the dry run's synthetic data produces no survivors and so no stale flag
    # can arise there.
    _fin = artifact(leg_computed=False)
    _fin["populations"] = {"btc": {"eval_n_actions": 100}}
    finalise_family(_fin)
    _bad = [k for k, c in _fin["family"]["cells"].items()
            if c["status"] == I11.CELL_STATUS_GATE_PARTIAL
            and c.get("survives_joint_reading_at_0_05")]
    ok(not _bad and _fin["family"]["gate_evaluation"]["cells_checked"] == 24,
       f"RR2-1 finalise_family RE-ASSEMBLES after re-statusing: no cell is "
       f"left GATE_PARTIALLY_EVALUATED while still flagged surviving (got "
       f"{len(_bad)} such cells)")
    ok(all(not c["survives_joint_reading_at_0_05"]
           for c in _fin["family"]["cells"].values()
           if c["head"] == "Q1_arrival"),
       "RR2-1 and through the ORDERED path the six Q1 cells stop surviving — "
       "which is the headline change this finding requires")

    # ORDERING: re-statusing without RE-ASSEMBLING leaves the old verdict
    # standing beside the new status. That is the step easiest to omit.
    _d = artifact(leg_computed=False)
    apply_gate_evaluation_status(_d)
    _stale = [k for k, c in _d["family"]["cells"].items()
              if c["status"] == I11.CELL_STATUS_GATE_PARTIAL
              and c.get("survives_joint_reading_at_0_05")]
    ok(bool(_stale),
       "RR2-1 ORDERING CONTROL: without the re-assembly the re-statused cells "
       "still carry survives=true — proving the re-assembly step in "
       "finalise_family is load-bearing rather than decorative")
    # A COMMENT IS NOT A CALL. This asserted the bare name `finalise_family`
    # and matched the COMMENT one line above the call, so a mutant replacing
    # the call with the ad-hoc sequence SURVIVED a guard written to catch
    # exactly that. Match the call, and require the ad-hoc form to be absent.
    _ms = Path(__file__).read_text(encoding="utf-8")
    _ms = _ms[_ms.index("\ndef main() -> int:"):]
    ok("finalise_family(out)" in _ms
       and "out[\"family\"] = evaluate_family(" not in _ms,
       "RR2-1 main() runs the ORDERED sequence: it CALLS finalise_family and "
       "does NOT rebuild the family ad-hoc — matched on the call, because the "
       "bare name also matches the comment beside it")


def _selftest_review_batch_f2_f7(ok):
    """F-2..F-7 from the reviewer's filing. Each must fire AND admit."""
    # ---- F-3 the floor disclosure -------------------------------------
    _f = permutation_floor_disclosure(1 / 501, 500, 24)
    _k = "holm_if_EVERY_at_floor_cell_moved_together_multiplied"
    ok(_f["at_permutation_floor"] is True
       and abs(_f["holm_at_floor"] - 24 / 501) < 1e-15
       and abs(_f[_k] - 48 / 501) < 1e-15
       and _f["margin_in_draws"] == 1,
       f"F-3 a p at 1/(n+1) is NAMED as the floor with the one-draw margin: "
       f"holm {_f['holm_at_floor']:.4f} -> {_f[_k]:.4f} if EVERY at-floor "
       f"cell moved together (RR4-2 names it as the whole-family case)")

    # RR4-2: the SINGLE-CELL consequence is milder, and it must be COMPUTED.
    # A cell that moves alone sorts BEHIND the still-tied cells, so Holm's
    # step-down gives it a smaller multiplier than m. Multiplying by m — the
    # round-1 framing, which the reviewer corrected as originally its own —
    # overstates what one draw would do to one cell.
    _tied = {f"c{i}": 1 / 501 for i in range(24)}
    _one = dict(_tied); _one["c0"] = 2 / 501
    _h_one = _holm_over(_one, 24)["c0"]
    _h_all = _holm_over({k: 2 / 501 for k in _tied}, 24)["c0"]
    ok(_h_one < _h_all and abs(_h_all - 48 / 501) < 1e-12,
       f"RR4-2 one cell moving alone is adjusted MORE GENTLY than the whole "
       f"family moving together ({_h_one:.4f} vs {_h_all:.4f}), because it "
       f"sorts behind the still-tied cells — the multiplied figure is the "
       f"whole-family case, not the single-cell one")
    ok(_h_one < 0.05 <= _h_all,
       f"RR4-2 and the two answer DIFFERENTLY at the bar: alone it would "
       f"still survive ({_h_one:.4f}), together it would not ({_h_all:.4f}) "
       f"— which is exactly why both are emitted")
    _nf = permutation_floor_disclosure(2 / 501, 500, 24)
    ok(_nf["at_permutation_floor"] is False and _nf["margin_in_draws"] is None,
       "F-3 KNOWN-BAD: a p ONE draw off the floor is NOT reported as at it — "
       "a disclosure that fired on every cell would say nothing")
    ok(permutation_floor_disclosure(None, 500)["at_permutation_floor"] is False
       and permutation_floor_disclosure(0.5, None)["at_permutation_floor"]
       is False,
       "F-3 a missing p or a missing draw count leaves the floor UNDEFINED "
       "rather than defaulting to a claim either way")
    # it must reach the SURVIVING cells, which is the whole finding
    _rc = {"family": {"cells": {}, "holm_denominator": 24},
           "results": {"btc": {"composed_linear": {"heads": {
               "Q1_arrival": {"matched_random": {"n_draws": 500}}}}}}}
    _rc["family"]["cells"]["composed_linear/Q1_arrival/5%"] = I11.build_cell(
        "composed_linear", "Q1_arrival", "5%", statistic=0.83,
        p_value=1 / 501, n_actions=10)
    _rc["family"]["cells"]["composed_linear/Q1_arrival/5%"][
        "survives_joint_reading_at_0_05"] = True
    _sm = attach_floor_disclosure(_rc)
    ok(_sm["cells_at_floor"] == 1
       and _sm["SURVIVING_cells_at_floor"] == ["composed_linear/Q1_arrival/5%"],
       f"F-3 the disclosure reaches the SURVIVING cell — it was present on "
       f"all six Q3 cells, which decide nothing, and on ZERO Q1 cells, which "
       f"are the head that survives (got {_sm['SURVIVING_cells_at_floor']})")

    # ---- F-4 the head's OWN declared gate ------------------------------
    _c3 = I11.build_cell("composed_linear", "Q3_magnitudes", "5%", n_actions=1)
    _c1 = I11.build_cell("composed_linear", "Q1_arrival", "5%", n_actions=1)
    ok(_c3["declared_gate"]["carries_incumbent_term"] is False
       and _c1["declared_gate"]["carries_incumbent_term"] is True,
       "F-4 each cell carries its head's OWN frozen gate, and the field "
       "DISCRIMINATES: Q3's gate has no incumbent term, Q1's does — so "
       "'survives' and 'passed its declared gate' are separable questions")
    ok("CI excludes 0" in _c3["declared_gate"]["gate"]
       and "no incumbent term" in _c3["declared_gate"]["note"].lower()
       or "NO incumbent term" in _c3["declared_gate"]["note"],
       "F-4 and Q3's cell states WHY a NO_INCUMBENT_COUNTERPART status is a "
       "question for the USER rather than a settled verdict")

    # ---- F-5 the unit travels with the number --------------------------
    _cw = I11.build_cell("composed_linear", "Q4_combined_ev", "5%",
                         n_actions=166, statistic_n=166,
                         statistic_n_unit="windows")
    _ca = I11.build_cell("composed_linear", "Q2_sign", "5%", n_actions=17604,
                         statistic_n=17604, statistic_n_unit="actions")
    ok(_cw["statistic_n_unit"] == "windows"
       and _ca["statistic_n_unit"] == "actions",
       "F-5 the unit travels with the n: Q4's 166 are WINDOWS while the other "
       "cells carry ACTIONS, and the field name asserts the wrong one")
    ok("UNSTATED" in I11.build_cell("a", "b", "c")["statistic_n_unit"],
       "F-5 KNOWN-BAD: an unstated unit says UNSTATED rather than defaulting "
       "to 'actions', which would be a claim nobody made")

    # ---- F-2 the frozen-form p is REPORTED -----------------------------
    _cp = I11.build_cell("composed_linear", "Q4_combined_ev", "5%",
                         p_value=0.02, p_two_sided=0.0499)
    ok(_cp["p_two_sided_REPORTED_NOT_ADJUDICATED"] == 0.0499
       and _cp["p_value"] == 0.02,
       "F-2 the FROZEN two-sided form travels beside the adjudicated "
       "one-sided p (R-288 promised it and it occurred ZERO times)")
    ok(I11.build_cell("a", "b", "c")["p_two_sided_REPORTED_NOT_ADJUDICATED"]
       is None,
       "F-2 KNOWN-BAD: a cell with no incremental null carries None, not a "
       "fabricated diagnostic")

    # ---- F-6 distinct results beside the cell count --------------------
    def famcells(q4_varies):
        cs = {}
        for a in I11.ARMS_011:
            for h in I11.HEADS_011:
                for i, b in enumerate(I11.BUDGETS_011):
                    st = (1.0 + i if (q4_varies and h == "Q4_combined_ev")
                          else 1.0)
                    cs[I11.cell_key(a, h, b)] = I11.build_cell(
                        a, h, b, statistic=st, p_value=0.01, n_actions=1)
        return cs
    _d = I11._distinct_results(famcells(True))
    # 3, not 4: Q1/Q2/Q3 all share (1.0, 0.01) in this fixture, so the
    # DISTINCT PAIRS are Q4's three. My first expectation counted heads
    # instead of pairs and the control caught it.
    ok(_d["declared_cells"] == 24 and _d["distinct_overall"] == 3
       and _d["distinct_per_head"]["Q1_arrival"] == 1
       and _d["distinct_per_head"]["Q4_combined_ev"] == 3,
       f"F-6 the DISTINCT result count is stated beside the cell count: "
       f"budget-invariant heads carry ONE number replicated three times "
       f"(got {_d['distinct_overall']} distinct over 24 cells)")
    _d2 = I11._distinct_results(famcells(False))
    ok(_d2["distinct_overall"] == 1 and _d2["distinct_per_head"][
        "Q4_combined_ev"] == 1,
       "F-6 KNOWN-BAD: with Q4 also invariant the count falls to 1 — the "
       "field measures the replication rather than restating the cell count")

    # ---- F-7 tri-state, named paths, carrying_commit -------------------
    # THE UNREADABLE-GIT BRANCH, EXECUTED. Injecting the failure is the only
    # way to reach it on a working machine, and without it a mutant deleting
    # the tri-state survived a green suite.
    _unk = producing_code_provenance(_git_runner=lambda *a: None)
    ok(_unk["working_tree_dirty"] == "unknown"
       and _unk["producing_code_was_clean"] is None
       and _unk["dirty_paths"] is None,
       f"F-7 KNOWN-BAD: an UNREADABLE git status reports dirty='unknown', not "
       f"False — rule 11 applies to the FLAG, not to a note beside it (got "
       f"{_unk['working_tree_dirty']!r})")
    ok(_unk["fit_code_ref_resolved"] is False
       and "UNAVAILABLE" in _unk["fit_code_ref"],
       "F-7 and the ref is a NAMED absence in the same failure, so neither "
       "field silently reads as a pass")
    _pv = producing_code_provenance()
    ok(_pv["working_tree_dirty"] in (True, False, "unknown")
       and _pv["working_tree_dirty"] != "unknown",
       f"F-7 the dirty flag is TRI-STATE (got "
       f"{_pv['working_tree_dirty']!r}); an unreadable git status is never "
       f"reported as clean")
    ok(_pv.get("carrying_commit") and _pv["carrying_commit"] == _pv["fit_code_ref"],
       "F-7 every result-bearing receipt gains a carrying_commit field")
    if _pv["working_tree_dirty"] is True:
        ok(isinstance(_pv["dirty_paths"], list) and _pv["dirty_paths"],
           f"F-7 a DIRTY tree NAMES its paths, so a reader need not use git "
           f"to learn whether the dirt touched the producing code (got "
           f"{len(_pv['dirty_paths'])} paths)")
        ok(_pv["producing_code_was_clean"] is (not _pv[
            "dirty_paths_touching_the_producing_code"]),
           "F-7 and it states whether any dirty path touched the 011 modules "
           "or the lattice — computed, not asserted")
    else:
        ok(_pv["dirty_paths"] == [] or _pv["dirty_paths"] is None,
           "F-7 a clean (or unreadable) tree names no paths")


def _selftest_artifact_wiring_guard(ok):
    """F-1: the guard that sees a value NOT FLOWING, which source text cannot."""
    def receipt(paired=True, q4_status=I11.CELL_STATUS_OK, comparable=True):
        cells = {}
        for a in I11.ARMS_011:
            for h in I11.HEADS_011:
                for b in I11.BUDGETS_011:
                    cells[I11.cell_key(a, h, b)] = I11.build_cell(
                        a, h, b, statistic=1.0, p_value=0.01,
                        status=(q4_status if h == "Q4_combined_ev"
                                else I11.CELL_STATUS_OK),
                        n_actions=10, detail="fixture")
        econ = {b: {"paired_against_incumbent": paired,
                    "incumbent_net_cents": (1.0 if paired else None),
                    "n_actions_total": 10} for b in I11.BUDGETS_011}
        return {"family": {"cells": cells},
                "incumbent_null_applicability": {"comparable": {
                    "Q1_arrival": True, "Q2_sign": False,
                    "Q3_magnitudes": False,
                    "Q4_combined_ev": comparable}},
                "results": {"btc": {a: {"economics": econ}
                                    for a in I11.ARMS_011}}}

    # POSITIVE CONTROL, and it must ADMIT: a properly wired receipt passes.
    _ev = assert_incumbent_applicability_honoured(receipt())
    ok(_ev["checks"] > 0 and "Q4_combined_ev" in _ev["comparable_heads"],
       f"F-1 POSITIVE CONTROL: a correctly wired receipt is ADMITTED and the "
       f"guard reports how much it read ({_ev['checks']} checks)")

    # THE REVIEWER'S MUTANT A, at the artifact: comparable true, cells saying
    # NO_INCUMBENT_COUNTERPART, economics unpaired. Every source-text guard
    # still passes on that code, which is exactly why this one reads OUTPUT.
    # EACH KNOWN-BAD ISOLATES ONE CONDITION. My first set triggered two at
    # once, so disabling either check left the other still firing and three
    # mutants survived a suite that looked thorough. A known-bad that trips
    # two guards proves neither.
    def econ_only(paired, net):
        return {**receipt(), "results": {"btc": {a: {"economics": {
            b: {"paired_against_incumbent": paired,
                "incumbent_net_cents": net} for b in I11.BUDGETS_011}}
            for a in I11.ARMS_011}}}

    for _lbl, _r in (
            # status alone: economics fully paired, only the cells disagree
            ("cells say no counterpart, economics FINE",
             {**econ_only(True, 1.0), "family": receipt(
                 q4_status=I11.CELL_STATUS_NO_COUNTERPART)["family"]}),
            # paired alone: net present, so only the paired flag is wrong
            ("economics UNPAIRED but net present", econ_only(False, 1.0)),
            # net alone: paired true, so only the null net is wrong
            ("paired but incumbent_net_cents null", econ_only(True, None))):
        try:
            assert_incumbent_applicability_honoured(_r)
            ok(False, f"F-1 KNOWN-BAD ({_lbl}) must REFUSE")
        except RuntimeError as e:
            ok("DECLARED comparable" in str(e),
               f"F-1 KNOWN-BAD ({_lbl}): the artifact-level guard FIRES where "
               f"the source-text guard cannot — a call whose result never "
               f"reaches the consumer leaves every guarded string intact")

    # ...and it must not fire on a head GENUINELY declared not comparable, or
    # it is a guard that refuses universally rather than discriminating. The
    # head used here is one INCUMBENT_COMPARABLE itself declares False
    # (Q2_sign); using a FLIPPED Q4 -- as this control did -- is now refused
    # by RR2-2's coverage floor, correctly, because that is the evasion.
    _ev2 = assert_incumbent_applicability_honoured(receipt())
    ok(_ev2["checks"] > 0
       and "Q2_sign" not in _ev2["comparable_heads"]
       and _ev2["coverage_is_complete"] is True,
       "F-1 CONTROL: heads declared NOT comparable (Q2/Q3) carry "
       "NO_INCUMBENT_COUNTERPART without refusing — the guard discriminates "
       "on the declaration, it does not refuse universally")

    # ---- the RR2-1 INTERACTION, both ways -----------------------------
    # Loosening the status set is the kind of change that quietly disables a
    # guard, so both halves are pinned. A DISCLOSED gap is admitted; the
    # CONTRADICTION it must not be confused with is still refused.
    _disc = receipt()
    for _c in _disc["family"]["cells"].values():
        if _c["head"] == "Q1_arrival":
            _c["status"] = I11.CELL_STATUS_GATE_PARTIAL
    _dv = assert_incumbent_applicability_honoured(_disc)
    ok(len(_dv["cells_admitted_as_DISCLOSED_gaps"]) == 6,
       f"RR2-1/F-1 a comparable head carrying GATE_PARTIALLY_EVALUATED is "
       f"ADMITTED and recorded as a DISCLOSED gap — refusing it would make "
       f"the honest disclosure unemittable and leave the false status as the "
       f"only way to ship (got {len(_dv['cells_admitted_as_DISCLOSED_gaps'])})")
    _contra = receipt()
    for _c in _contra["family"]["cells"].values():
        if _c["head"] == "Q1_arrival":
            _c["status"] = I11.CELL_STATUS_NO_COUNTERPART
    try:
        assert_incumbent_applicability_honoured(_contra)
        ok(False, "RR2-1/F-1 the CONTRADICTION must still refuse")
    except RuntimeError as e:
        ok("DECLARED comparable" in str(e),
           "RR2-1/F-1 KNOWN-BAD: NO_INCUMBENT_COUNTERPART on a comparable "
           "head still REFUSES — the loosened set admits the disclosure, not "
           "the denial")
    # AND MUTANT A MUST STILL DIE, now through the economics half, which no
    # cell status can satisfy. Without this the loosening could have quietly
    # reopened the HOLD.
    _mutA = receipt(paired=False)
    for _c in _mutA["family"]["cells"].values():
        if _c["head"] == "Q4_combined_ev":
            _c["status"] = I11.CELL_STATUS_GATE_PARTIAL
    try:
        assert_incumbent_applicability_honoured(_mutA)
        ok(False, "F-1 MUTANT A must still be refused after the loosening")
    except RuntimeError as e:
        ok("paired_against_incumbent is not true" in str(e),
           "F-1 KNOWN-BAD: an UNWIRED arm wearing the DISCLOSED-gap status is "
           "still refused by the ECONOMICS half — the loosening did not "
           "reopen the hold")

    # ---- RR2-2: coverage must not SHRINK ------------------------------
    # THE REVIEWER'S KNOWN-BAD, verbatim: Q4 unwired AND its `comparable`
    # flipped to false, cells and economics made consistent with it. That was
    # ADMITTED at 6 checks and one head, and the emitted `checks: 6` was the
    # only trace anything had shrunk. Same rule A1.4 already applies to the
    # Holm denominator: a set must not shrink to what was evaluable.
    _shrunk = receipt(paired=False,
                      q4_status=I11.CELL_STATUS_NO_COUNTERPART,
                      comparable=False)
    _shrunk["results"] = {"btc": {a: {"economics": {}} for a in I11.ARMS_011}}
    try:
        _r = assert_incumbent_applicability_honoured(_shrunk)
        ok(False, f"RR2-2 the coverage-shrink receipt must REFUSE (was "
                  f"ADMITTED at {_r.get('checks')} checks)")
    except RuntimeError as e:
        ok("Coverage SHRANK" in str(e) and "Q4_combined_ev" in str(e),
           "RR2-2 KNOWN-BAD: a receipt that flips a declared-comparable head "
           "to false is REFUSED by name — the expected set is a "
           "PRODUCER-RECORDED fact from INCUMBENT_COMPARABLE, never inferred "
           "from what the run happened to contain (R-230)")
    ok(assert_incumbent_applicability_honoured(receipt())[
           "expected_comparable_heads"] == ["Q1_arrival", "Q4_combined_ev"],
       "RR2-2 POSITIVE CONTROL: the expected set is EMITTED, so a reader "
       "compares realised coverage against a declared floor rather than "
       "against nothing")

    # BOTH HALVES, and neither is sufficient alone (rule 17). The
    # artifact-level guard proves a value FLOWED -- that is the half the
    # reviewer's MUTANT A defeated. This source check proves the seam is
    # still CALLED, which is the half no artifact can show: a deleted line
    # leaves nothing behind to inspect, and on a healthy pipeline removing a
    # guard changes no output at all. Measured: a mutant deleting
    # `assert_dry_run_family(out)` survived every other instrument here.
    # Anchored on the definition, never on the bare substring.
    _srcm = Path(__file__).read_text(encoding="utf-8")
    _mainsrc = _srcm[_srcm.index("\ndef main() -> int:"):]
    ok("assert_dry_run_family(out)" in _mainsrc,
       "F-1(2) main()'s dry path still CALLS assert_dry_run_family — a source "
       "check because the failure mode is a DELETED line, which no artifact "
       "and no unit test can see; it complements the artifact guard rather "
       "than substituting for it")
    ok(_mainsrc.count("assert_incumbent_applicability_honoured(out)") >= 1,
       "F-1 and main() still CALLS the artifact-level guard on the real path")
    _rj = _srcm[_srcm.index("\ndef readjudicate("):
                _srcm.index("\ndef declared_outputs_for(")]
    ok("assert_incumbent_applicability_honoured(out)" in _rj
       and "finalise_family(out)" in _rj,
       "F-1/RR2-1 --readjudicate carries the SAME guarantees through the SAME "
       "ordered sequence; a mode that skipped them would be a second door "
       "into the same room, and one that re-statused without re-assembling "
       "would leave the old verdict standing beside the new status")

    # An EMPTY read must refuse, never pass (R-289, the checker's chair).
    for _lbl, _bad, _want in (
            ("no cells", {"incumbent_null_applicability":
                          {"comparable": {"Q4_combined_ev": True}}},
             "empty read"),
            ("no declaration", {"family": {"cells": {"x": {}}}}, "empty read"),
            # THE ZERO-VISIT CASE, isolated: cells and declarations both
            # present and non-empty, but NOTHING matches -- so the loop body
            # never runs. The two above are caught by the entry check and
            # left the `checked == 0` refusal untested; a mutant deleting it
            # survived.
            ("declared head no cell carries",
             {"family": {"cells": {"k": {"head": "Q1_arrival",
                                         "status": "OK"}}},
              "incumbent_null_applicability": {"comparable": {"Q9_absent": True}},
              "results": {}},
             "ZERO cells")):
        try:
            assert_incumbent_applicability_honoured(_bad)
            ok(False, f"F-1 an empty read ({_lbl}) must REFUSE")
        except RuntimeError as e:
            ok(_want.lower() in str(e).lower(),
               f"F-1 KNOWN-BAD ({_lbl}): a guard that runs on an empty read "
               f"reports a pass it never established (wanted {_want!r})")

    # The dry seam's own assertion, in all four directions. It asserts the
    # OUTPUT, and it also asserts that the guards LEFT EVIDENCE -- because a
    # guard REMOVED from main() cannot be caught by running a healthy
    # pipeline: it only fires when something else is broken too.
    def emitted(**kw):
        r = receipt(**kw)
        r["incumbent_applicability_guard"] = {"checks": 18}
        r["frozen_prereg_anchor"] = {
            "chain_terminates_at": "fixture",
            "gate_carries_incumbent_term": {"Q1_arrival": True},
            "gate_text_verified_heads": list(I11.HEADS_011)}
        r["incumbent_legs_evaluated"] = {
            h: {"incumbent_counterpart_computed": True}
            for h, v in INCUMBENT_COMPARABLE.items() if v}
        r["family"]["gate_evaluation"] = {"cells_checked": 24,
                                          "cells_gate_partially_evaluated": []}
        r["family"]["declared_gate_outcomes"] = {
            "counts": {"passed": len(r["family"]["cells"]), "failed": 0,
                       "not_evaluable": 0}}
        for c in r["family"]["cells"].values():
            c["permutation_floor"] = {"at_permutation_floor": False}
        return r

    def _strip_floor(r):
        r = json.loads(json.dumps(r))
        for c in r["family"]["cells"].values():
            c.pop("permutation_floor", None)
        return r

    def _stale_receipt(r):
        """A cell re-statused but never re-assembled — the ordering defect."""
        r = json.loads(json.dumps(r))
        k = next(iter(r["family"]["cells"]))
        r["family"]["cells"][k]["status"] = I11.CELL_STATUS_GATE_PARTIAL
        r["family"]["cells"][k]["survives_joint_reading_at_0_05"] = True
        return r

    ok(assert_dry_run_family(emitted())["all_carry_an_increment"],
       "F-1(2) POSITIVE CONTROL: --dry-run's family assertion ADMITS a wired "
       "family that carries its guards' evidence")
    for _lbl, _r, _want in (
            ("Q4 carries no increment",
             emitted(paired=False,
                     q4_status=I11.CELL_STATUS_NO_COUNTERPART),
             "no increment"),
            ("the applicability guard left NO evidence",
             {**emitted(), "incumbent_applicability_guard": {}},
             "was not called"),
            # built from `emitted()` and stripped of ONLY the floor keys, so
            # the earlier gate/guard assertions still pass and this known-bad
            # isolates the condition it names rather than tripping two.
            ("the floor disclosure never reached the cells",
             _strip_floor(emitted()),
             "permutation_floor"),
            ("the declared-gate pass left NO evidence (R-397 ruling 2)",
             {**emitted(), "family": {**emitted()["family"],
                                      "declared_gate_outcomes": {}}},
             "declared-gate pass left evidence"),
            ("Q1's incumbent leg was not computed (R-397 ruling 1)",
             {**emitted(), "incumbent_legs_evaluated": {
                 "Q4_combined_ev": {"incumbent_counterpart_computed": True}}},
             "incumbent leg was NOT computed"),
            ("the frozen-prereg anchor left NO evidence",
             {**emitted(), "frozen_prereg_anchor": {}},
             "anchor RAN"),
            ("the gate-evaluation pass left NO evidence",
             {**emitted(), "family": {**emitted()["family"],
                                      "gate_evaluation": {}}},
             "gate-evaluation pass RAN"),
            ("a GATE_PARTIALLY_EVALUATED cell still flagged surviving",
             _stale_receipt(emitted()),
             "standing beside the new status")):
        try:
            assert_dry_run_family(_r)
            ok(False, f"F-1(2) the dry seam must REFUSE ({_lbl})")
        except RuntimeError as e:
            ok(_want in str(e),
               f"F-1(2) KNOWN-BAD ({_lbl}): --dry-run REFUSES rather than "
               f"exiting 0 — the harness that proves the wiring could not "
               f"fail when the wiring was cut")

    # The escalation field must SAY when a declared leg was never computed.
    _legs = incumbent_legs_evaluated(receipt())
    ok(_legs["Q1_arrival"]["declared_comparable"] is True
       and _legs["Q1_arrival"]["incumbent_counterpart_computed"] is False
       and "NO COUNTERPART COMPUTED" in _legs["Q1_arrival"]["note"],
       "F-1 Q1 is declared comparable and its incumbent leg is NOT computed; "
       "the artifact SAYS so rather than the guard's scoping hiding it")
    ok(_legs["Q4_combined_ev"]["incumbent_counterpart_computed"] is True
       and _legs["Q2_sign"]["declared_comparable"] is False,
       "F-1 and the field discriminates: Q4's leg reads computed, Q2 reads "
       "not-declared — a field that said the same thing everywhere would "
       "report nothing")


def _selftest_survivor_and_provenance(ok):
    """Q-DA-197 F2/F5/F6, plus the Holm-vs-Bonferroni separation."""
    def fam(spec):
        """A 24-cell family from (status, p) per head, everything else fixed."""
        cells = {}
        for a in I11.ARMS_011:
            for h in I11.HEADS_011:
                st, pv = spec.get(h, ("OK", None))
                for b in I11.BUDGETS_011:
                    cells[I11.cell_key(a, h, b)] = I11.build_cell(
                        a, h, b, statistic=1.0, p_value=pv, status=st,
                        n_actions=10, detail="fixture")
        return I11.assemble_family(cells)

    # THE DEFECT, red-first. A NO_INCUMBENT_COUNTERPART cell with a tiny p
    # was published as a survivor: Holm alone cannot see that the cell's own
    # status says its declared null does not exist. Pre-fix this asserted
    # False and the artifact said True (measured: 12 of 18).
    _f = fam({"Q1_arrival": ("OK", 0.0001),
              "Q2_sign": (I11.CELL_STATUS_NO_COUNTERPART, 0.0001),
              "Q3_magnitudes": (I11.CELL_STATUS_NO_COUNTERPART, 0.0001),
              "Q4_combined_ev": (I11.CELL_STATUS_OK, 0.0001)})
    _nonok = [k for k, c in _f["cells"].items()
              if c["survives_joint_reading_at_0_05"] and c["status"] != "OK"]
    ok(not _nonok,
       f"F2 KNOWN-BAD: a non-OK cell whose Holm p is 0.0024 must NOT be "
       f"published as surviving the joint reading (got {len(_nonok)} such "
       f"cells: {_nonok[:2]})")
    # ...AND THE CONTROL MUST ADMIT, or it is a predicate that only ever
    # refuses (rule 16): the OK cells at the same p still survive.
    _oks = [k for k, c in _f["cells"].items()
            if c["survives_joint_reading_at_0_05"]]
    ok(len(_oks) == 12 and all(_f["cells"][k]["status"] == "OK" for k in _oks),
       f"F2 POSITIVE CONTROL: OK cells at the same p DO survive (12 expected, "
       f"got {len(_oks)}), so the conjunct narrows the claim rather than "
       f"disabling it")
    ok(sorted(_f["surviving_cells"]) == sorted(_oks),
       "F2 the published LIST and the per-cell FLAG agree — two renderings "
       "of one predicate must not be able to disagree")
    ok(len(_f["cells_passing_holm_but_not_OK"]) == 12,
       f"F2 the cells Holm alone would have passed are still REPORTED, so "
       f"narrowing the claim does not hide what changed (got "
       f"{len(_f['cells_passing_holm_but_not_OK'])})")
    # An OK cell that FAILS Holm must not survive either — the other conjunct.
    _g = fam({"Q1_arrival": ("OK", 0.40), "Q2_sign": ("OK", 0.40),
              "Q3_magnitudes": ("OK", 0.40), "Q4_combined_ev": ("OK", 0.40)})
    ok(not _g["surviving_cells"],
       "F2 KNOWN-BAD the other way: OK status alone does not survive; both "
       "conjuncts must hold")

    # M17: the CARRIAGE, not just the computation. `_q3_compose` can return a
    # perfect statistic_n and `build_cell` can still drop it on the floor --
    # my first F1 controls tested only the former and a mutation that ignored
    # statistic_n survived.
    _bc = I11.build_cell("composed_linear", "Q2_sign", "5%", statistic=0.6,
                         p_value=0.01, n_actions=177674, arrival_n=177674,
                         statistic_n=17604, statistic_n_basis="worse side")
    ok(_bc["n_actions"] == 17604 and _bc["arrival_n"] == 177674
       and _bc["statistic_n"] == 17604,
       f"F1 build_cell CARRIES the statistic's own n in n_actions (17604) and "
       f"keeps the arrival population addressable as arrival_n (177674) -- "
       f"got n_actions={_bc['n_actions']}, arrival_n={_bc['arrival_n']}")
    _bn = I11.build_cell("composed_linear", "Q2_sign", "5%", n_actions=177674)
    ok(_bn["n_actions"] == 177674 and _bn["statistic_n"] is None
       and "NOT STATED" in _bn["statistic_n_basis"],
       "F1 KNOWN-BAD: with no statistic_n the cell falls back to the arrival "
       "n and SAYS SO, so a reader cannot mistake the fallback for a measured "
       "statistic population")

    # M19: the Q2 path specifically -- its statistic lives on the PREVENTABLE
    # base, which is the mismatch DA measured (17,604 behind a cell saying
    # 177,674). Driven through `_one_cell`, not through a helper.
    def _q2h(auc, n, p):
        return {"auc": auc, "n_actions": n, "status": "OK",
                "matched_random": {"status": "OK", "p_value": p,
                                   "n_draws": 500}}
    _q2r = {"btc": {"composed_linear": {
        "adjudicated_statistics": {"Q2_sign": 0.58, "Q2_cell_status": "OK"},
        "heads": {"Q2_p_pos": _q2h(0.61, 17604, 0.02),
                  "Q2_p_neg": _q2h(0.58, 17604, 0.30)}}}}
    _q2c = _one_cell("composed_linear", "Q2_sign", "5%", _q2r,
                     {"btc": {"eval_n_actions": 177674}})
    ok(_q2c["statistic_n"] == 17604 and _q2c["arrival_n"] == 177674
       and _q2c["n_actions"] == 17604,
       f"F1 a Q2 cell states the PREVENTABLE base behind its AUC (17604), not "
       f"the arrival population (177674) it was selected from (got "
       f"n_actions={_q2c['n_actions']}, arrival_n={_q2c['arrival_n']})")
    ok("Q2_p_neg" in _q2c["statistic_n_basis"],
       "F1 and it NAMES the worse side the n came from, so the n and the "
       "statistic can be checked to describe the same head")

    # HOLM MUST ACTUALLY BE A STEP-DOWN. At the all-tied floor Holm and a
    # flat Bonferroni are indistinguishable (monotonicity carries the first
    # step across the ties), so the artifact cannot evidence WHICH ran --
    # DA's point. Separated here on UNTIED p, where they must differ.
    _h = fam({"Q1_arrival": ("OK", 0.001), "Q2_sign": ("OK", 0.002),
              "Q3_magnitudes": ("OK", 0.003), "Q4_combined_ev": ("OK", 0.004)})
    _ps = {k: c["p_value"] for k, c in _h["cells"].items()}
    _holm = {k: c["holm_p"] for k, c in _h["cells"].items()}
    _bonf = {k: min(1.0, v * 24) for k, v in _ps.items()}
    _lower = [k for k in _holm if _holm[k] < _bonf[k] - 1e-12]
    ok(_lower,
       f"Holm is a STEP-DOWN, not a flat x24: on untied p at least one cell "
       f"is adjusted by a smaller multiplier than Bonferroni's (got "
       f"{len(_lower)} of 24)")
    ok(all(_holm[k] <= _bonf[k] + 1e-12 for k in _holm),
       "and Holm never EXCEEDS Bonferroni, which is the direction that would "
       "mean the step-down was applied backwards")
    _tied = fam({h: ("OK", 0.002) for h in I11.HEADS_011})
    _th = {k: c["holm_p"] for k, c in _tied["cells"].items()}
    ok(all(abs(_th[k] - min(1.0, 0.002 * 24)) < 1e-12 for k in _th),
       "and at the ALL-TIED floor the two coincide exactly — stated so the "
       "artifact's own numbers are never read as evidence of which procedure "
       "ran (Q-DA-197)")

    # F5/F6: the producing commit and the as-of are COMPUTED, not narrated.
    _pv = producing_code_provenance()
    # A TRUTHY STRING IS NOT A REF. My first version of this control asserted
    # `bool(fit_code_ref)` and PASSED on the function's own
    # "UNAVAILABLE: ..." placeholder -- a control that cannot distinguish the
    # thing from its named absence (rule 16). Caught by mutation M21.
    _ref = _pv.get("fit_code_ref")
    _is_sha = (isinstance(_ref, str) and len(_ref) == 40
               and all(ch in "0123456789abcdef" for ch in _ref))
    ok(_is_sha and _pv.get("fit_code_ref_resolved") is True,
       f"F5 the producing COMMIT is recorded as a real 40-hex sha AND flagged "
       f"resolved -- not left null, and not satisfied by the placeholder this "
       f"function emits when git fails (got {str(_ref)[:44]!r})")
    ok(_pv["fit_code_ref"] != "" and "UNAVAILABLE" not in str(_ref),
       "F5 KNOWN-BAD: the UNAVAILABLE placeholder must not read as a ref; it "
       "exists so an unreadable git is a NAMED absence, never a silent one")
    # And DIRTY must track reality, checked through a DIFFERENT git path than
    # the one the function uses: two readings of one fact must agree.
    import subprocess as _sp

    def _git_rc(*a):
        return _sp.run(("git", *a), cwd=str(Path(__file__).resolve().parent),
                       capture_output=True, text=True, timeout=20)
    _indep = (_git_rc("diff", "--quiet").returncode != 0
              or _git_rc("diff", "--cached", "--quiet").returncode != 0
              or bool(_git_rc("ls-files", "--others", "--exclude-standard")
                      .stdout.strip()))
    ok(isinstance(_pv.get("working_tree_dirty"), bool)
       and _pv["working_tree_dirty"] == _indep,
       f"F5 the DIRTY flag matches an INDEPENDENT read (diff/diff--cached/"
       f"ls-files vs status --porcelain): a ref alone does not identify the "
       f"bytes that ran when the tree has uncommitted edits (reported "
       f"{_pv.get('working_tree_dirty')}, independently {_indep})")
    ok(_pv.get("runner_sha256_prefix") and _pv.get("library_sha256_prefix"),
       "F5 both 011 module content hashes ride along, so the ref is "
       "checkable against bytes rather than trusted")
    _t = run_as_of()
    ok(isinstance(_t, str) and _t.endswith("Z") and _t[4] == "-",
       f"F6 an as-of is produced in ISO-8601 UTC (rule 8: every quoted "
       f"population carries its n AND its as-of); got {_t!r}")


def _selftest_q3_r306(ok):
    """R-306's conjunction + worse side: it must ADJUDICATE and it must REFUSE."""
    def head(slope, pv, status="OK", nsv=0.0, alt="greater", n_draws=500):
        # n_draws travels because the FLOOR disclosure is computed from it:
        # a control whose fixture omits the field the code reads cannot
        # exercise that branch, and would pass by never reaching it.
        return {"status": status, "calibration_slope": slope,
                "matched_random": {"status": status, "p_value": pv,
                                   "no_skill_value": nsv, "alternative": alt,
                                   "n_draws": n_draws}}

    def cell(harm, good, arm="composed_linear"):
        """A compose that RAISES is a NAMED failure, never a traceback.

        Measured: relaxing the conjunction makes `_q3_compose` read a
        sub-head that is not there, and an uncaught KeyError aborts the whole
        suite -- so the mutation looked killed while every control AFTER it
        never ran, which is indistinguishable from them passing."""
        try:
            _t = _q3_compose(arm, {"btc": {arm: {"heads": {
                "Q3_m_harm": harm, "Q3_m_good": good}}}})
            # _q3_compose returns (statistic, p, status, statistic_n, basis,
            # detail); the controls below read the first three and the
            # detail, so it is projected here rather than at every call.
            return (_t[0], _t[1], _t[2], _t[5])
        except Exception as e:                       # noqa: BLE001
            return (f"RAISED:{type(e).__name__}", f"RAISED:{e}",
                    f"RAISED:{type(e).__name__}", f"RAISED: {e}")

    def cell_full(harm, good, arm="composed_linear"):
        """The WHOLE tuple, for the controls that read statistic_n (F1)."""
        return _q3_compose(arm, {"btc": {arm: {"heads": {
            "Q3_m_harm": harm, "Q3_m_good": good}}}})

    # POSITIVE CONTROL, and it must ADMIT: both sides evaluable adjudicates.
    _st, _p, _s, _d = cell(head(0.70, 0.05), head(0.91, 0.44))
    ok(_s == I11.CELL_STATUS_OK and _p == 0.44 and _st == 0.70,
       f"R-306 POSITIVE CONTROL: both sides evaluable -> the cell adjudicates, "
       f"p = worse 0.44, statistic = min slope 0.70 (got {_s}, {_p}, {_st})")

    # THE WORSE SIDE IS COMPUTED, NOT POSITIONAL. Both orderings, because a
    # rule that reads the first sub-head passes one of them and fails the
    # other -- which is exactly how the old `_cell_p` looked correct.
    _, _p1, _, _ = cell(head(0.70, 0.05), head(0.91, 0.44))
    _, _p2, _, _ = cell(head(0.70, 0.44), head(0.91, 0.05))
    ok(_p1 == 0.44 and _p2 == 0.44,
       f"R-306 the worse p is found by VALUE in both orderings (got {_p1}, "
       f"{_p2}); a positional read gives 0.05 in one of them")
    _s1, _, _, _ = cell(head(0.91, 0.05), head(0.70, 0.05))
    ok(_s1 == 0.70,
       f"R-306 the min slope is found by VALUE, not by sub-head position "
       f"(got {_s1})")

    # DISAGREEMENT IS DISCLOSED, NOT RESOLVED: worse-p and min-slope sides
    # need not coincide, and a detail that never says so hides a choice.
    _, _, _, _dd = cell(head(0.70, 0.05), head(0.91, 0.44))
    ok("DISAGREE" in _dd,
       "R-306 when the worse-p side and the min-slope side differ, the cell "
       "DISCLOSES the disagreement rather than preferring one silently")
    _, _, _, _da = cell(head(0.70, 0.44), head(0.91, 0.05))
    ok("AGREE" in _da and "DISAGREE" not in _da and "TIED" not in _da,
       "R-306 and it says AGREE when they coincide, so the disclosure "
       "discriminates instead of always firing")

    # A TIE IS NOT A CONCURRENCE. On the real population BOTH sides sit at
    # the permutation floor, so this is the normal case, not a corner one --
    # and "AGREE" there would be a label that cannot tell two situations
    # apart while reading as evidence.
    _tp, _tpv, _ts, _dt = cell(head(0.70, 0.44), head(0.91, 0.44))
    ok("TIED ON p" in _dt and "AGREE" not in _dt and _tpv == 0.44
       and _ts == I11.CELL_STATUS_OK and _tp == 0.70,
       f"R-306 equal p on both sides reports TIED, never AGREE: the ruling's "
       f"worse-of-the-two cannot discriminate and the p is the same either "
       f"way (got p={_tpv}, stat={_tp})")
    ok("PERMUTATION FLOOR" not in _dt,
       "R-306 and a tie AWAY from the floor does not claim the floor -- the "
       "floor disclosure must discriminate, not decorate")
    _, _, _, _df = cell(head(0.70, 1 / 501), head(0.91, 1 / 501))
    ok("PERMUTATION FLOOR" in _df and "Q3_m_good" in _df and "Q3_m_harm" in _df,
       "R-306 p exactly at 1/(n_draws+1) is NAMED as the permutation floor on "
       "both sides: a resolution-limited null must not read as a measured one")

    # KNOWN-BAD: half a head must never carry the cell (the CONJUNCTION).
    for _lbl, _h, _g, _want in (
            ("m_good absent", head(0.70, 0.05), {}, I11.CELL_STATUS_UNEVALUABLE),
            ("m_harm no slope", {"status": "OK", "matched_random":
                                 {"status": "OK", "p_value": 0.01,
                                  "no_skill_value": 0.0,
                                  "alternative": "greater"}},
             head(0.91, 0.44), I11.CELL_STATUS_UNEVALUABLE),
            ("m_good no p", head(0.70, 0.05),
             {"status": "OK", "calibration_slope": 0.91,
              "matched_random": {"status": "OK", "no_skill_value": 0.0,
                                 "alternative": "greater"}},
             I11.CELL_STATUS_UNEVALUABLE),
            ("m_good underpowered", head(0.70, 0.05),
             head(0.91, 0.44, status=I11.UNDERPOWERED),
             I11.UNDERPOWERED)):
        _st, _p, _s, _ = cell(_h, _g)
        ok(_s == _want and _p is None,
           f"R-306 KNOWN-BAD ({_lbl}): the CONJUNCTION needs both sides, so "
           f"the cell is {_want} and WITHHOLDS its p (got {_s}, p={_p})")

    # KNOWN-BAD: the gate is "excludes 0", so a null declared against any
    # other no-skill value answers another question (R-286). This is the
    # control that keeps the REPORTED deviation-from-1 diagnostic out of the
    # adjudicated slot.
    for _lbl, _bad in (("no_skill_value 1.0", head(0.91, 0.44, nsv=1.0)),
                       ("two-sided", head(0.91, 0.44, alt="two-sided")),
                       ("no_skill_value absent",
                        {"status": "OK", "calibration_slope": 0.91,
                         "matched_random": {"status": "OK", "p_value": 0.44,
                                            "alternative": "greater"}})):
        _st, _p, _s, _d = cell(head(0.70, 0.05), _bad)
        ok(_s == I11.CELL_STATUS_UNEVALUABLE and _p is None
           and "no_skill_value" in _d,
           f"R-306/R-286 KNOWN-BAD ({_lbl}): a null declared against another "
           f"no-skill value is REFUSED under this gate's name (got {_s})")

    # The interval is NOT claimed. Rule 8 forbids one at G=0, and "CI excludes
    # 0" must never be read as an interval that was computed.
    ok("NO literal interval is claimed" in _d.replace("\n", " ")
       or "NO literal interval" in cell(head(0.70, 0.05),
                                        head(0.91, 0.44))[3],
       "R-306 the cell states that no literal interval is claimed (rule 8 "
       "forbids one at G=0 complete UTC days)")

    # F1: the cell's n is the BINDING SIDE's, not the arrival population.
    # This is the defect DA measured: Q2 adjudicates on 17,604 while every
    # cell stated 177,674, and one field cannot answer both questions.
    def nhead(slope, pv, n):
        return dict(head(slope, pv), n_actions=n)
    _f = cell_full(nhead(0.70, 0.05, 7988), nhead(0.91, 0.44, 9617))
    ok(_f[3] == 7988 and "Q3_m_harm" in _f[4] and "9617" in _f[4],
       f"F1 Q3 carries the BINDING side's n (7988 on Q3_m_harm), and names "
       f"both sides so the pair stays visible (got n={_f[3]}, "
       f"basis={str(_f[4])[:70]!r})")
    _f2 = cell_full(nhead(0.91, 0.05, 7988), nhead(0.70, 0.44, 9617))
    ok(_f2[3] == 9617,
       f"F1 KNOWN-BAD: swap which side binds and the n FOLLOWS the statistic "
       f"(9617, not 7988) -- a positional read would return the same n twice "
       f"(got {_f2[3]})")
    ok(cell_full(head(0.70, 0.05), head(0.91, 0.44))[3] is None,
       "F1 a head with no n_actions yields statistic_n None rather than a "
       "fabricated count; build_cell then says so in statistic_n_basis")

    # KNOWN-BAD at the boundary of its own competence: the coin axis is a
    # SEPARATE R-306 clause this runner does not implement, so two coins must
    # refuse here rather than be collapsed by a rule nobody wrote.
    try:
        _q3_compose("composed_linear", {
            "btc": {"composed_linear": {"heads": {}}},
            "eth": {"composed_linear": {"heads": {}}}})
        ok(False, "R-306 _q3_compose must REFUSE two coins")
    except RuntimeError as e:
        ok("adjudicates ONE coin" in str(e),
           "R-306 KNOWN-BAD: _q3_compose REFUSES a multi-coin call; the "
           "coin clause is not implemented and must not be improvised")

    # `_cell_p` is the multi-coin path's reader and must agree with the rule.
    _pc = {"btc": {"composed_linear": {"heads": {
        "Q3_m_harm": head(0.70, 0.05), "Q3_m_good": head(0.91, 0.44)}}}}
    ok(_cell_p(_pc, "composed_linear", "Q3_magnitudes") == [0.44],
       "R-306 _cell_p reads Q3's WORSE p (0.44), not the first sub-head's")
    _pc2 = {"btc": {"composed_linear": {"heads": {
        "Q3_m_harm": head(0.70, 0.05)}}}}
    ok(_cell_p(_pc2, "composed_linear", "Q3_magnitudes") == [],
       "R-306 KNOWN-BAD: with one side missing `_cell_p` yields NO p at all, "
       "so half a head cannot carry the multi-coin cell either")


def _selftest_incumbent_wiring(ok):
    """The Q4 increment path: it must be CALLED, and it must survive PACKING."""
    import numpy as _np
    _w = {"PM": 4, "FN": 2, "ST": 3}
    _rows = [{"slug": "w0", "side": "BUY_UP", "gen": i, "t_start": float(i),
              "t0": 1787650200.0} for i in range(5)]
    _PM = [[0.1 * (i + 1) + k for k in range(4)] for i in range(5)]
    _FN = [[0.5 + i, 1.5 + i] for i in range(5)]
    _ST = [[9.0 + i, 8.0 + i, 7.0 + i] for i in range(5)]
    _model = {"norm_mu": [0.2] * 6, "norm_sd": [1.3] * 6,
              "hazard_weights": [0.1] * 7, "value_weights": [0.3] * 7,
              "_verified": {"artifact": "fixture"}}

    def blk():
        return {"kept": list(_rows), "PM": [list(r) for r in _PM],
                "FN": [list(r) for r in _FN], "ST": [list(r) for r in _ST]}

    # compact_design must RECORD the widths packing destroys.
    _b = compact_design(blk())
    ok(_b.get("w") == _w and _b["X"].shape == (5, 9),
       f"compact_design records the family widths the pack destroys "
       f"(got {_b.get('w')})")

    def scored(fn, block, key):
        """A read that RAISES is a NAMED failure, never a traceback.

        This is the exact pre-fix behaviour -- `block["PM"][j]` on a packed
        block raises TypeError -- so without this the mutation that restores
        the defect takes the suite down and every control after it goes
        unrun, which is indistinguishable from them passing."""
        try:
            return fn(_model, block, range(5))[key]
        except Exception as e:                       # noqa: BLE001
            return f"RAISED:{type(e).__name__}: {e}"

    # THE DEFECT, red-first: pre-fix `apply_incumbent` read block["PM"], which
    # packing sets to None -- TypeError on every real block, invisible to a
    # suite whose fixtures are all unpacked.
    _un = scored(apply_incumbent, blk(), "expected_cancel_value")
    _pk = scored(apply_incumbent, _b, "expected_cancel_value")
    ok(isinstance(_pk, list) and _pk == _un and len(_pk) == 5,
       f"apply_incumbent gives BIT-IDENTICAL values on a PACKED and an "
       f"unpacked block (packed {str(_pk)[:60]} vs unpacked {str(_un)[:60]})")
    _h1 = scored(apply_incumbent_hazard, blk(), "p_fill")
    _h2 = scored(apply_incumbent_hazard, _b, "p_fill")
    ok(isinstance(_h2, list) and _h1 == _h2,
       f"apply_incumbent_hazard too: packed == unpacked "
       f"(packed {str(_h2)[:60]})")

    # ARM D's DEFINING PROPERTY survives the slice: state features never enter.
    _bs = blk(); _bs["ST"] = [[99.0, 98.0, 97.0] for _ in range(5)]
    _p99 = scored(apply_incumbent, compact_design(_bs), "expected_cancel_value")
    ok(isinstance(_p99, list) and _p99 == _pk,
       "the PM+FN prefix is the right slice: moving every STATE feature to "
       "99.0 moves NOTHING, which is what makes this the incumbent arm")

    # KNOWN-BAD: a packed block with no widths must REFUSE, never guess.
    _nw = compact_design(blk()); _nw.pop("w", None)
    try:
        apply_incumbent(_model, _nw, range(5))
        ok(False, "a packed block with no widths must REFUSE")
    except RuntimeError as e:
        ok("no family widths" in str(e),
           "KNOWN-BAD: a PACKED block without recorded widths REFUSES rather "
           "than slicing to the model's own length, which would make a shape "
           "disagreement read as a correct row")

    # KNOWN-BAD: recorded widths that disagree with the model REFUSE.
    _bw = compact_design(blk()); _bw["w"] = {"PM": 3, "FN": 2, "ST": 4}
    try:
        apply_incumbent(_model, _bw, range(5))
        ok(False, "disagreeing widths must REFUSE")
    except RuntimeError as e:
        ok("disagree about what a row is" in str(e),
           "KNOWN-BAD: a PM+fine width that disagrees with the incumbent's "
           "REFUSES (the slice would still return the right COUNT of numbers)")

    # SOURCE GUARD, because the defect is a call that is NOT THERE and no unit
    # test can see a missing line. Anchored on the definition, not on the
    # substring -- `.index("def main(")` matches this guard's own literal.
    _src = Path(__file__).read_text(encoding="utf-8")
    _mn = _src[_src.index("\ndef main() -> int:"):]
    ok("load_verified_incumbent(" in _mn and "apply_incumbent(" in _mn,
       "R-280 main() actually CALLS the load-verify-apply path (it was built "
       "red-first with ELEVEN falsifiers and ZERO call sites, so every Q4 "
       "cell reported NO_INCUMBENT_COUNTERPART -- defect I11-2's shape)")
    ok("incumbent=inc_pred" in _mn,
       "R-280 main() PASSES the applied incumbent to q4_economics; without "
       "the kwarg the call is legal, silent, and unpaired")
    ok("q4_economics(ap, EVAL[coin][\"kept\"])" not in _mn,
       "KNOWN-BAD: the UNPAIRED call shape is gone from main(); it is legal "
       "Python and produced twelve p-less cells")

    # The dry harness's own instrument: a synthetic incumbent must pass the
    # REAL loader (positive control), so the harness exercises the wiring
    # rather than bypassing it.
    _d = _dry_incumbent_fitdir("btc", 7)
    _m = load_verified_incumbent("btc", fitdir=_d)
    ok(_m.get("arm") == INCUMBENT_ARM and len(_m["norm_mu"]) == 7,
       "the dry harness's synthetic incumbent loads through the REAL "
       "verified loader; a harness that bypassed it would leave the hole it "
       "exists to close")
    (_d / "linear_d_btc.json").write_text(
        (_d / "linear_d_btc.json").read_text().replace("\"arm\"", "\"Arm\"", 1))
    try:
        load_verified_incumbent("btc", fitdir=_d)
        ok(False, "a tampered synthetic incumbent must be refused")
    except RuntimeError as e:
        ok("not the committed incumbent" in str(e),
           "KNOWN-BAD: tampering with the synthetic artifact is caught by the "
           "same hash binding, so the dry control can fail as well as pass")


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
def _selftest_coin_slice(ok):
    """Controls for the memory slicing. It must SELECT, and it must REFUSE."""
    # REGRESSION GUARD, red-first: the tape index must NEVER be coin-sliced.
    # `_feature_pass` builds BOTH coins from it and refuses when a coin joins
    # zero rows -- measured, slicing made eth fail 520,033 of 520,033 and the
    # absorption bound REFUSED ("Drops absorb row-level anomalies, never total
    # input failures"). The saving must come from what we CARRY downstream,
    # never from what we index. A source check, because the failure is a
    # DELETED line and no unit test can see one that is not there.
    _src_r = Path(__file__).read_text(encoding="utf-8")
    # ANCHOR ON THE DEFINITION, not on the substring: `.index("def main(")`
    # matched THIS GUARD'S OWN string literal (it appears above the real
    # function), so the slice began inside the selftest and the check read
    # itself. A self-matching guard is the "expected value coincided with
    # the mutant's output" defect wearing different clothes.
    _m_r = _src_r[_src_r.index("\ndef main() -> int:"):]
    ok('PA.tape_index("train")' in _m_r and 'PA.tape_index("score")' in _m_r,
       "(memory) both tape indexes are built UNSLICED")
    ok("_coin_slice(PA.tape_index" not in _m_r,
       "(memory KNOWN-BAD) the tape index is never coin-sliced -- doing so "
       "starves the other coin's feature join and the absorption bound refuses")

    # ---- the DECLARED OUTPUT must follow the MODE ----
    ok(declared_outputs_for(["p"]) == DECLARED_OUTPUTS,
       "(output) an unsliced run declares the full artifact")
    _dbtc = declared_outputs_for(["p", "--coin", "btc"])
    ok(len(_dbtc) == 1 and _dbtc[0].name.endswith("__coin_btc.json"),
       f"(output) a --coin run declares the SLICED artifact "
       f"(got {_dbtc[0].name})")
    ok(_dbtc[0] != DECLARED_OUTPUTS[0],
       "(output KNOWN-BAD) it does NOT declare the full artifact -- demanding "
       "a file the mode never writes made a COMPLETED run refuse at its exit "
       "after producing 96 KB of results")

    # ---- the valuation gate must be RESTORED, and by the canonical rule ----
    import harmful_exposure_rows as _HERt
    _lat_fill = {"5": {"preventable_shares": 2.0, "preventable_value_cents": 1.0}}
    _blk_g = {"kept": [{"latency": _lat_fill}, {"latency": {}}, {"latency": None},
                       {"latency": _lat_fill, "any_fill_ahead": False}]}
    restore_valuation_gate(_blk_g)
    _k = _blk_g["kept"]
    ok(_k[0]["any_fill_ahead"] is True,
       "(gate) a row with preventable shares gets any_fill_ahead=True")
    ok(_k[1]["any_fill_ahead"] is False and _k[2]["any_fill_ahead"] is False,
       "(gate) empty and absent latency both give False, not a crash")
    ok(_k[3]["any_fill_ahead"] is False,
       "(gate CONTROL) an EXISTING gate is left alone -- restoration fills a "
       "missing field, it never overwrites one that is present")
    ok(all(isinstance(r["any_fill_ahead"], bool) for r in _k),
       "(gate) every restored value is a real bool -- validate_row refuses "
       "NON_BOOLEAN_GATE, so a truthy int would fail one layer down")
    # It must be THE canonical function, not a local copy: two definitions of a
    # valuation gate is one too many, and the source says so in those words.
    _srcg = Path(__file__).read_text(encoding="utf-8")
    _fn = _srcg[_srcg.index("def restore_valuation_gate("):]
    _fn = _fn[:_fn.index("\ndef ")]
    ok("_HER.any_fill_ahead(" in _fn and "preventable_shares" not in _fn,
       "(gate) restoration CALLS harmful_exposure_rows.any_fill_ahead and does "
       "not reimplement its predicate locally")
    # And validate_row must actually accept a restored row end to end.
    _vr = I11.validate_row({"any_fill_ahead": False, "latency": None},
                           D.TARGET_LATENCY_MS)
    ok(isinstance(_vr, dict),
       "(gate) a restored no-fill row passes validate_row rather than "
       "refusing MISSING_GATE, which is the failure this closes")

    # ---- compact_design must be EXACTLY equivalent, and must free ----
    import random as _rnd
    _rnd.seed(11)
    _nr = 40
    _blk = {"kept": [{"i": i} for i in range(_nr)],
            "PM": [[_rnd.uniform(-1e6, 1e6) for _ in range(3)] for _ in range(_nr)],
            "FN": [[_rnd.uniform(-1, 1) for _ in range(2)] for _ in range(_nr)],
            "ST": [[_rnd.uniform(0, 1e-9) for _ in range(4)] for _ in range(_nr)]}
    _before = [build_design(_blk, i) for i in range(_nr)]
    compact_design(_blk)
    _after = [build_design(_blk, i) for i in range(_nr)]
    ok(_before == _after,
       "(memory) compact_design is EXACTLY equivalent -- bit-for-bit, across "
       "large, tiny and negative magnitudes. A Python float IS a float64, so "
       "packing changes the container and never the number")
    ok(all(isinstance(r, list) for r in _after),
       "(memory) build_design still returns a LIST after packing, so every "
       "downstream consumer sees the type it always saw")
    ok(_blk["PM"] is None and _blk["FN"] is None and _blk["ST"] is None,
       "(memory) the Python lists are RELEASED -- packing that keeps both "
       "representations alive saves nothing, which is the whole point")
    ok(_blk["X"].dtype.name == "float64" and _blk["X"].shape == (_nr, 9),
       f"(memory) the packed array is float64 and correctly shaped "
       f"(got {_blk['X'].dtype.name}, {_blk['X'].shape})")
    _empty = {"kept": [], "PM": [], "FN": [], "ST": []}
    compact_design(_empty)
    ok(_empty.get("X") is None,
       "(memory CONTROL) an empty block is left alone rather than packed into "
       "a zero-row array that would then read as compacted")

    _blocks = {"btc": {"kept": [1]}, "eth": {"kept": [2]}}
    _d = _coin_drop(dict(_blocks), "btc", "t")
    ok(list(_d) == ["btc"],
       "(drop) the non-selected coin's block is released after the pass")
    _r3 = ""
    try:
        _coin_drop({"eth": {}}, "btc", "t")
    except RuntimeError as e:
        _r3 = str(e)
    ok("absent from" in _r3,
       "(drop KNOWN-BAD) a coin the pass never produced refuses -- a missing "
       "coin is not an empty one")
    # The probe must be freed BEFORE the topup pass, which is where both OOMs
    # landed. Source-level, because the ordering is the whole fix.
    _src = Path(__file__).read_text(encoding="utf-8")
    _m = _src[_src.index("\ndef main() -> int:"):]   # same anchoring fix
    ok(_m.index("del probe") < _m.index('PA._feature_pass(PA.TOPUP'),
       "(memory) the embargo probe is released BEFORE the topup pass, not "
       "after it -- holding ~640k dicts across the pass put them inside the "
       "measured peak for no reason")


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


def _readjudicate_arg(argv=None):
    """The artifact `--readjudicate` names, or None. Refuses a bare flag."""
    a = sys.argv if argv is None else argv
    if "--readjudicate" not in a:
        return None
    i = a.index("--readjudicate")
    if i + 1 >= len(a) or a[i + 1].startswith("-"):
        raise SystemExit("REFUSED: --readjudicate needs the artifact to "
                         "re-adjudicate; a flag with no value must not read "
                         "as 'the declared output'.")
    return Path(a[i + 1])


def readjudicated_path(src: Path) -> Path:
    """Where a re-adjudication is written. NEVER over its source (rule 13)."""
    return src.with_name(f"{src.stem}__readjudicated_v2{src.suffix}")


def readjudicate(src: Path) -> tuple:
    """Re-adjudicate a PRESERVED family from its own evidence. No refit.

    `evaluate_family` is a pure function of `results` and `populations`, and
    every 011 artifact carries both: per-head statistics, per-head
    matched-random nulls, per-budget economics. So a RULING that changes only
    how cells COMPOSE their heads can be applied to a completed run exactly,
    without refitting anything -- which is the point, because refitting to
    apply a composition rule would change the numbers being ruled on.

    What it can and cannot reach is a real boundary, not a caveat: a rule that
    needs evidence the run never computed (Q4's increment needs the
    incumbent's own composed value on those rows) CANNOT be recovered here,
    and the cell will keep saying so. Re-adjudication moves adjudication, not
    measurement.

    Rule 13: the source is never edited. The result is a SUPERSEDING artifact
    naming its source by hash, and the source stays as provenance."""
    import hashlib
    if not src.exists():
        raise RuntimeError(f"REFUSED: no artifact at {src}.")
    d = json.loads(src.read_text())
    if d.get("artifact") != "iter011_conditional_value_v1":
        raise RuntimeError(
            f"REFUSED: {src.name} identifies as {d.get('artifact')!r}, not an "
            f"011 conditional-value artifact. Re-adjudicating a file of "
            f"another shape would compose cells out of fields that mean "
            f"something else.")
    res, pops = d.get("results") or {}, d.get("populations") or {}
    # R-289, THE CHECKER'S CHAIR: assert the parse actually READ the
    # population AND the fields it composes on. A re-adjudication over an
    # empty results block would rebuild 24 cells of UNEVALUABLE and exit 0,
    # which is a vacuum wearing the shape of a completed correction.
    n_arms = sum(len([a for a in r if a in I11.ARMS_011]) for r in res.values())
    n_heads = sum(len((r[a].get("heads") or {})) for r in res.values()
                  for a in r if a in I11.ARMS_011)
    if not res or not pops or n_arms == 0 or n_heads == 0:
        raise RuntimeError(
            f"REFUSED: {src.name} yielded {len(res)} coins, {n_arms} arms and "
            f"{n_heads} head reports. A re-adjudication that read nothing "
            f"would still build 24 cells and exit 0 -- 'found nothing' from a "
            f"reader that touched nothing is the empty-set trap (R-289).")
    before = ((d.get("family") or {}).get("cells")) or {}
    fam = evaluate_family(res, pops)   # for the CHANGED-CELL diff below
    changed = {}
    for k, c in fam["cells"].items():
        b = before.get(k) or {}
        was = (b.get("status"), b.get("p_value"), b.get("statistic"))
        now = (c["status"], c["p_value"], c["statistic"])
        if was != now:
            changed[k] = {"was": {"status": was[0], "p_value": was[1],
                                  "statistic": was[2]},
                          "now": {"status": now[0], "p_value": now[1],
                                  "statistic": now[2]}}
    out = dict(d)
    out["as_of"] = run_as_of()
    out["as_of_names"] = ("the RE-ADJUDICATION instant. No population was "
                          "read: the source artifact's own as_of names the "
                          "read this evidence came from, and is preserved in "
                          "`supersedes.source`.")
    out["producing_code"] = producing_code_provenance()
    # A RE-ADJUDICATION MUST CARRY THE SAME GUARANTEES AS AN EMISSION.
    # Otherwise this mode becomes the way a receipt reaches a reader without
    # the checks the main path enforces -- a second door into the same room.
    # Same ORDERED sequence, so a re-adjudicated cell cannot keep a survivor
    # flag its final status contradicts.
    finalise_family(out)
    fam = out["family"]
    out["incumbent_applicability_guard"] = \
        assert_incumbent_applicability_honoured(out)
    out["supersedes"] = {
        "source": str(src), "source_sha256_prefix":
            hashlib.sha256(src.read_bytes()).hexdigest()[:16],
        "source_untouched": True,
        "mode": "READJUDICATION ONLY -- no refit, no rescore",
        "what_moved": "cell COMPOSITION under a ruling that already existed; "
                      "every head statistic, every matched-random null and "
                      "every economics block is carried over BYTE-FOR-BYTE "
                      "from the source",
        "reason": "R-306 (USER, 2026-08-29, frozen A1.4) rules Q3's two slope "
                  "gates as CONJUNCTION + WORSE SIDE; the code had never "
                  "implemented it and withheld Q3's p as AGGREGATION_"
                  "UNDECLARED. Rule 13: corrections supersede in-band.",
        "n_cells_changed": len(changed), "cells_changed": changed,
        "read_evidence": {"coins": sorted(res), "arm_reports": n_arms,
                          "head_reports": n_heads,
                          "why": "recorded so this correction cannot be a "
                                 "vacuum that rebuilt cells from nothing"}}
    out["readjudication_limits"] = {
        "cannot_recover": "any cell needing evidence the source run never "
                          "computed. Q4's increment needs the INCUMBENT's own "
                          "composed value on the identical rows; the source "
                          "ran unpaired, so those cells stay "
                          "NO_INCUMBENT_COUNTERPART here and are closed only "
                          "by a re-run.",
        "development_evidence_only": True}
    assert_receipt_has_all_cells(out)
    dst = readjudicated_path(src)
    with dst.open("w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
        fh.flush(); os.fsync(fh.fileno())
    return out, dst


# ---------------------------------------------------------------------------
# F-1: THE ARTIFACT-LEVEL GUARD. A source guard cannot see a value that does
# not flow.
# ---------------------------------------------------------------------------
# The reviewer's MUTANT A defeated the previous protection with ONE line
# inserted immediately before the consumer:
#
#     inc_pred = None           # every guarded string left intact
#     rep["economics"] = q4_economics(ap, EVAL[coin]["kept"], incumbent=inc_pred)
#
# selftest GREEN, --dry-run exit 0, the "[btc/incumbent] ... applied to 400
# rows" line still printed (it is emitted BEFORE the unwiring), and the emitted
# artifact reproduced the exact Q-DA-197 contradiction: comparable true, six
# NO_INCUMBENT_COUNTERPART cells, and a q4_incumbent block still asserting the
# incumbent was loaded and applied.
#
# Source-text assertions prove a LINE EXISTS. This proves the RESULT ARRIVED,
# read off the emitted artifact, which is the only thing a reader has.


def _frozen_prereg_section3(commit: str = None, doc: str = None) -> dict:
    """The frozen section-3 gate table, READ FROM GIT at `commit`. RR3-1.

    Not from the working tree: the whole point is to terminate at bytes the
    USER froze. A working-tree read would move with the tree and prove
    nothing, which is the regress this closes.

    THE "same" ROW IS AN INHERITANCE AND IT IS EXPLICIT. Q2's gate reads
    "same, on the fill-conditional population" -- it inherits Q1's gate, so a
    parser that read it literally would find no incumbent term in Q2 and
    report a mismatch against a correct constant. The rule is applied by name
    and controlled, not assumed.
    """
    import subprocess
    commit = commit or PREREG_COMMIT
    doc = doc or PREREG_DOC
    path = f"live/pm_research/plans/{doc}"
    try:
        r = subprocess.run(("git", "show", f"{commit}:{path}"),
                           cwd=str(Path(__file__).resolve().parents[2]),
                           capture_output=True, text=True, timeout=30)
    except Exception as e:                           # noqa: BLE001
        raise RuntimeError(
            f"REFUSED: could not read the frozen preregistration "
            f"({type(e).__name__}: {e}). A premise check that cannot read its "
            f"premise must REFUSE, never pass quietly.")
    if r.returncode != 0:
        raise RuntimeError(
            f"REFUSED: `git show {commit}:{path}` failed "
            f"({r.stderr.strip()[:160]}). The artifact NAMES this commit as "
            f"its preregistration; if it cannot be read the chain does not "
            f"terminate anywhere.")
    return _parse_section3(r.stdout, f"{doc}@{commit}")


def _parse_section3(body: str, origin: str = "<body>") -> dict:
    """The section-3 gate table, from raw text. Split out so the PARSE RULES
    are drivable: with the real document every rule holds, so a mutant that
    deleted one changed nothing and survived."""
    try:
        sec = body[body.index("\n## 3."):]
        sec = sec[:sec.index("\n## 4.")]
    except ValueError:
        raise RuntimeError(
            f"REFUSED: no section 3 found in {origin}. The gate table is the "
            f"premise; its absence is a refusal, not an empty result.")
    head_of = {"Q1": "Q1_arrival", "Q2": "Q2_sign",
               "Q3": "Q3_magnitudes", "Q4": "Q4_combined_ev"}
    gates, prev = {}, None
    for line in sec.splitlines():
        if not line.startswith("|"):
            continue
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) < 5 or cols[0] not in head_of:
            continue
        gate = cols[-1]
        # THE INHERITANCE, applied by name.
        if gate.lower().startswith("same"):
            if prev is None:
                raise RuntimeError(
                    "REFUSED: a gate row says 'same' with no preceding row to "
                    "inherit from.")
            resolved = f"{prev} [inherited via 'same'] ({gate})"
            carries = "incumbent" in prev.lower()
        else:
            resolved = gate
            carries = "incumbent" in gate.lower()
            prev = gate
        gates[head_of[cols[0]]] = {"gate_text": resolved,
                                   "gate_text_raw": gate,
                                   "carries_incumbent_term": carries,
                                   "inherited": gate is not resolved}
    if len(gates) != 4:
        raise RuntimeError(
            f"REFUSED: parsed {len(gates)} gate rows from section 3 of "
            f"{origin}, expected 4 ({sorted(gates)}). A partial parse would "
            f"anchor the constants to a fragment.")
    return gates


def assert_gate_text_matches(doc_text: dict, code_text: dict) -> dict:
    """The published gate STRING must be the frozen document's, verbatim.

    `assert_gate_terms_match` compares the derived boolean, and a mutation
    that rewrote only the TEXT survived it — the artifact would then publish,
    per cell, a gate the frozen preregistration does not contain. Whitespace
    is normalised because a table cell wraps; nothing else is."""
    def norm(t):
        return " ".join(str(t or "").split())
    drift = {h: {"document": norm(doc_text[h]), "code": norm(code_text.get(h))}
             for h in doc_text if norm(doc_text[h]) != norm(code_text.get(h))}
    if drift:
        raise RuntimeError(
            f"REFUSED: DECLARED_GATES publishes gate TEXT that is not the "
            f"frozen section-3 wording: {drift}. A cell's declared_gate is "
            f"what a reader resolves against the preregistration; a "
            f"paraphrase there is a claim the document does not make.")
    return {"heads_checked": sorted(doc_text)}


def assert_gate_terms_match(doc_terms: dict, code_terms: dict) -> dict:
    """`DECLARED_GATES` must transcribe the frozen document. RR3-1.

    Explicit arguments for the same reason as the harmonisation guard: on the
    real document the two already agree, so a mutant deleting this check
    changed nothing and survived a suite that only ever drove it with
    matching inputs."""
    drift = {h: (doc_terms[h], code_terms.get(h)) for h in doc_terms
             if doc_terms[h] != code_terms.get(h)}
    if drift:
        raise RuntimeError(
            f"REFUSED: DECLARED_GATES disagrees with the FROZEN section-3 "
            f"gate table (head: document, code): {drift}. The document is "
            f"the premise; the constant transcribes it.")
    return {"heads_checked": sorted(doc_terms)}


def assert_propositions_not_harmonised(doc_terms: dict, comparable: dict) -> list:
    """The two maps must DISAGREE somewhere, or they have been merged.

    Split out and given explicit arguments so it is REACHABLE: inside
    `assert_constants_match_frozen_prereg` the artifact anchor fires first on
    every input that would harmonise them, so this branch could never be
    driven and would have been a check that cannot fail (rule 16). It is
    cheap insurance for the day the anchors change, and now it is testable."""
    divergent = sorted(h for h in doc_terms
                       if doc_terms[h] != bool(comparable.get(h)))
    if not divergent:
        raise RuntimeError(
            "REFUSED: the gate-term map and the comparability map agree on "
            "every head, which means the two propositions have been "
            "HARMONISED. Q2 must differ -- its gate names an incumbent and "
            "the incumbent has no sign head -- and that difference is what "
            "RR2-1 rests on.")
    return divergent


def assert_constants_match_frozen_prereg(fitdir=None) -> dict:
    """RR3-1. Terminate the coverage floor's premise at UNFORGEABLE things.

    `INCUMBENT_COMPARABLE` was described as "transcribing the frozen prereg 3
    gates" and the transcription was never checked, so editing the CONSTANT
    and the receipt together was admitted with `coverage_is_complete: true`.

    IT DOES NOT ANCHOR BOTH CONSTANTS TO ONE SOURCE, because they encode
    DIFFERENT PROPOSITIONS and the reviewer is explicit that the difference is
    load-bearing for RR2-1. They terminate in two different places, and
    neither is a line a seat can edit:

      DECLARED_GATES.carries_incumbent_term
          <- the FROZEN DOCUMENT at the recorded commit: does this head's
             gate NAME an incumbent?
      INCUMBENT_COMPARABLE
          <- the HASH-VERIFIED incumbent ARTIFACT: does the incumbent
             actually HAVE this head?

    Q2 is exactly why they cannot be merged: its gate names an incumbent
    (`true` from the document) while the incumbent has no sign head
    (`false` from the artifact). Harmonising them would delete that, so this
    function REQUIRES the disagreement to persist rather than tolerating it.

    The tie between them is the one implication the pair does support:
    a counterpart can only be comparable for a head whose gate names one."""
    gates = _frozen_prereg_section3()
    doc_terms = {h: g["carries_incumbent_term"] for h, g in gates.items()}
    code_terms = {h: bool((I11.DECLARED_GATES.get(h) or {})
                          .get("carries_incumbent_term"))
                  for h in gates}
    assert_gate_terms_match(doc_terms, code_terms)
    # AND THE TEXT, not only the boolean. Measured: rewriting Q3's gate
    # string to "beats the incumbent" left the boolean untouched and passed
    # every check, while the artifact then PUBLISHED a gate the frozen
    # document does not contain. The text is what a reader resolves.
    _text_ev = assert_gate_text_matches(
        {h: g["gate_text_raw"] for h, g in gates.items()},
        {h: (I11.DECLARED_GATES.get(h) or {}).get("gate") for h in gates})

    # WHICH HEADS THE INCUMBENT ACTUALLY HAS, read from the artifact whose
    # identity `load_verified_incumbent` binds by hash. This is a property of
    # the fitted model, not of the prereg, which is why it cannot come from
    # the document.
    model = load_verified_incumbent("btc", fitdir=fitdir)
    has = {"Q1_arrival": bool(model.get("hazard_weights")),
           "Q2_sign": bool(model.get("sign_weights")),
           "Q3_magnitudes": bool(model.get("magnitude_weights")),
           "Q4_combined_ev": bool(model.get("hazard_weights")
                                  and model.get("value_weights"))}
    cmp_drift = {h: (has[h], bool(INCUMBENT_COMPARABLE.get(h)))
                 for h in has if has[h] != bool(INCUMBENT_COMPARABLE.get(h))}
    if cmp_drift:
        raise RuntimeError(
            f"REFUSED: INCUMBENT_COMPARABLE disagrees with the HASH-VERIFIED "
            f"incumbent artifact (head: artifact, constant): {cmp_drift}. "
            f"Editing the constant and the receipt together used to be "
            f"admitted with coverage_is_complete true; the expected set now "
            f"terminates at the model's own weights (RR3-1).")

    # The one implication the pair supports, and the disagreement it must keep.
    bad_imp = sorted(h for h in gates
                     if INCUMBENT_COMPARABLE.get(h) and not doc_terms[h])
    if bad_imp:
        raise RuntimeError(
            f"REFUSED: {bad_imp} are declared comparable while the frozen "
            f"gate names no incumbent term. A counterpart cannot be "
            f"comparable for a gate that does not ask for one.")
    divergent = assert_propositions_not_harmonised(doc_terms,
                                                   dict(INCUMBENT_COMPARABLE))
    return {"preregistration": PREREG_DOC, "preregistration_commit": PREREG_COMMIT,
            # RECORDED, so removing the call removes the evidence. On the
            # real document the texts already match, so a mutant deleting the
            # check changed no behaviour and survived; the seam below asks
            # for this field instead of grepping for the call.
            "gate_text_verified_heads": _text_ev["heads_checked"],
            "source_of_DECLARED_GATES": "the frozen section-3 gate table, read "
                                        "from git at the recorded commit",
            "source_of_INCUMBENT_COMPARABLE": "the hash-verified incumbent "
                                              "artifact's own weight blocks",
            "gate_carries_incumbent_term": doc_terms,
            "incumbent_has_head": has,
            "heads_where_they_DIVERGE": divergent,
            "why_divergence_is_required": "the two encode different "
                                          "propositions (gate names one vs "
                                          "counterpart exists); harmonising "
                                          "them would delete what RR2-1 "
                                          "rests on",
            "chain_terminates_at": "a USER-frozen document and a hash-bound "
                                   "artifact, not at an editable constant"}


def _gate_conjuncts(cell: dict, receipt: dict, legs: dict) -> dict:
    """This cell's declared conjuncts, for the status pass. RR4-1.

    Shared with `attach_declared_gate_outcomes` so ONE definition of "which
    conjuncts does this head declare" exists; two would be two rules."""
    head, arm = cell.get("head"), cell.get("arm")
    holm, p = cell.get("holm_p"), cell.get("p_value")
    null_ok = None if (holm is None or p is None) else bool(holm < 0.05)
    q1 = {}
    for arms in (receipt.get("results") or {}).values():
        for a, rep in arms.items():
            if rep.get("q1_incremental"):
                q1[a] = rep["q1_incremental"]
    if head == "Q1_arrival":
        return {"matched_random": null_ok,
                "incumbent_hazard": (q1.get(arm) or {}).get(
                    "beats_incumbent_hazard_head")}
    if head == "Q2_sign":
        return {"matched_random": null_ok, "incumbent": None}
    if head == "Q3_magnitudes":
        return {"slope_excludes_zero_m_harm": null_ok,
                "slope_excludes_zero_m_good": null_ok}
    return {"increment_beats_incumbent": null_ok, "matched_random": None}


def attach_declared_gate_outcomes(receipt: dict) -> dict:
    """Each head's OWN gate, evaluated separately from the joint reading.

    R-397 RULING 2. `NO_INCUMBENT_COUNTERPART` was blocking Q3, whose frozen
    gate — "calibration slope CI excludes 0 for each, reported separately" —
    carries NO incumbent term at all. A status invented for heads the
    incumbent cannot answer was deciding a head that never asked it anything.

    The two questions are now answered in two fields, because they are two
    questions and one flag cannot carry both:

      declared_gate_outcome           did THIS head pass ITS OWN frozen gate?
      survives_joint_reading_at_0_05  did this cell survive Holm over 24?

    Conjuncts are evaluated per head from what the run actually computed. A
    conjunct nobody computed reads `null`, never `false`: "not evaluated" and
    "evaluated and failed" are different findings, and collapsing them is how
    Q1 came to publish a survival on half a gate."""
    fam = receipt.get("family") or {}
    cells = fam.get("cells") or {}
    results = receipt.get("results") or {}
    if not cells:
        raise RuntimeError(
            "REFUSED: the declared-gate pass read ZERO cells; a pass that "
            "touched nothing must not report an outcome (R-289).")
    # per-arm Q1 incremental evidence, READ from the result
    q1inc = {}
    for arms in results.values():
        for arm, rep in arms.items():
            if rep.get("q1_incremental"):
                q1inc[arm] = rep["q1_incremental"]
    counts = {"passed": 0, "failed": 0, "not_evaluable": 0}
    for key, c in cells.items():
        head, arm = c.get("head"), c.get("arm")
        gate = c.get("declared_gate") or {}
        holm, p = c.get("holm_p"), c.get("p_value")
        # conjunct 1, common to Q1/Q2/Q4: the head's own null, read at the
        # family-wide bar so the gate and the joint reading use one alpha.
        null_ok = None if (holm is None or p is None) else bool(holm < 0.05)
        # ONE definition of the conjunct set, shared with the status pass
        # (RR4-1). Q2's incumbent conjunct is UNANSWERABLE (R-237, no sign
        # head); Q4's matched-random conjunct is simply UNCOMPUTED. Both read
        # null, and the difference is carried in the cell's status and detail
        # rather than by pretending one of them is false.
        conj = _gate_conjuncts(c, receipt, {})
        vals = list(conj.values())
        passed = (None if any(v is None for v in vals) else all(vals))
        counts["not_evaluable" if passed is None else
               ("passed" if passed else "failed")] += 1
        c["declared_gate_outcome"] = {
            "gate": gate.get("gate"), "conjuncts": conj, "passed": passed,
            "alpha": "holm_p < 0.05 over the declared family of 24",
            "why": "R-397 ruling 2: a head's survival is ITS OWN declared "
                   "gate. A conjunct nobody computed reads null, never "
                   "false — 'not evaluated' and 'evaluated and failed' are "
                   "different findings."}
    fam["declared_gate_outcomes"] = {
        "counts": counts,
        "heads_passing_their_own_gate": sorted(
            {c["head"] for c in cells.values()
             if (c["declared_gate_outcome"] or {}).get("passed") is True}),
        "separate_from": "survives_joint_reading_at_0_05, which is the "
                         "family-wide Holm reading and answers a different "
                         "question",
        "ruling": "R-397 ruling 2 (USER)"}
    return fam["declared_gate_outcomes"]


def finalise_family(receipt: dict) -> dict:
    """Build, re-status and re-assemble the family, IN ORDER. RR2-1.

    The order is load-bearing and was wrong by construction before: survivors
    were computed by `assemble_family` BEFORE anything knew which incumbent
    legs had been evaluated, so no later pass could change the flag it had
    already published. The sequence is now:

      1. `incumbent_legs_evaluated` -- a property of the RUN, computed first;
      2. `evaluate_family` -- cells, statuses, Holm;
      3. `apply_gate_evaluation_status` -- re-status the half-evaluated gates;
      4. RE-ASSEMBLE -- so `survives`, `surviving_cells`, `cells_by_status`
         and `distinct_results` are all derived from the FINAL statuses.

    Step 4 is the one that is easy to omit: re-statusing a cell without
    re-deriving the flags leaves the old verdict standing beside the new
    status, which is the contradiction this whole finding is about."""
    receipt["incumbent_legs_evaluated"] = incumbent_legs_evaluated(receipt)
    receipt["family"] = evaluate_family(receipt.get("results") or {},
                                        receipt.get("populations") or {})
    # RR4-1 needs the CONJUNCTS, so they are computed before the status pass
    # and recomputed after the re-assembly (holm moves, and a conjunct reads
    # it). Two passes, because the two facts genuinely depend on each other.
    attach_declared_gate_outcomes(receipt)
    ge = apply_gate_evaluation_status(receipt)
    cells = receipt["family"]["cells"]
    carried = {k: v for k, v in receipt["family"].items()
               if k in ("incumbent_null_applicability", "gate_evaluation")}
    receipt["family"] = I11.assemble_family(cells)
    receipt["family"].update(carried)
    receipt["family"]["gate_evaluation"] = ge
    attach_floor_disclosure(receipt)
    # AFTER the re-assembly, because a gate outcome reads holm_p and the
    # FINAL status; computing it earlier would read a verdict still to move.
    #
    # THIS CALL WENT MISSING ONCE, and the suite stayed green because every
    # control invokes `attach_declared_gate_outcomes` DIRECTLY — rule 17's
    # shape in the code written to close rule 17's shape. It was surfaced by
    # a mutant that could not find its anchor, not by a failing check. The
    # seam now demands this pass's evidence in the artifact, so the call
    # cannot vanish quietly again.
    attach_declared_gate_outcomes(receipt)
    return receipt["family"]


def apply_gate_evaluation_status(receipt: dict) -> dict:
    """RR2-1. A cell whose gate has an UNEVALUATED conjunct is re-statused.

    THE PREDICATE, from the reviewer's filing, verbatim in code:

        NOT (declared_gate.carries_incumbent_term
             AND comparable[head]
             AND NOT incumbent_counterpart_computed)

    All three inputs were ALREADY IN THE ARTIFACT and nothing compared them,
    so `survives = true` was printed beside the fields that contradict it --
    rule 10, in the only surviving head.

    THIS IS NOT THE F2 CONJUNCT AGAIN. F2 covered a counterpart that does NOT
    EXIST. Here it EXISTS and was not computed, which is the stronger claim
    and the weaker evidence. The cell is REPORTED under its own status and
    never dropped: the denominator stays 24, and the statistic and p it
    already carries stay readable so a later ruling can be applied without
    re-running.

    It does NOT wire the missing leg. Whether `apply_incumbent_hazard` runs
    for Q1 is an estimand-adjacent change to the only surviving head and is
    the USER's (rule 14); this function only stops the artifact asserting a
    joint reading it did not complete."""
    fam = receipt.get("family") or {}
    cells = fam.get("cells") or {}
    comparable = ((receipt.get("incumbent_null_applicability") or {})
                  .get("comparable") or {})
    legs = receipt.get("incumbent_legs_evaluated") or {}
    if not cells or not comparable or not legs:
        raise RuntimeError(
            f"REFUSED: the gate-evaluation pass read {len(cells)} cells, "
            f"{len(comparable)} declarations and {len(legs)} leg records. It "
            f"must not report 'nothing to re-status' from a read that "
            f"touched nothing (R-289).")
    moved, checked = [], 0
    for key, c in cells.items():
        head = c.get("head")
        gate = c.get("declared_gate") or {}
        checked += 1
        computed = (legs.get(head) or {}).get("incumbent_counterpart_computed")
        # RR4-1: DERIVE FROM THE CONJUNCTS, not from the incumbent-leg fact.
        # `gate_conjuncts_evaluated` claimed "every conjunct was evaluated"
        # while being computed from ONE conjunct, so twelve cells asserted
        # true while carrying a null conjunct (Q2's unanswerable incumbent
        # term, Q4's uncomputed matched-random one). Harmless while those
        # cells fail anyway; the RR2-1 shape the moment Q4's p improves.
        conj = ((c.get("declared_gate_outcome") or {}).get("conjuncts")
                if c.get("declared_gate_outcome") else None)
        if conj is None:
            conj = _gate_conjuncts(c, receipt, legs)
        unevaluated = sorted(k for k, v in conj.items() if v is None)
        c["gate_conjuncts_evaluated"] = not unevaluated
        c["gate_conjuncts_unevaluated"] = unevaluated
        # The incumbent-leg case keeps its own, more specific reason.
        partial_leg = bool(gate.get("carries_incumbent_term")
                           and comparable.get(head) and not computed)
        # A cell that already carries a MORE SPECIFIC non-OK status keeps it:
        # NO_INCUMBENT_COUNTERPART says WHY the conjunct is unanswerable, and
        # replacing it with the generic status would lose that.
        partial = partial_leg or bool(
            unevaluated and c.get("status") == I11.CELL_STATUS_OK)
        if not partial:
            continue
        moved.append(key)
        c["gate_partial_reason"] = ("incumbent leg declared and not computed"
                                    if partial_leg else
                                    f"conjuncts never evaluated: {unevaluated}")
        c["status_before_gate_check"] = c.get("status")
        c["status"] = I11.CELL_STATUS_GATE_PARTIAL
        c["detail"] = (
            f"GATE PARTIALLY EVALUATED ({c['gate_partial_reason']}; RR2-1 + "
            f"RR4-1). This head's frozen gate is "
            f"{gate.get('gate')!r} with conjuncts {list(gate.get('conjuncts') or ())}. "
            f"The incumbent counterpart EXISTS (comparable={comparable.get(head)}) "
            f"and was NOT COMPUTED, so only the matched-random conjunct was "
            f"evaluated and the cell cannot be published as passing a JOINT "
            f"reading. Its statistic and p are UNCHANGED and still carried, so "
            f"a ruling can be applied without re-running. Previous status: "
            f"{c['status_before_gate_check']!r}. " + str(c.get("detail") or ""))
    if checked == 0:
        raise RuntimeError(
            "REFUSED: the gate-evaluation pass visited ZERO cells.")
    fam["gate_evaluation"] = {
        "cells_checked": checked,
        "cells_gate_partially_evaluated": sorted(moved),
        "predicate": "NOT (declared_gate.carries_incumbent_term AND "
                     "comparable[head] AND NOT "
                     "incumbent_counterpart_computed)",
        "heads_affected": sorted({cells[k]["head"] for k in moved}),
        "wiring_decision_is_the_USERS": (
            "apply_incumbent_hazard is built and has no production call "
            "site. Wiring it changes the adjudication of the only surviving "
            "head, so it is escalated, not done here (rule 14)."),
        "why": "all three inputs were already in the artifact and nothing "
               "compared them; `survives = true` was printed beside the "
               "fields that contradict it (rule 10)."}
    return fam["gate_evaluation"]


def assert_dry_run_family(receipt: dict) -> dict:
    """--dry-run must ASSERT its own family before exiting 0. F-1(2).

    The dry run substitutes populations and runs everything else unchanged,
    so it is the right instrument for wiring -- but it is exempt from the
    output guard by declared mode and asserted nothing, so MUTANT A produced
    a contradictory artifact through it and exited 0. At minimum Q4 must
    carry an increment: that is the value the whole wiring exists to deliver,
    and it is absent exactly when the wiring is cut."""
    cells = ((receipt.get("family") or {}).get("cells")) or {}
    q4 = {k: c for k, c in cells.items() if c.get("head") == "Q4_combined_ev"}
    if not q4:
        raise RuntimeError(
            "REFUSED: --dry-run produced no Q4 cells at all, so it cannot "
            "have exercised the path it exists to exercise.")
    unpaired = sorted(k for k, c in q4.items()
                      if c.get("status") == I11.CELL_STATUS_NO_COUNTERPART
                      or c.get("p_value") is None)
    if unpaired:
        raise RuntimeError(
            f"REFUSED: --dry-run reached its exit with {len(unpaired)} of "
            f"{len(q4)} Q4 cells carrying no increment ({unpaired[:3]}). The "
            f"dry harness exists to prove the incumbent wiring end to end; "
            f"exiting 0 here is the silent-success shape one layer up.")
    # AND THE GUARDS MUST HAVE RUN, not merely exist. A guard REMOVED from
    # main() cannot be caught by running a healthy pipeline -- it only fires
    # when something else is also broken -- so the seam asserts the guards'
    # own EVIDENCE is present. This is how a deleted call site becomes
    # visible without a source-text check.
    ev = receipt.get("incumbent_applicability_guard") or {}
    if not isinstance(ev.get("checks"), int) or ev["checks"] <= 0:
        raise RuntimeError(
            f"REFUSED: the artifact carries no evidence that the "
            f"incumbent-applicability guard RAN ({ev!r}). A guard that was "
            f"not called is indistinguishable from one that passed.")
    if receipt.get("as_of") and not receipt.get("as_of_names"):
        raise RuntimeError(
            "REFUSED: the artifact carries an as_of that does not say WHICH "
            "INSTANT it names. as_of precedes written_at by the fit's "
            "duration, and rule 8's as-of is the POPULATION READ instant; a "
            "bare timestamp leaves a reader to infer which one it is.")
    # R-397 ruling 1: Q1's leg must have ARRIVED, checked at the artifact.
    # A source grep would pass on a call whose result never reaches the cell,
    # which is exactly how MUTANT A defeated the previous guard.
    _legs = receipt.get("incumbent_legs_evaluated") or {}
    _owed = sorted(h for h, v in (
        (receipt.get("incumbent_null_applicability") or {}).get("comparable")
        or {}).items() if v)
    _missing = sorted(h for h in _owed
                      if not (_legs.get(h) or {}).get(
                          "incumbent_counterpart_computed"))
    for _arms in (receipt.get("results") or {}).values():
        for _a, _rep in _arms.items():
            _qi = _rep.get("q1_incremental")
            if _qi is not None and not (_qi.get("incumbent_provenance") or {}
                                        ).get("sha256_prefix"):
                raise RuntimeError(
                    f"REFUSED: {_a}'s Q1 incremental leg carries no incumbent "
                    f"provenance. A constant vector produces a well-formed "
                    f"AUC comparison and would otherwise read as a computed "
                    f"leg (R-397 ruling 1).")
    if _missing:
        raise RuntimeError(
            f"REFUSED: {_missing} are declared comparable and their incumbent "
            f"leg was NOT computed. R-397 ruling 1 wired Q1's leg; an "
            f"artifact emitted without it is the half-evaluated gate the "
            f"ruling exists to close.")
    fa = receipt.get("frozen_prereg_anchor") or {}
    if (not fa.get("chain_terminates_at")
            or not fa.get("gate_carries_incumbent_term")
            or len(fa.get("gate_text_verified_heads") or []) != 4):
        raise RuntimeError(
            f"REFUSED: the artifact carries no evidence that the frozen-"
            f"preregistration anchor RAN ({sorted(fa)}). Without it the "
            f"coverage floor's expected set terminates at an editable "
            f"constant (RR3-1), and a check not called is indistinguishable "
            f"from one that passed.")
    dgo = ((receipt.get("family") or {}).get("declared_gate_outcomes")) or {}
    _c = dgo.get("counts") or {}
    if sum(_c.values()) != len(cells):
        raise RuntimeError(
            f"REFUSED: the declared-gate pass left evidence for "
            f"{sum(_c.values())} cells against {len(cells)} in the family "
            f"({dgo.get('counts')!r}). R-397 ruling 2 makes each head's own "
            f"gate outcome a published field; a pass that did not run leaves "
            f"the joint-reading flag answering a question it was never asked.")
    ge = ((receipt.get("family") or {}).get("gate_evaluation")) or {}
    if not isinstance(ge.get("cells_checked"), int) or ge["cells_checked"] <= 0:
        raise RuntimeError(
            f"REFUSED: the artifact carries no evidence that the "
            f"gate-evaluation pass RAN ({ge!r}). Without it a cell whose "
            f"declared gate was half evaluated is published as surviving "
            f"(RR2-1), and a pass that was not called is indistinguishable "
            f"from one that found nothing.")
    stale = sorted(k for k, c in cells.items()
                   if c.get("status") == I11.CELL_STATUS_GATE_PARTIAL
                   and c.get("survives_joint_reading_at_0_05"))
    if stale:
        raise RuntimeError(
            f"REFUSED: {len(stale)} cells are GATE_PARTIALLY_EVALUATED and "
            f"still flagged surviving ({stale[:3]}). The family was "
            f"re-statused without being RE-ASSEMBLED, so the old verdict is "
            f"standing beside the new status.")
    nofloor = sorted(k for k, c in cells.items()
                     if "permutation_floor" not in c)
    if nofloor:
        raise RuntimeError(
            f"REFUSED: {len(nofloor)} cells carry no permutation_floor "
            f"disclosure ({nofloor[:3]}). F-3's whole finding is that the "
            f"disclosure was absent exactly on the cells that decide.")
    return {"q4_cells": len(q4), "all_carry_an_increment": True,
            "guard_evidence_checks": ev["checks"],
            "gate_evaluation_cells_checked": ge["cells_checked"],
            "cells_with_floor_disclosure": len(cells)}


def assert_incumbent_applicability_honoured(receipt: dict) -> dict:
    """A head declared comparable must SHOW its incumbent in the cells. F-1.

    The declaration and the cells are two statements about the same fact and
    nothing compared them: `incumbent_null_applicability.comparable[head]`
    said the incumbent applies while every cell of that head said it had no
    counterpart, for a whole batch, with the run exiting 0.

    SCOPE, stated rather than silently chosen. Two different obligations:
      * EVERY comparable head must carry status OK in all its cells -- that is
        what a live counterpart looks like from the cell side;
      * heads adjudicated FROM ECONOMICS (Q4) must additionally show
        `paired_against_incumbent` and a non-null `incumbent_net_cents` in the
        economics block behind every cell.
    Q1_arrival is comparable and is NOT adjudicated from economics, so the
    second obligation does not reach it. That is a real gap and it is
    REPORTED, not silently scoped away -- see `incumbent_legs_evaluated`."""
    fam = (receipt.get("family") or {})
    cells = fam.get("cells") or {}
    ina = receipt.get("incumbent_null_applicability") or {}
    comparable = ina.get("comparable") or {}
    if not cells or not comparable:
        raise RuntimeError(
            f"REFUSED: the incumbent-applicability guard read {len(cells)} "
            f"cells and {len(comparable)} declarations. A guard that runs on "
            f"an empty read reports a pass it never established (R-289).")
    # RR2-2: THE EXPECTED SET IS DECLARED BY THE PRODUCER, NOT INFERRED FROM
    # WHAT THE RUN HAPPENED TO CONTAIN. `checked == 0` catches an empty read
    # and nothing caught COVERAGE SHRINKING: with Q4 unwired and its
    # `comparable` flipped to false, the guard was ADMITTED at 6 checks and
    # one head, and the emitted `checks: 6` was the only trace. Same rule
    # A1.4 already applies to the Holm denominator -- a set must not shrink
    # to what was evaluable (R-230: expected sets are producer-recorded
    # facts, never checker assumptions).
    expected = sorted(h for h, v in INCUMBENT_COMPARABLE.items() if v)
    realised = sorted(h for h, v in comparable.items() if v)
    if set(realised) < set(expected):
        raise RuntimeError(
            f"REFUSED: the artifact declares comparable heads {realised} but "
            f"INCUMBENT_COMPARABLE -- which transcribes the frozen prereg 3 "
            f"gates -- declares {expected}. Coverage SHRANK, and a smaller "
            f"realised set would otherwise be reported as a smaller pass: "
            f"missing {sorted(set(expected) - set(realised))}.")
    bad_status, bad_econ, checked = [], [], 0
    for head, is_comp in comparable.items():
        if not is_comp:
            continue
        for key, c in cells.items():
            if c.get("head") != head:
                continue
            checked += 1
            # RR2-1 INTERACTION, resolved on the DIFFERENCE THAT MATTERS.
            # This guard exists to catch a cell that DENIES a counterpart the
            # declaration says exists. GATE_PARTIALLY_EVALUATED does the
            # opposite: it DISCLOSES that the counterpart exists and was not
            # computed, which is precisely what RR2-1 requires the artifact
            # to say. Refusing it would make the honest disclosure
            # unemittable and leave NO_INCUMBENT_COUNTERPART -- the false
            # one -- as the only way to ship.
            #
            # The contradiction is still refused, and so is MUTANT A: an
            # unwired Q4 also fails the economics check below, which no
            # status can satisfy. Verified in the suite, both ways.
            if c.get("status") not in (I11.CELL_STATUS_OK,
                                       I11.CELL_STATUS_GATE_PARTIAL):
                bad_status.append(f"{key}: comparable={head} but status "
                                  f"{c.get('status')!r}")
    if comparable.get("Q4_combined_ev"):
        for coin, arms in (receipt.get("results") or {}).items():
            for arm, rep in arms.items():
                for b, econ in (rep.get("economics") or {}).items():
                    checked += 1
                    if not econ.get("paired_against_incumbent"):
                        bad_econ.append(f"{coin}/{arm}/{b}: "
                                        f"paired_against_incumbent is not true")
                    elif econ.get("incumbent_net_cents") is None:
                        bad_econ.append(f"{coin}/{arm}/{b}: "
                                        f"incumbent_net_cents is null")
    if checked == 0:
        raise RuntimeError(
            "REFUSED: the guard visited ZERO cells and ZERO economics blocks, "
            "so it proved nothing. Absence must never read as a pass.")
    if bad_status or bad_econ:
        raise RuntimeError(
            f"REFUSED: a head is DECLARED comparable to the incumbent and its "
            f"cells do not show one. This is the Q-DA-197 contradiction, and "
            f"it is what an unwired incumbent looks like at the artifact even "
            f"when every source-text guard passes. Cells: {bad_status[:4]}; "
            f"economics: {bad_econ[:4]}. ({len(bad_status)} cell(s), "
            f"{len(bad_econ)} economics block(s) over {checked} checks.)")
    partial = sorted(k for k, c in cells.items()
                     if c.get("status") == I11.CELL_STATUS_GATE_PARTIAL)
    return {"checks": checked, "comparable_heads": realised,
            "expected_comparable_heads": expected,
            "coverage_is_complete": set(realised) >= set(expected),
            "cells_admitted_as_DISCLOSED_gaps": partial,
            "admitted_gap_rule": (
                "a comparable head's cell may carry OK or "
                "GATE_PARTIALLY_EVALUATED. The first says the counterpart was "
                "computed; the second DISCLOSES that it exists and was not. "
                "NO_INCUMBENT_COUNTERPART on a comparable head is the "
                "CONTRADICTION this guard refuses, and an unwired arm also "
                "fails the economics check, which no status can satisfy."),
            "verified": "declared-comparable heads show OK cells; "
                        "economics-adjudicated ones also show a paired "
                        "incumbent with a non-null net"}


def incumbent_legs_evaluated(receipt: dict) -> dict:
    """Which comparable heads actually had an incumbent counterpart COMPUTED.

    The guard above cannot demand economics from a head that is not
    adjudicated from economics, so scoping it correctly would quietly hide
    that Q1_arrival is declared comparable and its incremental leg is NOT
    computed: `apply_incumbent_hazard` exists, is falsifier-proven, and has no
    call site -- defect I11-2's shape, in the head that SURVIVES.

    Reported, never acted on. Changing what Q1's cells claim is an
    adjudication change on the only surviving head; models estimate, policy
    decides (rule 14). BE files it; the USER rules it."""
    ina = receipt.get("incumbent_null_applicability") or {}
    comparable = ina.get("comparable") or {}
    econ_backed = set()
    for arms in (receipt.get("results") or {}).values():
        for rep in arms.values():
            if any((e or {}).get("paired_against_incumbent")
                   for e in (rep.get("economics") or {}).values()):
                econ_backed.add("Q4_combined_ev")
            # R-397 ruling 1: Q1's leg is COMPUTED when the arm carries a
            # real incumbent-hazard comparison on the identical population.
            # Read from the RESULT, never from the presence of a call site.
            _q1i = rep.get("q1_incremental") or {}
            if (_q1i.get("incumbent_auc") is not None
                    and (_q1i.get("incumbent_provenance") or {})
                    .get("sha256_prefix")):
                econ_backed.add("Q1_arrival")
    out = {}
    for head, is_comp in comparable.items():
        if not is_comp:
            out[head] = {"declared_comparable": False,
                         "incumbent_counterpart_computed": False,
                         "note": "not declared comparable; no leg is owed"}
            continue
        done = head in econ_backed
        out[head] = {
            "declared_comparable": True,
            "incumbent_counterpart_computed": done,
            "note": ("the incumbent's own value is computed on the identical "
                     "rows and enters the adjudicated statistic" if done else
                     "DECLARED COMPARABLE, NO COUNTERPART COMPUTED. The "
                     "frozen gate for this head carries an incumbent term and "
                     "only the matched-random leg was evaluated; the cell's "
                     "OK status therefore rests on one of its two declared "
                     "conjuncts. `apply_incumbent_hazard` is built and "
                     "unwired. ESCALATED, not silently corrected.")}
    return out


def declared_outputs_for(argv=None) -> tuple:
    """The outputs THIS invocation is required to produce.

    A `--coin` run writes the sliced artifact and deliberately NOT the full
    one, because a one-coin file at the whole run's path would be read as the
    whole result. The guard must therefore ask for the file this MODE declares
    -- checking for the unsliced name made a completed sliced run refuse at its
    exit, having already written 96 KB of results.

    The refusal itself was RIGHT and stays: "a clean exit that produced nothing
    is the silent-success shape". Only the DECLARATION was wrong, and a guard
    that demands the wrong artifact is a guard that cannot pass.
    """
    a = sys.argv if argv is None else argv
    _rj = _readjudicate_arg(a)
    if _rj is not None:
        # A re-adjudication declares the SUPERSEDING file, never the source:
        # demanding the source would pass on a run that wrote nothing, and
        # demanding the full artifact is the mode-blind bug fixed at 0b1f6bb.
        return (readjudicated_path(_rj),)
    if "--coin" in a:
        i = a.index("--coin")
        if i + 1 < len(a) and not a[i + 1].startswith("-"):
            return (OUT.with_name(f"{OUT.stem}__coin_{a[i + 1]}{OUT.suffix}"),)
    return DECLARED_OUTPUTS


def assert_outputs_written(outputs=None, argv=None) -> dict:
    outputs = declared_outputs_for(argv) if outputs is None else outputs
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


def _dry_incumbent_fitdir(coin: str, width: int, seed: int = 20260901):
    """A SYNTHETIC incumbent for --dry-run, written so the REAL loader reads it.

    The dry harness must exercise the wiring, not bypass it: if the dry path
    called `apply_incumbent` directly it would prove nothing about
    `load_verified_incumbent`, and the whole reason this harness exists is
    that a component suite cannot see an unwired main(). So this writes a
    genuine artifact plus a genuine `fit_manifest.json` binding its hash, and
    main() loads it through the same verified path a real run uses.

    It reads NO real model artifact -- the dry run's declaration stays true --
    and its width is taken from the synthetic block, so a shape drift refuses
    here exactly as it would on real data."""
    import hashlib
    import random as _r
    import tempfile as _tf
    rr = _r.Random(seed)
    d = Path(_tf.mkdtemp(prefix="iter011_dry_incumbent_"))
    art = d / f"linear_d_{coin}.json"
    art.write_text(json.dumps({
        "arm": INCUMBENT_ARM,
        "norm_mu": [rr.random() for _ in range(width)],
        "norm_sd": [1.0 + rr.random() for _ in range(width)],
        "hazard_weights": [rr.uniform(-1, 1) for _ in range(width + 1)],
        "value_weights": [rr.uniform(-1, 1) for _ in range(width + 1)],
        "SYNTHETIC": "dry-run only; not a fitted model"}, sort_keys=True))
    (d / PA.FIT_MANIFEST).write_text(json.dumps({"file_hashes": {
        art.name: hashlib.sha256(art.read_bytes()).hexdigest()[:16]}}))
    return d


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

    # --readjudicate: apply a COMPOSITION ruling to a completed run's own
    # preserved evidence. It fits nothing and scores nothing, so it is not
    # gated on memory or on the tape; it IS gated on the selftest above,
    # because a re-adjudication is a number-bearing output like any other.
    _rj = _readjudicate_arg()
    if _rj is not None:
        _o, _p = readjudicate(_rj)
        _ch = _o["supersedes"]["n_cells_changed"]
        print(f"\nREADJUDICATED {_rj.name} -> {_p.name}: {_ch} of 24 cells "
              f"moved; source untouched", flush=True)
        for _k, _v in sorted(_o["supersedes"]["cells_changed"].items()):
            print(f"  {_k}: {_v['was']['status']} p={_v['was']['p_value']} "
                  f"-> {_v['now']['status']} p={_v['now']['p_value']}",
                  flush=True)
        print(f"WROTE {_p.name}: {assert_outputs_written((_p,))}", flush=True)
        return 0

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
    # --coin <name>: run ONE coin end to end. Memory, not estimand -- see
    # `_coin_slice`. Unknown names refuse; a bare `--coin` with no value
    # refuses too, rather than silently running the full population under a
    # flag the operator believed had restricted it.
    _coin = None
    if "--coin" in sys.argv:
        _i = sys.argv.index("--coin")
        if _i + 1 >= len(sys.argv) or sys.argv[_i + 1].startswith("-"):
            raise SystemExit("REFUSED: --coin needs a value "
                             f"(one of {COINS_011}); a flag with no value "
                             "must not read as 'no restriction'.")
        _coin = sys.argv[_i + 1]
        if _coin not in COINS_011:
            raise SystemExit(f"REFUSED: --coin {_coin!r} is not one of "
                             f"{COINS_011}.")
        print(f"  COIN SLICE: {_coin} only. Same rows, same numbers, "
              f"less resident memory -- the results loop was already "
              f"per-coin. Output carries the slice in its name.", flush=True)
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
        # NOT coin-sliced: `_feature_pass` builds BOTH coins from this index
        # and refuses when a coin joins 0 rows -- "Drops absorb row-level
        # anomalies, never total input failures". Slicing here made eth fail
        # 520,033 of 520,033 and the absorption guard REFUSED, correctly.
        # The saving has to come from what we CARRY, not from what we index.
        TP = PA.tape_index("train")
        print(f"  train split indexed: {len(TP):,} rows", flush=True)
        FIT = _coin_drop(PA._feature_pass(PA.FRAGMENT, "fragment", TAPE=TP),
                         _coin, "fragment")
        for _c in list(FIT):
            restore_valuation_gate(FIT[_c])
        del TP
        print("  indexing score split for the embargo boundary...", flush=True)
        SP = PA.tape_index("score")   # same reason as the train index above
        print(f"  score split indexed: {len(SP):,} rows", flush=True)
    else:
        ident = {"tape_sha256_prefix": "DRY_RUN", "DRY_RUN": True,
                 "fragment_sha256_prefix": "DRY_RUN",
                 "topup_sha256_prefix": "DRY_RUN"}
        FIT, EVAL = _synth_pair()
        # THE DRY HARNESS MUST COVER THE SLICE TOO. Without this, --dry-run
        # --coin btc ran BOTH coins and reported success, so the slicing would
        # have reached a real run having never been exercised through main() --
        # the exact hole this harness exists to close (defect I11-2).
        FIT = _coin_drop(FIT, _coin, "dry fragment")
        EVAL = _coin_drop(EVAL, _coin, "dry topup")
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
    # FREE THE PROBE BEFORE THE TOPUP PASS, NOT AFTER. It is finished the
    # moment the purge loop above ends, and it is ~640k dicts. Holding it
    # across `_feature_pass` -- which itself json-loads the whole topup file
    # and builds both coins' families -- put it inside the peak for no reason.
    # Measured: the OOM landed in `[topup/eth]`, holding FIT + SP + probe.
    del probe
    # PACK FIT BEFORE THE TOPUP PASS ALLOCATES. This is the ordering that
    # matters: the purge above is the last consumer of the three parallel
    # lists, and the topup pass is the measured peak. Releasing ~4 GB of
    # Python float objects here is what lets the next pass fit.
    for _c in list(FIT):
        compact_design(FIT[_c])
    import gc as _gc
    _gc.collect()
    if not _dry:
        EVAL = _coin_drop(PA._feature_pass(PA.TOPUP, "topup", TAPE=SP),
                          _coin, "topup")
        for _c in list(EVAL):
            restore_valuation_gate(EVAL[_c])
    del SP
    for _c in list(EVAL):
        compact_design(EVAL[_c])
    _gc.collect()

    _prov = producing_code_provenance()
    # F5: `_tape_identity` lives in `phase2_arms.py`, a LATTICE file the frozen
    # candidate binds, so the ref is filled HERE rather than there. It is only
    # ever FILLED, never overwritten: if the lattice ever starts recording its
    # own ref, that one wins and the disagreement is reported instead of
    # silently resolved.
    if not ident.get("fit_code_ref"):
        ident["fit_code_ref"] = _prov["fit_code_ref"]
        ident["fit_code_ref_source"] = ("filled by the runner (outside the "
                                        "lattice); phase2_arms._tape_identity "
                                        "records content hashes only")
    elif ident["fit_code_ref"] != _prov["fit_code_ref"]:
        ident["fit_code_ref_disagreement"] = {
            "lattice": ident["fit_code_ref"], "runner": _prov["fit_code_ref"],
            "why": "two sources named different commits; neither is dropped"}
    out = {"artifact": "iter011_conditional_value_v1",
           "receipt_family": RECEIPT_FAMILY,
           # LOW (re-review): as_of precedes written_at by ~13 min and did not
           # say which instant it names. Rule 8's as-of exists because the
           # TAPE GROWS DURING MEASUREMENT, so the POPULATION READ instant is
           # the meaningful one; `written_at` is the emission. Both travel,
           # and the field now says which is which rather than leaving a
           # reader to infer it from the ordering.
           "as_of": run_as_of(),
           "as_of_names": "the POPULATION READ instant — the moment this run "
                          "finished reading its populations and began "
                          "composing results. It is the instant rule 8 asks "
                          "for (the tape grows during measurement). "
                          "`written_at` is the LATER emission instant; the "
                          "gap between them is fit and adjudication time, "
                          "not additional data.",
           "producing_code": _prov,
           "preregistration": PREREG_DOC,
           "preregistration_commit": PREREG_COMMIT,
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
        # PLACED BEFORE Xf/Xe ARE MATERIALISED, deliberately. Those two
        # rebuild the Python lists `compact_design` just released and
        # are the measured peak (12.0 G against a 12 G cap on the last
        # run); the incumbent depends only on EVAL[coin] and its own
        # artifact, so computing it here keeps it out of that window.
        # It changes no number: same model, same rows, same order.
        # ---- Q4's INCUMBENT: LOAD, VERIFY, APPLY -- now actually CALLED ----
        # THE DEFECT THIS CLOSES. R-280 built this path red-first and its own
        # commit says "not yet run": eleven falsifiers, a positive control, a
        # tampered-hash known-bad -- and ZERO call sites in main(). The run
        # therefore called `q4_economics(ap, rows)` with no incumbent, every
        # Q4 cell reported NO_INCUMBENT_COUNTERPART, and the decision metric
        # had no result. That is defect I11-2's shape exactly (rule 17:
        # suite-green is not pipeline-wired), one batch later.
        #
        # Loaded ONCE per coin, because the incumbent does not depend on the
        # arm, and applied to the SAME row range the candidate scored -- the
        # identical action population prereg 5.2 requires. `q4_economics`
        # refuses a length mismatch, so "identical" is checked, not asserted.
        _incdir = (_dry_incumbent_fitdir(
            coin, EVAL[coin]["w"]["PM"] + EVAL[coin]["w"]["FN"])
            if _dry else None)
        _incm = load_verified_incumbent(coin, fitdir=_incdir)
        _rng = range(len(EVAL[coin]["kept"]))
        inc_pred = apply_incumbent(_incm, EVAL[coin], _rng)
        # R-397 RULING 1: apply_incumbent_hazard gets its PRODUCTION CALL
        # SITE. It was built red-first at R-280, falsifier-proven, and had
        # ZERO call sites, so Q1 -- the head that carried the family's only
        # survivors -- was adjudicated on ONE of its two declared conjuncts.
        # The USER ruled the leg wired; the cells re-adjudicate honestly and
        # either answer is the result of record.
        inc_haz = apply_incumbent_hazard(_incm, EVAL[coin], _rng)
        out.setdefault("q4_incumbent", {})[coin] = {
            "arm": inc_pred["arm"], "n_scored": inc_pred["n"],
            "provenance": inc_pred["provenance"],
            "synthetic_dry_run": bool(_dry),
            "why": "prereg 5.2's increment is candidate MINUS the COMMITTED "
                   "incumbent on the IDENTICAL action population; the "
                   "incumbent is loaded and hash-verified, never re-fitted "
                   "(a re-fit incumbent is a different incumbent, R-280)"}
        print(f"  [{coin}/incumbent] {inc_pred['arm']} applied to "
              f"{inc_pred['n']:,} rows; "
              f"{(inc_pred['provenance'] or {}).get('sha256_prefix')}",
              flush=True)

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
            rep["economics"] = q4_economics(ap, EVAL[coin]["kept"],
                                            incumbent=inc_pred)
            # Q1's second conjunct, on the IDENTICAL action population the
            # candidate scored (prereg 5.2's "identical action population").
            rep["q1_incremental"] = q1_incremental(
                ap["p_fill"], inc_haz, EVAL[coin]["kept"], tge["y_fill"])
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
        del inc_pred, inc_haz, Xf, Xe, FIT[coin]["PM"], FIT[coin]["FN"], FIT[coin]["ST"]

    # I11-2: the DECLARED family, actually evaluated and adjudicated.
    # RR2-1: ordered in `finalise_family`, because survivors used to be
    # published before anything knew which incumbent legs had been evaluated.
    finalise_family(out)
    out["cluster_disclosure"] = I11.cluster_disclosure(
        min((p["eval_population_and_reach"]["G_complete_utc_days"]
             for p in out["populations"].values()), default=0), "window")
    assert_receipt_has_all_cells(out)
    # F-1: the declaration and the cells must AGREE, checked at the artifact.
    # RR3-1: check the PREMISE where it is CONSUMED. The coverage floor below
    # reads INCUMBENT_COMPARABLE, so the anchor runs in the emission chain and
    # its evidence is emitted -- not only in the suite, where a reader of the
    # artifact could not see it. In a dry run the synthetic fitdir supplies
    # the incumbent, so the check exercises the same path without reading a
    # real model artifact.
    out["frozen_prereg_anchor"] = assert_constants_match_frozen_prereg(
        fitdir=_incdir)
    out["incumbent_applicability_guard"] = \
        assert_incumbent_applicability_honoured(out)
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
    if _coin is not None:
        # A SLICED RUN MUST NOT CLAIM THE DECLARED OUTPUT. The artifact name is
        # what a reader resolves; a one-coin file sitting at the full run's
        # path would be read as the whole result, and the eth half's absence
        # would be invisible. The slice travels in the NAME and in the body.
        _out_path = OUT.with_name(f"{OUT.stem}__coin_{_coin}{OUT.suffix}")
        out["COIN_SLICE"] = {
            "coin": _coin, "is_partial_run": True,
            "declared_output_not_written": OUT.name,
            "why": "memory slicing, not an estimand change: the results loop "
                   "is per-coin, so this file holds exactly the rows and "
                   "numbers a full run would have produced for this coin",
            "adjudication_note": "the 24-cell family adjudicates btc alone "
                                 "(R-306); eth is reported and never "
                                 "adjudicated, so an eth slice can never "
                                 "carry a verdict"}
    if _dry:
        # F-1(2): THE SEAM MUST BE ABLE TO FAIL. The dry harness runs main()'s
        # own path -- it is how the packing defect was found -- but it
        # asserted NOTHING about its own output and is exempt from the output
        # guard by declared mode, so it exited 0 with the wiring cut. A
        # harness that cannot fail is not a harness (rule 16).
        assert_dry_run_family(out)
        import tempfile as _tf_dry
        _out_path = Path(_tf_dry.mkdtemp()) / OUT.name
        out["DRY_RUN"] = {
            "synthetic_populations": True, "real_data_read": False,
            "model_artifacts_read": False,
            "synthetic_incumbent_written_and_loaded": True,
            "why_synthetic_incumbent": "the Q4 increment path is WIRING, and "
                                       "a harness that bypassed the loader "
                                       "would leave exactly the hole this "
                                       "harness exists to close; no real "
                                       "model artifact is read",
            "output_path": str(_out_path),
            "why": "exercises main()'s OWN path — a component suite cannot see "
                   "an unwired main(), which is defect I11-2"}
    # BOTH ENDS OF A LONG RUN. `as_of` above is stamped when the results
    # block is composed; this run takes ~45 min and the tape grows underneath
    # it, so a single timestamp would silently stand for an interval.
    out["written_at"] = run_as_of()
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


def run_as_of() -> str:
    """This run's UTC wall-clock, ISO-8601. Q-DA-197 F6.

    Rule 8: every quoted population carries its n AND its as-of, "the tape
    grows during measurement". The artifact carried n throughout and no
    timestamp anywhere, so two runs over a growing tape were indistinguishable
    from each other at the artifact."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def producing_code_provenance(_git_runner=None) -> dict:
    """The COMMIT this run is produced from, plus the bytes. Q-DA-197 F5.

    `identity.fit_code_ref` was null while the content identity was present
    and verifiable, so the artifact could prove WHAT ran and never say WHICH
    COMMIT it was. Filled here, in the runner, because `_tape_identity` lives
    in `phase2_arms.py` -- a lattice file the frozen candidate binds.

    A REF ALONE IS NOT AN IDENTITY. A dirty tree means the commit names bytes
    that are not the bytes that ran, so `working_tree_dirty` travels with it
    and both 011 module hashes ride along; the ref is then checkable rather
    than trusted. A git failure is a NAMED absence, never a silent empty
    string -- absence must not read as a pass (rule 11)."""
    import hashlib
    import subprocess

    def git(*a):
        # `_git_runner` exists so the UNREADABLE-GIT path can be executed.
        # It injects the FAILURE (an input), never the answer -- a fixture
        # that supplied the output would be the rule-16 trap. Without it the
        # tri-state's "unknown" branch is unreachable on a healthy machine and
        # a mutant deleting it survives, which is measured, not feared.
        if _git_runner is not None:
            return _git_runner(*a)
        try:
            r = subprocess.run(("git", *a), cwd=str(Path(__file__).resolve().parent),
                               capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except Exception:                            # noqa: BLE001
            return None

    me = Path(__file__).resolve()
    lib = me.parent / "phase2_iter011.py"

    def sha(f):
        return hashlib.sha256(f.read_bytes()).hexdigest()[:16]

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    # F-7 TRI-STATE. `bool(dirty)` reported an UNREADABLE git status as
    # CLEAN, with the disclosure demoted to a separate field -- rule 11
    # applies to the flag itself, not to a note beside it. "unknown" is now a
    # value of the flag, so a reader who looks only at the flag cannot be
    # told "clean" by a check that never ran.
    if status is None:
        dirty = "unknown"
        paths, code_paths = None, None
    else:
        lines = [ln for ln in status.splitlines() if ln.strip()]
        dirty = bool(lines)
        paths = sorted(ln[3:].strip() for ln in lines)[:40]
        # F-7: and WHICH paths, because "dirty: true" alone made a reader use
        # git to establish that the dirt did not touch the producing code.
        code_paths = sorted(x for x in (paths or [])
                            if x.endswith(("phase2_iter011.py",
                                           "phase2_iter011_run.py"))
                            or Path(x).name in PA.CODE_IDENTITY_FILES)
    return {
        "fit_code_ref": head or "UNAVAILABLE: git rev-parse failed; the "
                                "producing commit could not be read and is "
                                "NOT being reported as absent-but-fine",
        "fit_code_ref_resolved": bool(head),
        "working_tree_dirty": dirty,
        "working_tree_dirty_is_tristate": "true | false | 'unknown'",
        "working_tree_status_read": status is not None,
        "dirty_paths": paths,
        "dirty_paths_touching_the_producing_code": code_paths,
        "producing_code_was_clean": (None if status is None
                                     else not code_paths),
        "runner": me.name, "runner_sha256_prefix": sha(me),
        "library": lib.name, "library_sha256_prefix": sha(lib),
        "carrying_commit": head or "UNAVAILABLE",
        "why": "a content hash says WHAT ran; a commit ref says WHICH "
               "COMMIT. The artifact carried the first and null for the "
               "second (Q-DA-197 F5). If the tree is dirty the ref names "
               "bytes that are not these bytes, so both travel together."}


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
