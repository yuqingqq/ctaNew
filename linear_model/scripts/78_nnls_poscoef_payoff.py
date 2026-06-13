"""Step 78 (compressed): constrained-linear-model payoff trial.

Question (compressed per user + claude review): Steps 76/77 already settled
that (a) the model wastes feature signs and (b) the equal-/shrunk-IC composite
of the 22 V2 features has an economically dead payoff (decile rho -0.49,
inverts at every K). The ONLY linear variant not yet tried is a *constrained*
model that does implicit feature selection — NNLS and positive-coefficient
Ridge on sign-oriented features. If a fitted non-negative subset produces a
MONOTONE, tradable payoff that the hand-weighted composites do not, the
linear-on-current-features line is worth developing; if not, it is closed and
we pivot to Batch B (price x volume interactions).

All five model forms are run through the SAME Step-77 payoff diagnostic
(imported verbatim), on the SAME Step-76 testbed / fold protocol / per-cycle
cross-sectional-z features — so the ONLY thing that varies is the weighting
scheme. raw-Ridge and signed-equal/shrunk are recomputed in-harness as anchors
(signed_shrunk MUST reproduce Step-76 +0.0517 — built-in sanity).

PRE-REGISTERED PAYOFF GATE (fixed before run; IC is context only, NOT the gate
— the 75->76->77 arc proved IC alone is a trap):
  a model 'rescues' the line iff its OOS composite has
    decile monotonicity rho >= +0.60   AND
    K=3 long/short spread   >= +9.0 bps AND
    >= 6/9 folds with positive K=3 spread.
If neither NNLS nor positive-Ridge clears -> linear-on-current-features
CLOSED; next = Batch B only. No backtest here.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
s77 = _imp("s77", "linear_model/scripts/77_orientation_decile_diag.py")
from ml.research.alpha_v4_xs_1d import _slice

OUT = REPO / "linear_model/results/step78_nnls_poscoef_payoff"
OUT.mkdir(parents=True, exist_ok=True)
OOS = s76.OOS
BLOCK = s76.BLOCK
DROP = s76.DROP
ALPHAS = s58.ALPHAS
GATE_RHO, GATE_SPREAD, GATE_FOLDS = 0.60, 9.0, 6


def _xsz(g: pd.DataFrame, fc: list) -> np.ndarray:
    """Per-cycle cross-sectional z of features (identical to s76/s77)."""
    X = g[fc].to_numpy(float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd <= 1e-12] = 1.0
    return np.nan_to_num((X - mu) / sd)


def pos_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Positive-coefficient Ridge via augmented NNLS (exact, version-proof)."""
    n_f = X.shape[1]
    Xa = np.vstack([X, np.sqrt(alpha) * np.eye(n_f)])
    ya = np.concatenate([y, np.zeros(n_f)])
    coef, _ = nnls(Xa, ya)
    return coef


def build_model_scores(model: str) -> pd.DataFrame:
    """Per-(cycle,symbol) OOS score for one model form. Columns:
    open_time, fold, symbol, score, y(=alpha_beta)."""
    if build_model_scores._cache is None:
        _panel, px, fc, folds = s76.s67.build_panel(DROP)
        px["open_time"] = pd.to_datetime(px["open_time"], utc=True)
        grid = sorted(px["open_time"].unique())[::BLOCK]
        dec = px[px["open_time"].isin(set(grid))].copy()
        dec = s76.assign_folds(dec, folds)
        build_model_scores._cache = (dec, fc, folds)
    dec, fc, folds = build_model_scores._cache

    rows = []
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0].dropna(subset=["alpha_beta"])
        tr_t = tr.dropna(subset=["target_z"]) if model != "signed_equal" else tr
        if len(tr) < 500:
            continue
        # orientation sign from pre-fold per-cycle IC (identical basis to s76)
        w76 = s76.fit_weights(tr, fc, "alpha_beta")
        sign = np.array([np.sign(w76[f]) or 1.0 for f in fc], float)
        shrunk = np.array([w76[f] for f in fc], float)

        coef = None
        if model in ("ridge_xsz", "nnls_oriented", "posridge_oriented"):
            Xtr, ytr = [], []
            for _, g in tr_t.groupby("open_time", sort=True):
                if len(g) < 5:
                    continue
                Z = _xsz(g, fc)
                yv = g["target_z"].to_numpy(float)
                if not np.isfinite(yv).all() or np.std(yv) <= 1e-12:
                    continue
                if model != "ridge_xsz":
                    Z = Z * sign                       # orient
                Xtr.append(Z)
                ytr.append(yv)
            if not Xtr:
                continue
            Xtr = np.vstack(Xtr)
            ytr = np.concatenate(ytr)
            if model == "ridge_xsz":
                m = RidgeCV(alphas=ALPHAS, scoring="r2", fit_intercept=False)
                m.fit(Xtr, ytr)
                coef = m.coef_
            elif model == "nnls_oriented":
                coef, _ = nnls(Xtr, ytr)
            else:  # posridge_oriented — alpha from unconstrained RidgeCV
                a = RidgeCV(alphas=ALPHAS, scoring="r2",
                            fit_intercept=False).fit(Xtr, ytr).alpha_
                coef = pos_ridge(Xtr, ytr, float(a))

        te = dec[dec["fold"] == k].dropna(subset=["alpha_beta"]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            Z = _xsz(g, fc)
            yv = g["alpha_beta"].to_numpy(float)
            if np.std(yv) <= 1e-12:
                continue
            if model == "signed_equal":
                sc = (Z * sign) @ np.ones(len(fc))
            elif model == "signed_shrunk":
                sc = Z @ shrunk                        # == Step-76 A_4h
            elif model == "ridge_xsz":
                sc = Z @ coef
            else:                                       # oriented + fitted coef
                sc = (Z * sign) @ coef
            if np.std(sc) <= 1e-12:
                continue
            rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                      "symbol": g["symbol"].to_numpy(),
                                      "score": sc, "y": yv}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


build_model_scores._cache = None


def diagnose(df: pd.DataFrame) -> dict:
    ic = df.groupby("open_time").apply(
        lambda g: g["score"].corr(g["y"], method="spearman")).dropna()
    dec = s77._binned(df, 10)
    rho = s77._mono(dec)
    ks = s77._ksweep(df, ks=(1, 3, 10))
    k3 = ks[ks["K"] == 3].iloc[0]
    # per-fold K=3 spread sign
    fp = 0
    for _, g in df.groupby("fold"):
        sp = []
        for _, gg in g.groupby("open_time"):
            if len(gg) < 6:
                continue
            s = gg.sort_values("score", ascending=False)
            sp.append(s.head(3)["y"].mean() - s.tail(3)["y"].mean())
        if sp and np.mean(sp) > 0:
            fp += 1
    n = len(ic)
    t = float(ic.mean() / (ic.std(ddof=1) / np.sqrt(n))) if n > 2 else np.nan
    passed = (rho >= GATE_RHO and k3["spread_bps"] >= GATE_SPREAD
              and fp >= GATE_FOLDS)
    return dict(ic_mean=float(ic.mean()), ic_t=t, decile_rho=float(rho),
                k1_spread=float(ks[ks["K"] == 1].iloc[0]["spread_bps"]),
                k3_spread=float(k3["spread_bps"]),
                k10_spread=float(ks[ks["K"] == 10].iloc[0]["spread_bps"]),
                k3_folds_pos=fp, gate_pass=bool(passed))


def main():
    print("=" * 96, flush=True)
    print("  STEP 78: constrained-linear payoff trial (NNLS / positive-Ridge)",
          flush=True)
    print(f"  PRE-REGISTERED GATE: decile rho>=+{GATE_RHO}  AND  K=3 spread "
          f">=+{GATE_SPREAD} bps  AND  >={GATE_FOLDS}/9 folds K3+ "
          f"(IC = context only)", flush=True)
    print("=" * 96, flush=True)
    t0 = time.time()
    models = ["ridge_xsz", "signed_equal", "signed_shrunk",
              "nnls_oriented", "posridge_oriented"]
    res = []
    for mdl in models:
        df = build_model_scores(mdl)
        if df.empty:
            print(f"  {mdl:20s} -> no scores", flush=True)
            continue
        d = diagnose(df)
        d["model"] = mdl
        res.append(d)
        tag = ""
        if mdl == "signed_shrunk":
            tag = ("  [sanity: must ~+0.0517 vs Step-76]"
                   if abs(d["ic_mean"] - 0.0517) < 0.004 else
                   f"  [!! sanity FAIL exp +0.0517 got {d['ic_mean']:+.4f}]")
        print(f"  {mdl:20s} IC={d['ic_mean']:+.4f} t={d['ic_t']:+5.2f} | "
              f"decile rho={d['decile_rho']:+.3f} | K1={d['k1_spread']:+6.2f} "
              f"K3={d['k3_spread']:+6.2f} K10={d['k10_spread']:+6.2f} bps | "
              f"K3 folds+={d['k3_folds_pos']}/9 | "
              f"{'PASS' if d['gate_pass'] else 'FAIL'}{tag}", flush=True)

    out = pd.DataFrame(res)[["model", "ic_mean", "ic_t", "decile_rho",
                             "k1_spread", "k3_spread", "k10_spread",
                             "k3_folds_pos", "gate_pass"]]
    out.to_csv(OUT / "summary.csv", index=False)

    constrained = out[out["model"].isin(["nnls_oriented", "posridge_oriented"])]
    rescued = bool(constrained["gate_pass"].any())
    print("\n" + "=" * 96, flush=True)
    print("  VERDICT", flush=True)
    print("=" * 96, flush=True)
    if rescued:
        v = ("A constrained model CLEARS the payoff gate -> linear-on-current-"
             "features is worth developing for that variant only; next = "
             "robustness/audit of it, still NO backtest.")
    else:
        v = ("Neither NNLS nor positive-Ridge clears the payoff gate (as "
             "predicted: 76/77 located the pathology in features->target, not "
             "the weighting). LINEAR-ON-CURRENT-FEATURES IS CLOSED. Next = "
             "Batch B only (price x volume interactions), dedupe-vs-history "
             "recorded, pre-registered payoff gate, no backtest until it "
             "clears.")
    print(f"  {v}", flush=True)
    pd.DataFrame([{"constrained_rescued": rescued, "verdict": v}]).to_csv(
        OUT / "verdict.csv", index=False)
    print(f"\nSaved under {OUT}\nTotal: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
