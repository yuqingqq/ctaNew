"""Step 86: diagnose the Step-84 24h sqbtcrel Ridge edge.

Step 84 showed a strong 24h ridge_xsz/sqbtcrel result, but it failed the
estimator-robustness gate because signed_equal and NNLS did not agree. This
script does not backtest. It asks the narrower diagnostic question:

  Is the 24h payoff supported by stable feature contributions, or is it a
  fragile collinear Ridge fit?

Outputs:
  - variant_summary.csv: ridge alpha/grid, group, and leave-one variants.
  - coefficients_by_fold.csv: fold coefficients for the reference RidgeCV.
  - feature_stability.csv: sign stability, coefficient dispersion, and
    leave-one impact per feature.
  - verdict.csv: data-driven interpretation.
"""
from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(n: str, r: str):
    s = importlib.util.spec_from_file_location(n, REPO / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


s76 = _imp("s76", "linear_model/scripts/76_minimal_orientation.py")
s78 = _imp("s78", "linear_model/scripts/78_nnls_poscoef_payoff.py")
s79 = _imp("s79", "linear_model/scripts/79_broader_universe_attrib.py")
s80b = _imp("s80b", "linear_model/scripts/80b_vol_interaction_payoff.py")
s84 = _imp("s84", "linear_model/scripts/84_proper_24h_rigor.py")
from ml.research.alpha_v4_xs_1d import _slice

OUT = REPO / "linear_model/results/step86_ridge_24h_diagnostic"
OUT.mkdir(parents=True, exist_ok=True)

OOS = s76.OOS
ALPHAS_WIDE = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0,
               1_000.0, 10_000.0]
SQ = s80b.SQ
BTCREL = s80b.BTCREL
REF = SQ + BTCREL


def _train_matrix(tr: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, g in tr.dropna(subset=["target_z"]).groupby("open_time"):
        if len(g) < 5:
            continue
        z = s78._xsz(g, features)
        y = g["target_z"].to_numpy(float)
        if np.isfinite(y).all() and np.std(y) > 1e-12:
            xs.append(z)
            ys.append(y)
    if not xs:
        return np.empty((0, len(features))), np.empty(0)
    return np.vstack(xs), np.concatenate(ys)


def _score_variant(dec: pd.DataFrame, folds: list, features: list[str],
                   mode: str, alpha: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, coef_rows = [], []
    features = [c for c in features if c in dec.columns]
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0].dropna(subset=["alpha_beta"])
        if len(tr) < 500:
            continue
        xtr, ytr = _train_matrix(tr, features)
        if len(ytr) < 500:
            continue
        if mode == "ridge_cv":
            mdl = RidgeCV(alphas=ALPHAS_WIDE, scoring="r2", fit_intercept=False)
            mdl.fit(xtr, ytr)
            coef = mdl.coef_.astype(float)
            chosen_alpha = float(mdl.alpha_)
        elif mode == "ridge_fixed":
            mdl = Ridge(alpha=float(alpha), fit_intercept=False)
            mdl.fit(xtr, ytr)
            coef = mdl.coef_.astype(float)
            chosen_alpha = float(alpha)
        elif mode == "signed_equal":
            w = s76.fit_weights(tr, features, "alpha_beta")
            coef = np.array([np.sign(w[f]) or 1.0 for f in features], float)
            chosen_alpha = np.nan
        else:
            raise ValueError(mode)
        coef_rows.extend({"fold": k, "feature": f, "coef": c,
                          "alpha": chosen_alpha}
                         for f, c in zip(features, coef))
        te = dec[dec["fold"] == k].dropna(subset=["alpha_beta"]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            z = s78._xsz(g, features)
            y = g["alpha_beta"].to_numpy(float)
            if np.std(y) <= 1e-12:
                continue
            sc = z @ coef
            if np.std(sc) <= 1e-12:
                continue
            rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                      "symbol": g["symbol"].to_numpy(),
                                      "score": sc, "y": y}))
    scored = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    coefs = pd.DataFrame(coef_rows)
    return scored, coefs


def _score_group_model(dec: pd.DataFrame, folds: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit RidgeCV on two group scores: mean xsz(SQ), mean xsz(BTCREL)."""
    rows, coef_rows = [], []
    groups = {"sq": [c for c in SQ if c in dec.columns],
              "btcrel": [c for c in BTCREL if c in dec.columns]}
    for k in OOS:
        if k >= len(folds):
            continue
        tr = _slice(dec, folds[k])[0].dropna(subset=["alpha_beta"])
        gx, gy = [], []
        for _, g in tr.dropna(subset=["target_z"]).groupby("open_time"):
            if len(g) < 5:
                continue
            cols = []
            for fs in groups.values():
                cols.append(s78._xsz(g, fs).mean(axis=1))
            y = g["target_z"].to_numpy(float)
            if np.std(y) > 1e-12:
                gx.append(np.column_stack(cols))
                gy.append(y)
        if not gx:
            continue
        xtr, ytr = np.vstack(gx), np.concatenate(gy)
        mdl = RidgeCV(alphas=ALPHAS_WIDE, scoring="r2", fit_intercept=False).fit(xtr, ytr)
        coef = mdl.coef_.astype(float)
        coef_rows.extend({"fold": k, "feature": name, "coef": c,
                          "alpha": float(mdl.alpha_)}
                         for name, c in zip(groups, coef))
        te = dec[dec["fold"] == k].dropna(subset=["alpha_beta"]).copy()
        for t, g in te.groupby("open_time", sort=True):
            if len(g) < 5:
                continue
            cols = [s78._xsz(g, fs).mean(axis=1) for fs in groups.values()]
            y = g["alpha_beta"].to_numpy(float)
            sc = np.column_stack(cols) @ coef
            if np.std(y) > 1e-12 and np.std(sc) > 1e-12:
                rows.append(pd.DataFrame({"open_time": t, "fold": k,
                                          "symbol": g["symbol"].to_numpy(),
                                          "score": sc, "y": y}))
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(),
            pd.DataFrame(coef_rows))


def _metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"net_sharpe": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "net_bps_cyc": np.nan, "gross_k3_bps": np.nan,
                "decile_rho": np.nan, "folds_pos": 0, "ic_mean": np.nan}
    net = s84.ls_series(df)
    sh, lo, hi, mu = s84.sharpe_ci(net["net"])
    d24 = df[df["open_time"].isin(set(sorted(df["open_time"].unique())[::s84.H24]))]
    rho = s79.payoff(d24)["decile_rho"] if not d24.empty else np.nan
    ic = df.groupby("open_time").apply(
        lambda g: g["score"].corr(g["y"], method="spearman")).dropna()
    folds_pos = sum(1 for _, g in net.groupby("fold") if g["net"].mean() > 0)
    return {"net_sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "net_bps_cyc": mu, "gross_k3_bps": mu + s84.COST_BPS,
            "decile_rho": float(rho), "folds_pos": int(folds_pos),
            "ic_mean": float(ic.mean()) if len(ic) else np.nan}


def main():
    print("=" * 100, flush=True)
    print("  STEP 86: 24h sqbtcrel Ridge diagnostic — coefficients, shrinkage, leave-one",
          flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    _, _, dec, _, folds, _ = s84.build_dec()
    ref = [c for c in REF if c in dec.columns]
    print(f"  dec: {dec['symbol'].nunique()} syms, {dec['open_time'].nunique()} cycles; "
          f"features={len(ref)}", flush=True)

    rows = []
    coef_ref = pd.DataFrame()

    # Reference and alpha sensitivity.
    for mode, alpha, name in [("ridge_cv", None, "sqbtcrel_ridge_cv"),
                              ("signed_equal", None, "sqbtcrel_signed_equal")]:
        df, co = _score_variant(dec, folds, ref, mode, alpha)
        if name == "sqbtcrel_ridge_cv":
            coef_ref = co.copy()
        rows.append({"variant": name, "n_feat": len(ref), **_metrics(df)})
        print(f"  {name:26s} netSh={rows[-1]['net_sharpe']:+.2f} "
              f"CI[{rows[-1]['ci_lo']:+.2f},{rows[-1]['ci_hi']:+.2f}] "
              f"grossK3={rows[-1]['gross_k3_bps']:+.1f} "
              f"rho={rows[-1]['decile_rho']:+.3f} f+={rows[-1]['folds_pos']}/9",
              flush=True)

    for a in [0.1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0]:
        df, _ = _score_variant(dec, folds, ref, "ridge_fixed", a)
        rows.append({"variant": f"sqbtcrel_ridge_alpha_{a:g}", "n_feat": len(ref),
                     **_metrics(df)})
        print(f"  ridge alpha={a:<8g}        netSh={rows[-1]['net_sharpe']:+.2f} "
              f"grossK3={rows[-1]['gross_k3_bps']:+.1f} "
              f"rho={rows[-1]['decile_rho']:+.3f} f+={rows[-1]['folds_pos']}/9",
              flush=True)

    # Group compression.
    for name, fs in [("only_squared", SQ), ("only_btcrel", BTCREL)]:
        fs = [c for c in fs if c in dec.columns]
        df, _ = _score_variant(dec, folds, fs, "ridge_cv")
        rows.append({"variant": name, "n_feat": len(fs), **_metrics(df)})
        print(f"  {name:26s} netSh={rows[-1]['net_sharpe']:+.2f} "
              f"grossK3={rows[-1]['gross_k3_bps']:+.1f} "
              f"rho={rows[-1]['decile_rho']:+.3f} f+={rows[-1]['folds_pos']}/9",
              flush=True)
    df, co_group = _score_group_model(dec, folds)
    rows.append({"variant": "two_group_ridge", "n_feat": 2, **_metrics(df)})
    print(f"  {'two_group_ridge':26s} netSh={rows[-1]['net_sharpe']:+.2f} "
          f"grossK3={rows[-1]['gross_k3_bps']:+.1f} "
          f"rho={rows[-1]['decile_rho']:+.3f} f+={rows[-1]['folds_pos']}/9",
          flush=True)

    # Leave-one feature sensitivity.
    base_net = float(rows[0]["net_sharpe"])
    base_gross = float(rows[0]["gross_k3_bps"])
    loo = []
    for f in ref:
        fs = [x for x in ref if x != f]
        df, _ = _score_variant(dec, folds, fs, "ridge_cv")
        m = _metrics(df)
        loo.append({"feature": f, "loo_net_sharpe": m["net_sharpe"],
                    "loo_gross_k3_bps": m["gross_k3_bps"],
                    "loo_decile_rho": m["decile_rho"],
                    "delta_net_sharpe_vs_ref": m["net_sharpe"] - base_net,
                    "delta_gross_k3_vs_ref": m["gross_k3_bps"] - base_gross})
        rows.append({"variant": f"drop_{f}", "n_feat": len(fs), **m})
        print(f"  drop {f:24s} netSh={m['net_sharpe']:+.2f} "
              f"grossK3={m['gross_k3_bps']:+.1f} "
              f"Δgross={m['gross_k3_bps'] - base_gross:+.1f}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "variant_summary.csv", index=False)
    coef_ref.to_csv(OUT / "coefficients_by_fold.csv", index=False)
    co_group.to_csv(OUT / "group_coefficients_by_fold.csv", index=False)

    if coef_ref.empty:
        stability = pd.DataFrame()
    else:
        stability = (coef_ref.groupby("feature")["coef"]
                     .agg(coef_mean="mean", coef_std="std",
                          coef_min="min", coef_max="max")
                     .reset_index())
        stability["sign_pos_folds"] = stability["feature"].map(
            coef_ref.groupby("feature")["coef"].apply(lambda s: int((s > 0).sum())))
        stability["sign_neg_folds"] = stability["feature"].map(
            coef_ref.groupby("feature")["coef"].apply(lambda s: int((s < 0).sum())))
        stability["abs_mean_over_std"] = (
            stability["coef_mean"].abs() / stability["coef_std"].replace(0, np.nan))
        loo_df = pd.DataFrame(loo)
        stability = stability.merge(loo_df, on="feature", how="left")
    stability.to_csv(OUT / "feature_stability.csv", index=False)

    best = summary.sort_values("net_sharpe", ascending=False).iloc[0]
    ref_row = summary[summary["variant"] == "sqbtcrel_ridge_cv"].iloc[0]
    signed = summary[summary["variant"] == "sqbtcrel_signed_equal"].iloc[0]
    alpha_ok = summary[summary["variant"].str.startswith("sqbtcrel_ridge_alpha_")]
    stable_alpha = int((alpha_ok["net_sharpe"] > 0).sum())
    critical = stability.sort_values("delta_gross_k3_vs_ref").head(3) if not stability.empty else pd.DataFrame()
    crit_txt = "; ".join(
        f"{r.feature}: drop changes gross K3 {r.delta_gross_k3_vs_ref:+.1f}bps"
        for r in critical.itertuples())
    verdict = (
        f"Reference ridge_cv netSh {ref_row.net_sharpe:+.2f}, gross K3 "
        f"{ref_row.gross_k3_bps:+.1f}bps, rho {ref_row.decile_rho:+.3f}. "
        f"Signed_equal netSh {signed.net_sharpe:+.2f}; {stable_alpha}/6 fixed "
        f"ridge alphas stay positive. Best variant is {best.variant} "
        f"(netSh {best.net_sharpe:+.2f}). Leave-one sensitivity: {crit_txt}."
    )
    pd.DataFrame([{"verdict": verdict}]).to_csv(OUT / "verdict.csv", index=False)
    print("\n" + "=" * 100, flush=True)
    print(f"  VERDICT: {verdict}", flush=True)
    print(f"\nSaved under {OUT}\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
