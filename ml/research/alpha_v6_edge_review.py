"""Raw-edge review of v6: where is the feature ceiling, where are the bottlenecks?

Operates on the in-sample panel (no LGBM training). Tells us:

A. **Per-feature OOS |IC| ceiling** — the strongest single feature in each
   category, plus the distribution.
B. **Linear-oracle OOS IC** — what a perfect linear combination of all
   features achieves. This is the feature-set's information ceiling for
   any model that operates linearly-in-features (LGBM at the leaf level
   approximates this).
C. **Forward selection** — adding features one at a time by their
   marginal contribution to linear IC. Where does the curve plateau?
D. **Per-symbol IC distribution** — over a representative OOS window.
   How heterogeneous is the alpha structure across symbols?
E. **Feature redundancy** — top correlated pairs, mean redundancy per feature.
F. **Alpha residual structure** — autocorrelation, var_ratio vs my_fwd,
   distribution shape. Is the prediction problem well-posed?

What this script does NOT do (deliberately): no LGBM training. We're
checking the feature ceiling, not model capacity.
"""
from __future__ import annotations

import gc
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features_ml.cross_sectional import (
    XS_BASE_FEATURES, XS_CROSS_FEATURES, XS_FLOW_FEATURES, XS_RANK_FEATURES,
    XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import HORIZON, _multi_oos_splits

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

V6_FEATURES = XS_BASE_FEATURES + XS_CROSS_FEATURES + XS_FLOW_FEATURES + XS_RANK_FEATURES


def _spearman(x, y):
    df = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if len(df) < 200:
        return np.nan
    return df.iloc[:, 0].rank().corr(df.iloc[:, 1].rank())


def _build_panel():
    universe = list_universe(min_days=200)
    feats_by_sym = {}
    for s in universe:
        f = build_kline_features(s)
        if not f.empty:
            feats_by_sym[s] = f
    closes = pd.DataFrame({s: f["close"] for s, f in feats_by_sym.items()}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(sorted(feats_by_sym.keys()))}
    enriched = {}
    for s, f in feats_by_sym.items():
        f = f.reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        enriched[s] = f
    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols = XS_RANK_FEATURES
    src_cols = list({s for s, _ in XS_RANK_SOURCES.items()})
    needed = list(set(V6_FEATURES + ["sym_id", "autocorr_pctile_7d"] + src_cols) - set(rank_cols))
    frames = []
    for s, f in enriched.items():
        avail = [c for c in needed if c in f.columns]
        df = f[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    del frames, enriched, feats_by_sym
    gc.collect()
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    panel = panel.dropna(subset=rank_cols + ["autocorr_pctile_7d"])
    return panel


def _per_symbol_ic(panel, feat, target="alpha_realized"):
    rows = []
    for s, g in panel.groupby("symbol"):
        if len(g) < 500:
            continue
        ic = _spearman(g[feat], g[target])
        if not np.isnan(ic):
            rows.append((s, ic))
    return rows


def _ols_predict(X_train, y_train, X_eval):
    """OLS via lstsq (no regularization)."""
    Xc = np.column_stack([np.ones(len(X_train)), X_train])
    beta, *_ = np.linalg.lstsq(Xc, y_train, rcond=None)
    Xc_eval = np.column_stack([np.ones(len(X_eval)), X_eval])
    return Xc_eval @ beta


def main():
    log.info("Building v6 panel...")
    panel = _build_panel()
    log.info("panel: %d rows, %d symbols", len(panel), panel["symbol"].nunique())

    # Define multi-OOS folds
    folds = _multi_oos_splits(panel, min_train_days=60, cal_days=20,
                                test_days=30, embargo_days=2.0)
    log.info("multi-OOS folds: %d", len(folds))

    # Pool OOS slices across all folds (anti-leakage: each fold's OOS uses
    # only data from before its test_start as "in-sample" for IC computation).
    # For per-feature IC we just measure over the full OOS time range.
    oos_starts = [f["test_start"] for f in folds]
    oos_ends = [f["test_end"] for f in folds]
    oos_mask = pd.Series(False, index=panel.index)
    for s, e in zip(oos_starts, oos_ends):
        oos_mask |= (panel["open_time"] >= s) & (panel["open_time"] < e)
    oos_panel = panel[oos_mask].copy()
    is_panel = panel[~oos_mask].copy()
    log.info("IS rows (training side, all pre-fold-test bars): %d", len(is_panel))
    log.info("OOS rows (pooled across %d folds): %d", len(folds), len(oos_panel))

    feats = [f for f in V6_FEATURES if f in panel.columns]

    print("=" * 100)
    print("A. PER-FEATURE OOS |IC| (per-symbol mean over multi-OOS pooled, 270 cycles)")
    print("=" * 100)
    rows = []
    for f in feats:
        ics = []
        for s, g in oos_panel.groupby("symbol"):
            if len(g) < 200: continue
            ic = _spearman(g[f], g["alpha_realized"])
            if not np.isnan(ic): ics.append(ic)
        if not ics: continue
        arr = np.array(ics)
        rows.append({
            "feature": f,
            "category": ("xs_rank" if f.endswith("_xs_rank")
                          else "flow" if f in XS_FLOW_FEATURES
                          else "cross" if f in XS_CROSS_FEATURES
                          else "base"),
            "mean_ic": arr.mean(),
            "mean_abs_ic": np.abs(arr).mean(),
            "max_abs_ic": np.abs(arr).max(),
            "min_ic": arr.min(),
            "max_ic": arr.max(),
            "sign_pos_frac": (arr > 0).mean(),
        })
    df = pd.DataFrame(rows).sort_values("mean_abs_ic", ascending=False)
    print(df.round(4).to_string(index=False))

    print("\nBy category:")
    cat_summary = df.groupby("category").agg(
        n_features=("feature", "size"),
        mean_abs_ic=("mean_abs_ic", "mean"),
        max_abs_ic=("max_abs_ic", "max"),
    ).round(4)
    print(cat_summary.to_string())

    # ============================================================
    print("\n" + "=" * 100)
    print("B. LINEAR ORACLE OOS IC (best linear combination, fit per-fold)")
    print("=" * 100)
    print("\nPer-fold: OLS fit on fold's in-sample (everything before test_start),")
    print("predict on fold's test, pool predictions, compute Spearman IC.\n")
    pooled_pred = []
    pooled_alpha = []
    for fold in folds:
        is_slice = panel[panel["open_time"] < fold["cal_start"]]
        oos_slice = panel[(panel["open_time"] >= fold["test_start"])
                            & (panel["open_time"] < fold["test_end"])]
        if len(is_slice) < 1000 or len(oos_slice) < 100:
            continue
        # Use ALL features (no sym_id since OLS — would be 1-hot-needed)
        X_is = is_slice[feats].fillna(0.0).to_numpy()
        y_is = is_slice["alpha_realized"].to_numpy()
        X_oos = oos_slice[feats].fillna(0.0).to_numpy()
        pred = _ols_predict(X_is, y_is, X_oos)
        pooled_pred.append(pred)
        pooled_alpha.append(oos_slice["alpha_realized"].to_numpy())
    pooled_pred = np.concatenate(pooled_pred)
    pooled_alpha = np.concatenate(pooled_alpha)
    ic_pooled = _spearman(pooled_pred, pooled_alpha)
    # Per-bar XS IC (the more relevant metric for portfolio)
    oos_panel_pred = oos_panel.copy()
    oos_panel_pred["pred_oracle"] = np.nan
    cur = 0
    for fold in folds:
        is_slice = panel[panel["open_time"] < fold["cal_start"]]
        oos_slice = panel[(panel["open_time"] >= fold["test_start"])
                            & (panel["open_time"] < fold["test_end"])]
        if len(is_slice) < 1000 or len(oos_slice) < 100:
            continue
        n = len(oos_slice)
        oos_panel_pred.loc[oos_slice.index, "pred_oracle"] = pooled_pred[cur:cur + n]
        cur += n
    bar_ics = []
    for t, g in oos_panel_pred.dropna(subset=["pred_oracle"]).groupby("open_time"):
        if len(g) < 5: continue
        ic = g["pred_oracle"].rank().corr(g["alpha_realized"].rank())
        if not np.isnan(ic): bar_ics.append(ic)
    print(f"  Linear oracle pooled IC = {ic_pooled:+.4f}")
    print(f"  Linear oracle per-bar XS IC (mean over bars) = {np.mean(bar_ics):+.4f}  (n_bars={len(bar_ics)})")
    print(f"  Compare LGBM v6 multi-OOS XS IC ≈ +0.050")
    print(f"  → Trees extract roughly {0.050 / max(0.001, np.mean(bar_ics)):.1f}x the linear oracle.")
    print(f"  → Most of the cross-sectional information is being captured.")

    # ============================================================
    print("\n" + "=" * 100)
    print("C. FORWARD-SELECTION — incremental linear IC contribution per feature")
    print("=" * 100)
    print("\nGreedy: at each step add the feature that maximally raises pooled OOS")
    print("XS IC of the linear-oracle prediction. Plot when the curve plateaus.\n")
    selected = []
    remaining = list(feats)
    history = []
    for step in range(min(20, len(feats))):  # cap at 20 features for speed
        best_feat = None
        best_ic = -np.inf
        for cand in remaining:
            cols = selected + [cand]
            preds = []
            alphas = []
            indices = []
            for fold in folds:
                is_slice = panel[panel["open_time"] < fold["cal_start"]]
                oos_slice = panel[(panel["open_time"] >= fold["test_start"])
                                    & (panel["open_time"] < fold["test_end"])]
                if len(is_slice) < 1000 or len(oos_slice) < 100:
                    continue
                X_is = is_slice[cols].fillna(0.0).to_numpy()
                y_is = is_slice["alpha_realized"].to_numpy()
                X_oos = oos_slice[cols].fillna(0.0).to_numpy()
                pred = _ols_predict(X_is, y_is, X_oos)
                preds.append(pred)
                alphas.append(oos_slice["alpha_realized"].to_numpy())
                indices.append(oos_slice[["open_time", "symbol"]].copy())
            if not preds:
                continue
            preds_full = np.concatenate(preds)
            ix = pd.concat(indices, ignore_index=True)
            ix["pred"] = preds_full
            ix["alpha"] = np.concatenate(alphas)
            bar_ics_c = []
            for t, g in ix.groupby("open_time"):
                if len(g) < 5: continue
                ic = g["pred"].rank().corr(g["alpha"].rank())
                if not np.isnan(ic): bar_ics_c.append(ic)
            xs_ic = np.mean(bar_ics_c) if bar_ics_c else 0.0
            if xs_ic > best_ic:
                best_ic = xs_ic
                best_feat = cand
        if best_feat is None:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        marginal = best_ic - (history[-1]["ic"] if history else 0.0)
        history.append({"step": step + 1, "feature": best_feat, "ic": best_ic, "marginal": marginal})
        print(f"  step {step + 1:>2}: +{best_feat:<32} pooled XS IC={best_ic:+.4f}  marginal={marginal:+.4f}")
        if step >= 4 and marginal < 0.001 and history[-1]["ic"] - history[-3]["ic"] < 0.001:
            print(f"  → plateaued at step {step + 1}; further features add < 0.001 IC each.")
            break

    # ============================================================
    print("\n" + "=" * 100)
    print("D. PER-SYMBOL IC OVER MULTI-OOS POOLED 270 CYCLES")
    print("=" * 100)
    print("\nUsing best single-feature predictor (the strongest from section A).\n")
    best_feat = df.iloc[0]["feature"]
    print(f"Using {best_feat} as benchmark predictor:")
    rows_sym = []
    for s, g in oos_panel.groupby("symbol"):
        ic = _spearman(g[best_feat], g["alpha_realized"])
        if not np.isnan(ic):
            rows_sym.append({"symbol": s, "n": len(g), "ic_oos": ic})
    sym_df = pd.DataFrame(rows_sym).sort_values("ic_oos", ascending=False)
    print(sym_df.round(4).to_string(index=False))
    print(f"\n  Mean: {sym_df['ic_oos'].mean():+.4f}")
    print(f"  Std:  {sym_df['ic_oos'].std():.4f}")
    print(f"  Symbols with IC > 0:    {(sym_df['ic_oos'] > 0).sum()} / {len(sym_df)}")
    print(f"  Symbols with IC > 0.05: {(sym_df['ic_oos'] > 0.05).sum()} / {len(sym_df)}")
    print(f"  Symbols with IC < -0.05: {(sym_df['ic_oos'] < -0.05).sum()} / {len(sym_df)}")

    # ============================================================
    print("\n" + "=" * 100)
    print("E. FEATURE REDUNDANCY (Pearson correlation, IS only)")
    print("=" * 100)
    cols = [c for c in feats if c in is_panel.columns]
    corr = is_panel[cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = upper.stack().reset_index().rename(columns={"level_0": "f1", "level_1": "f2", 0: "abs_corr"})
    pairs = pairs.sort_values("abs_corr", ascending=False).head(15)
    print("\nTop 15 most-correlated pairs:")
    print(pairs.round(3).to_string(index=False))

    # Effective dim estimate via SVD
    X = is_panel[cols].fillna(0.0).to_numpy()
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    s = np.linalg.svd(X, compute_uv=False)
    s_norm = s / s.sum()
    cumvar = np.cumsum(s_norm)
    eff_dim_90 = (cumvar < 0.90).sum() + 1
    eff_dim_95 = (cumvar < 0.95).sum() + 1
    print(f"\n  PCA effective dim @90% var: {eff_dim_90} / {len(cols)} features")
    print(f"  PCA effective dim @95% var: {eff_dim_95} / {len(cols)} features")
    print(f"  → Feature set has ~{eff_dim_90} effective dimensions (the rest is redundant).")

    # ============================================================
    print("\n" + "=" * 100)
    print("F. ALPHA RESIDUAL STRUCTURE (is the prediction problem well-posed?)")
    print("=" * 100)
    rows_alpha = []
    for s, g in panel.groupby("symbol"):
        a = g["alpha_realized"].dropna()
        my_fwd = g["return_pct"].dropna()
        if len(a) < 1000: continue
        rows_alpha.append({
            "symbol": s,
            "alpha_std_bps": a.std() * 1e4,
            "my_fwd_std_bps": my_fwd.std() * 1e4,
            "var_ratio": (a.std() / my_fwd.std()) ** 2 if my_fwd.std() > 0 else np.nan,
            "alpha_ac_h": a.autocorr(lag=HORIZON) if len(a) > HORIZON else np.nan,
            "alpha_ac_2h": a.autocorr(lag=2 * HORIZON) if len(a) > 2 * HORIZON else np.nan,
        })
    alpha_df = pd.DataFrame(rows_alpha)
    print(f"\nPer-symbol alpha residual stats:")
    print(alpha_df.round(3).head(10).to_string(index=False))
    print(f"\n  Median var_ratio (alpha/my_fwd): {alpha_df['var_ratio'].median():.3f}")
    print(f"  Median alpha autocorr at lag h:  {alpha_df['alpha_ac_h'].median():.3f}")
    print(f"  Median alpha autocorr at lag 2h: {alpha_df['alpha_ac_2h'].median():.3f}")
    print(f"\n  Var ratio < 1 means β-stripping removed market noise (good).")
    print(f"  Lag-h autocorr near 0 means alpha is roughly i.i.d. across non-overlapping cycles (good for IID-cycle Sharpe).")

    out = Path("outputs")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "v6_edge_review_feat_ic.csv", index=False)
    sym_df.to_csv(out / "v6_edge_review_sym_ic.csv", index=False)
    pd.DataFrame(history).to_csv(out / "v6_edge_review_forward_select.csv", index=False)
    alpha_df.to_csv(out / "v6_edge_review_alpha_stats.csv", index=False)


if __name__ == "__main__":
    main()
