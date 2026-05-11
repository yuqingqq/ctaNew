"""Quick test: single basket-vol regime feature added to v6_clean.

DVOL (8 broadcast features): -1.86 ΔSharpe.
Question: does a SINGLE carefully-chosen regime feature behave differently?

Test:
  Feature: basket_vol_30d (rolling 30d basket realized vol, annualized)
  Variants:
    A. v6_clean baseline (28 features)
    B. v6_clean + basket_vol_30d (29 features, direct LGBM)
    C. v6_clean + basket_vol_30d + basket_vol_30d_pctile (30 features)
    D. Hybrid: LGBM(v6_clean) + Ridge(basket_vol_30d) blended

(D) is interesting because broadcast features in a separate ridge head
contribute a CONSTANT per bar to all symbols, which cancels in cross-
sectional ranking → ridge head will have zero contribution. We test it
to confirm the structural argument.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import (
    XS_FEATURE_COLS_V6_CLEAN, XS_RANK_SOURCES,
    add_basket_features, add_engineered_flow_features, add_xs_rank_features,
    build_basket, build_kline_features, list_universe, make_xs_alpha_labels,
)
from ml.research.alpha_v4_xs_1d import (
    ENSEMBLE_SEEDS, _multi_oos_splits, _slice, _train,
)
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_conviction_v2 import evaluate_portfolio

HORIZON = 48
TOP_K = 7
COST_PER_LEG = 4.5
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
GATE_PCTILE = 0.30
NEW_SYMBOLS = {"ETCUSDT", "HBARUSDT", "ICPUSDT", "LDOUSDT", "TRBUSDT",
                "AAVEUSDT", "MKRUSDT", "AXSUSDT", "GMXUSDT",
                "1000PEPEUSDT", "1000SHIBUSDT", "TONUSDT", "ORDIUSDT", "WIFUSDT"}
OUT_DIR = REPO / "outputs/h48_regime_feature"
OUT_DIR.mkdir(parents=True, exist_ok=True)
sharpe_est = lambda x: x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR) if x.std() > 0 else 0


def fit_predict_ridge(X_tr, y_tr, X_te, alpha=1.0):
    sc = StandardScaler()
    Xs = sc.fit_transform(np.nan_to_num(X_tr, nan=0.0))
    Xte = sc.transform(np.nan_to_num(X_te, nan=0.0))
    Xs = np.nan_to_num(Xs, nan=0.0); Xte = np.nan_to_num(Xte, nan=0.0)
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(Xs, y_tr)
    return m.predict(Xte)


def z(p):
    s = p.std()
    return (p - p.mean()) / (s if s > 1e-8 else 1.0)


def build_panel():
    universe_full = sorted(list_universe(min_days=200))
    orig25 = sorted([s for s in universe_full if s not in NEW_SYMBOLS])
    print(f"Building panel for {len(orig25)} ORIG25 syms…", flush=True)
    feats = {s: build_kline_features(s) for s in orig25}
    closes = pd.DataFrame({s: feats[s]["close"] for s in orig25}).sort_index()
    basket_ret, basket_close = build_basket(closes)
    sym_to_id = {s: i for i, s in enumerate(orig25)}

    # Compute REGIME features per BAR (broadcast — same value for all 25 syms at time t)
    # 4h-cadence basket returns
    bk_close_4h = basket_close.resample("4h").last().ffill()
    bk_ret_4h = bk_close_4h.pct_change()
    # Rolling 30d annualized vol (180 4h bars = 30 days)
    basket_vol_30d_4h = bk_ret_4h.rolling(180, min_periods=42).std() * np.sqrt(2190)
    basket_vol_30d_4h = basket_vol_30d_4h.shift(1)  # PIT

    # Rolling percentile of vol (252-day = 1512 4h bars = 1y trailing)
    basket_vol_pctile_4h = basket_vol_30d_4h.rolling(1512, min_periods=180).rank(pct=True)

    # Reindex to 5min cadence (panel uses 5min bars)
    enriched = {}
    for s in orig25:
        f = feats[s].reindex(closes.index)
        f = add_basket_features(f, basket_close, basket_ret)
        f = add_engineered_flow_features(f)
        f["sym_id"] = sym_to_id[s]
        if f.index.tz is None:
            f.index = f.index.tz_localize("UTC")
        # Map 4h-cadence regime values to 5min via ffill
        # bk_close_4h was resampled FROM 5min closes; reindex back to f.index
        bv30d_5min = basket_vol_30d_4h.reindex(f.index, method="ffill")
        bvpct_5min = basket_vol_pctile_4h.reindex(f.index, method="ffill")
        f["basket_vol_30d"] = bv30d_5min
        f["basket_vol_30d_pctile"] = bvpct_5min
        enriched[s] = f

    labels = make_xs_alpha_labels(enriched, basket_close, HORIZON)
    rank_cols_v6 = [c for c in XS_FEATURE_COLS_V6_CLEAN if c.endswith("_xs_rank")]
    src_cols_v6 = list({src for src, dst in XS_RANK_SOURCES.items() if dst in rank_cols_v6})
    new_cols = ["basket_vol_30d", "basket_vol_30d_pctile"]
    needed = list(set(list(XS_FEATURE_COLS_V6_CLEAN)
                        + ["sym_id", "autocorr_pctile_7d", "beta_short_vs_bk"]
                        + src_cols_v6 + new_cols) - set(rank_cols_v6))
    frames = []
    for s, ff in enriched.items():
        avail = [c for c in needed if c in ff.columns]
        df = ff[avail].join(labels[s], how="inner")
        df["symbol"] = s
        df = df.reset_index().rename(columns={"index": "open_time"})
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = add_xs_rank_features(panel, sources=XS_RANK_SOURCES)
    for c in rank_cols_v6:
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")
    panel = panel.dropna(subset=list(XS_FEATURE_COLS_V6_CLEAN)
                            + ["autocorr_pctile_7d", "demeaned_target", "return_pct"])
    print(f"  panel: {len(panel):,} rows", flush=True)
    print(f"  basket_vol_30d: mean {panel['basket_vol_30d'].mean():.3f}, "
          f"std {panel['basket_vol_30d'].std():.3f}, "
          f"non-null {panel['basket_vol_30d'].notna().sum():,}", flush=True)
    return panel


def main():
    panel = build_panel()
    folds = _multi_oos_splits(panel)
    v6_clean = list(XS_FEATURE_COLS_V6_CLEAN)
    print(f"Folds: {len(folds)}", flush=True)

    cycles = {"A_baseline": [], "B_v6+vol": [], "C_v6+vol+pctile": [],
               "D_hybrid_v6+ridge_vol": []}
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200:
            continue

        avail_v6 = [c for c in v6_clean if c in panel.columns]
        # Variant A: baseline
        Xt = tr[avail_v6].to_numpy(dtype=np.float32)
        yt_ = tr["demeaned_target"].to_numpy(dtype=np.float32)
        Xc = ca[avail_v6].to_numpy(dtype=np.float32)
        yc_ = ca["demeaned_target"].to_numpy(dtype=np.float32)
        Xtest = test[avail_v6].to_numpy(dtype=np.float32)
        models_A = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        pred_A = np.mean([m.predict(Xtest, num_iteration=m.best_iteration)
                            for m in models_A], axis=0)

        # Variant B: v6 + basket_vol_30d
        cols_B = avail_v6 + ["basket_vol_30d"]
        Xt = tr[cols_B].to_numpy(dtype=np.float32)
        Xc = ca[cols_B].to_numpy(dtype=np.float32)
        Xtest_B = test[cols_B].to_numpy(dtype=np.float32)
        models_B = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        pred_B = np.mean([m.predict(Xtest_B, num_iteration=m.best_iteration)
                            for m in models_B], axis=0)

        # Variant C: v6 + vol + pctile
        cols_C = avail_v6 + ["basket_vol_30d", "basket_vol_30d_pctile"]
        Xt = tr[cols_C].to_numpy(dtype=np.float32)
        Xc = ca[cols_C].to_numpy(dtype=np.float32)
        Xtest_C = test[cols_C].to_numpy(dtype=np.float32)
        models_C = [_train(Xt, yt_, Xc, yc_, seed=seed) for seed in ENSEMBLE_SEEDS]
        pred_C = np.mean([m.predict(Xtest_C, num_iteration=m.best_iteration)
                            for m in models_C], axis=0)

        # Variant D: hybrid LGBM(v6) + Ridge(broadcast vol features)
        avail_regime = ["basket_vol_30d", "basket_vol_30d_pctile"]
        X_full_r = np.vstack([tr[avail_regime].to_numpy(dtype=np.float64),
                                ca[avail_regime].to_numpy(dtype=np.float64)])
        y_full = np.concatenate([yt_.astype(np.float64), yc_.astype(np.float64)])
        Xtest_r = test[avail_regime].to_numpy(dtype=np.float64)
        ridge_pred = fit_predict_ridge(X_full_r, y_full, Xtest_r)
        pred_D = 0.9 * z(pred_A) + 0.1 * z(ridge_pred)

        for tag, pred in [("A_baseline", pred_A), ("B_v6+vol", pred_B),
                            ("C_v6+vol+pctile", pred_C), ("D_hybrid_v6+ridge_vol", pred_D)]:
            df = evaluate_portfolio(test, pred, use_gate=True, gate_pctile=GATE_PCTILE,
                                     use_magweight=False, top_k=TOP_K)
            for _, r in df.iterrows():
                cycles[tag].append({"fold": fold["fid"], "time": r["time"],
                                      "net": r["net_bps"], "skipped": r["skipped"],
                                      "gross": r["spread_ret_bps"], "cost": r["cost_bps"]})
        print(f"  fold {fold['fid']}: {time.time() - t0:.0f}s", flush=True)

    print("\n" + "=" * 110, flush=True)
    print("REGIME FEATURE TEST", flush=True)
    print("=" * 110, flush=True)
    print(f"  {'variant':<32} {'gross':>7} {'cost':>6} {'net':>7} {'Sharpe':>7} {'95% CI':>15} {'Δbase':>7}", flush=True)
    base_sh = sharpe_est(np.array([r["net"] for r in cycles["A_baseline"]]))
    summary = {}
    for tag in ["A_baseline", "B_v6+vol", "C_v6+vol+pctile", "D_hybrid_v6+ridge_vol"]:
        df = pd.DataFrame(cycles[tag])
        traded = df[df["skipped"] == 0]
        sh, lo, hi = block_bootstrap_ci(df["net"].values, statistic=sharpe_est,
                                          block_size=7, n_boot=2000)
        d = sh - base_sh
        print(f"  {tag:<32} "
              f"{traded['gross'].mean() if len(traded) > 0 else 0:>+6.2f}  "
              f"{traded['cost'].mean() if len(traded) > 0 else 0:>5.2f}  "
              f"{df['net'].mean():>+6.2f}  "
              f"{sh:>+6.2f}  [{lo:>+5.2f},{hi:>+5.2f}]  {d:>+6.2f}", flush=True)
        summary[tag] = {"n_cycles": int(len(df)), "net": float(df["net"].mean()),
                          "sharpe": float(sh), "ci": [float(lo), float(hi)],
                          "delta_vs_baseline": float(d)}

    with open(OUT_DIR / "alpha_v9_regime_feature_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
