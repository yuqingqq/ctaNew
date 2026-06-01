"""X16 — Proper fix for crossX/aggT coverage gap issue.

Root cause (X15 diagnostic): aggT is BINARY per-symbol (25 syms 97% / 26 syms 0%);
crossX is partial (~63% coverage with sym variability). Pool+symid Ridge fits ONE
shared coefficient → coverage gaps dilute the coefficient → signal lost.

Fix variants to test:
  F1: HL-50 minus zero-aggT syms = 25-sym subset → LGBM Pool+symid +aggT
  F2: HL-50 minus low-crossX-coverage syms (drop syms with <80% crossX coverage)
      → Ridge Pool+symid +crossX (C1)
  F3: Subset training: Pool+symid Ridge trained ONLY on rows where crossX is non-NaN
      (same universe, but drop NaN rows from training)
  F4: Add has_aggT indicator feature: Ridge learns separate intercepts for
      cover/no-cover symbols

Compare to:
  - Per-sym Ridge +crossX = +1.12 (X7, the architecture that works)
  - Per-sym Ridge +aggT = +0.45 (X7)
  - Pool+symid Ridge +crossX = +0.43 (baseline, our problem)
  - Pool+symid Ridge +aggT = +1.22 (X6)
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)

WIDER_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())

LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=31, max_depth=6, min_data_in_leaf=300,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, verbose=-1, n_estimators=400,
)


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def load_panel(universe_syms):
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE + x6.AGGT_EXTRAS)
    p = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                        columns=list(set(needed)))
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p[p["symbol"].isin(universe_syms) & (p["symbol"] != "BTCUSDT")].copy()
    # crossX
    cross_path = REPO / "data/ml/cache/cross_exchange_features.parquet"
    cross_df = pd.read_parquet(cross_path)
    cross_df["open_time"] = pd.to_datetime(cross_df["open_time"], utc=True)
    z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    p = p.merge(cross_df[["symbol", "open_time"] + z_cols],
                 on=["symbol", "open_time"], how="left")
    p = x6.build_target_z(p)
    for c in p.columns:
        if p[c].dtype in ("float64",): p[c] = p[c].astype("float32")
    for c in z_cols: x6.HEAVY_TAIL.add(c)
    return p, z_cols


def get_sym_dum_norm(panel):
    syms_sorted = sorted(panel["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(syms_sorted)}
    sym_codes = panel["symbol"].map(sym_idx).to_numpy(np.int32)
    n_dum = len(syms_sorted) - 1
    dum = np.zeros((len(panel), n_dum), dtype=np.float32)
    m = sym_codes > 0
    dum[m, sym_codes[m] - 1] = 1.0
    return dum


def train_ridge_c1(panel, folds, feats, drop_nan_rows=False):
    """Ridge Pool+symid with C1 (norm sym_id, wider α). Optionally drop NaN-in-feats rows."""
    dum_all = get_sym_dum_norm(panel)
    all_preds = []
    sstats0 = hstats0 = None
    for f, ts, te, ec in folds:
        train_mask = ((panel["exit_time"] < ec).to_numpy()
                      & panel["target_z"].notna().to_numpy())
        test_mask = ((panel["open_time"] >= ts) & (panel["open_time"] <= te)).to_numpy()

        if drop_nan_rows:
            # Drop training rows where ANY of the test features is NaN
            feat_nan_mask = panel[feats].isna().any(axis=1).to_numpy()
            train_mask = train_mask & ~feat_nan_mask

        train = panel.iloc[train_mask]
        test = panel.iloc[test_mask]
        if len(train) < 5000 or len(test) < 1000: continue

        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, feats)
        Xtr = x6.apply_preproc(train, feats, sstats0, hstats0).astype(np.float32)
        Xte = x6.apply_preproc(test, feats, sstats0, hstats0).astype(np.float32)
        ytr = train["target_z"].to_numpy(np.float32)

        dum_tr = dum_all[train_mask]
        dum_te = dum_all[test_mask]
        mean = dum_tr.mean(axis=0); std = dum_tr.std(axis=0); std[std == 0] = 1.0
        dum_tr_n = ((dum_tr - mean) / std).astype(np.float32)
        dum_te_n = ((dum_te - mean) / std).astype(np.float32)
        X_train = np.hstack([Xtr, dum_tr_n]); X_test = np.hstack([Xte, dum_te_n])

        m = RidgeCV(alphas=WIDER_GRID).fit(X_train, ytr)
        pred = m.predict(X_test).astype(np.float32)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred; out["fold"] = f
        all_preds.append(out)
        del Xtr, Xte, X_train, X_test, ytr, m, dum_tr_n, dum_te_n
        gc.collect()
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])


def train_lgbm_pool_symid(panel, folds, feats_no_symid, drop_nan_rows=False):
    syms_sorted = sorted(panel["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(syms_sorted)}
    panel = panel.copy()
    panel["sym_id"] = panel["symbol"].map(sym_map).astype("int32")
    feats = feats_no_symid + ["sym_id"]
    all_preds = []
    for f, ts, te, ec in folds:
        train = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        if drop_nan_rows:
            train = train.dropna(subset=feats_no_symid)
        if len(train) < 5000 or len(test) < 1000: continue
        m = lgb.LGBMRegressor(random_state=20260520, **LGB_PARAMS)
        m.fit(train[feats], train["target_z"].to_numpy(np.float32), categorical_feature=["sym_id"])
        pred = m.predict(test[feats]).astype(np.float32)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized",
                    "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred; out["fold"] = f
        all_preds.append(out)
        del m; gc.collect()
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])


def main():
    t0 = time.time()
    print("=== X16 subset universe / drop-nan-rows test ===\n", flush=True)

    # Identify universe subsets
    panel_full, crossX_cols = load_panel(HL_SYMS)
    log_mem("after full load")

    # Per-sym aggT coverage
    aggT_cov = panel_full.groupby("symbol")["aggr_ratio_4h"].apply(lambda x: x.notna().mean())
    syms_with_aggT = set(aggT_cov[aggT_cov > 0.5].index)
    print(f"Syms with aggT > 50% coverage: {len(syms_with_aggT)} of {len(aggT_cov)}")

    # Per-sym crossX coverage — measured ONLY on 4h-aligned bars (where crossX exists)
    is_4h_aligned = ((panel_full["open_time"].dt.hour % 4 == 0)
                     & (panel_full["open_time"].dt.minute == 0))
    panel_4h = panel_full[is_4h_aligned]
    cx_cov = panel_4h.groupby("symbol")[crossX_cols[0]].apply(lambda x: x.notna().mean())
    syms_with_crossX = set(cx_cov[cx_cov > 0.5].index)
    print(f"Syms with crossX > 50% coverage (at 4h-aligned bars): {len(syms_with_crossX)} of {len(cx_cov)}")

    # Common universe (intersection)
    syms_with_both = syms_with_aggT & syms_with_crossX
    print(f"Syms with BOTH aggT+crossX > 50%: {len(syms_with_both)}")

    del panel_full; gc.collect()
    log_mem("after subset analysis")

    results = []
    feats_aggT = x6.BASE + x6.AGGT_EXTRAS
    feats_crossX = x6.BASE + crossX_cols

    # === F1: LGBM Pool+symid +aggT on syms-with-aggT subset (25 syms) ===
    print(f"\n[F1] LGBM Pool+symid +aggT on {len(syms_with_aggT)}-sym subset (aggT coverage)")
    tf = time.time()
    panel_a, _ = load_panel(syms_with_aggT)
    folds = x6.get_folds(panel_a)
    apd = train_lgbm_pool_symid(panel_a, folds, feats_aggT)
    pred_path = CACHE / "x16_F1_lgbm_pool_aggT_subset.parquet"
    apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    m = x6.run_sleeve_on_preds(pred_path, "x16_F1")
    print(f"  trained: {len(apd):,} rows, IC={ic:+.4f}, "
          f"Sharpe={m.get('sharpe', '?'):+.2f} folds={m.get('folds_pos','?')} "
          f"conc={m.get('concentration','?')} [{time.time()-tf:.0f}s]", flush=True)
    results.append({"variant": "F1_lgbm_pool_aggT_subset",
                    "desc": f"LGBM Pool+symid +aggT on {len(syms_with_aggT)}-sym subset",
                    "n_syms": len(syms_with_aggT), "train_ic": round(ic, 4), **m})
    del panel_a, apd; gc.collect()
    log_mem("after F1")

    # === F2: Ridge Pool+symid +crossX C1 on syms-with-crossX subset ===
    print(f"\n[F2] Ridge Pool+symid +crossX C1 on {len(syms_with_crossX)}-sym subset (crossX coverage)")
    tf = time.time()
    panel_c, _ = load_panel(syms_with_crossX)
    folds = x6.get_folds(panel_c)
    apd = train_ridge_c1(panel_c, folds, feats_crossX, drop_nan_rows=False)
    pred_path = CACHE / "x16_F2_ridge_pool_crossX_subset.parquet"
    apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    m = x6.run_sleeve_on_preds(pred_path, "x16_F2")
    print(f"  trained: {len(apd):,} rows, IC={ic:+.4f}, "
          f"Sharpe={m.get('sharpe', '?'):+.2f} folds={m.get('folds_pos','?')} "
          f"conc={m.get('concentration','?')} [{time.time()-tf:.0f}s]", flush=True)
    results.append({"variant": "F2_ridge_pool_crossX_subset",
                    "desc": f"Ridge Pool+symid +crossX C1 on {len(syms_with_crossX)}-sym subset",
                    "n_syms": len(syms_with_crossX), "train_ic": round(ic, 4), **m})
    del panel_c, apd; gc.collect()
    log_mem("after F2")

    # === F3: Ridge Pool+symid +crossX C1, full universe, DROP NaN training rows ===
    print(f"\n[F3] Ridge Pool+symid +crossX C1 on FULL universe, DROP NaN training rows")
    tf = time.time()
    panel_full, _ = load_panel(HL_SYMS)
    folds = x6.get_folds(panel_full)
    apd = train_ridge_c1(panel_full, folds, feats_crossX, drop_nan_rows=True)
    pred_path = CACHE / "x16_F3_ridge_pool_crossX_drop_nan.parquet"
    apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    m = x6.run_sleeve_on_preds(pred_path, "x16_F3")
    print(f"  trained: {len(apd):,} rows (after drop NaN), IC={ic:+.4f}, "
          f"Sharpe={m.get('sharpe', '?'):+.2f} folds={m.get('folds_pos','?')} "
          f"conc={m.get('concentration','?')} [{time.time()-tf:.0f}s]", flush=True)
    results.append({"variant": "F3_ridge_pool_crossX_drop_nan",
                    "desc": "Ridge Pool+symid +crossX C1 FULL univ, drop NaN training rows",
                    "n_syms": 50, "train_ic": round(ic, 4), **m})
    del panel_full, apd; gc.collect()
    log_mem("after F3")

    # === F4: LGBM Pool+symid +aggT on FULL universe, drop NaN rows from training ===
    print(f"\n[F4] LGBM Pool+symid +aggT on FULL universe, DROP NaN training rows")
    tf = time.time()
    panel_full, _ = load_panel(HL_SYMS)
    folds = x6.get_folds(panel_full)
    apd = train_lgbm_pool_symid(panel_full, folds, feats_aggT, drop_nan_rows=True)
    pred_path = CACHE / "x16_F4_lgbm_pool_aggT_drop_nan.parquet"
    apd.to_parquet(pred_path, index=False)
    ic = float(apd["pred"].corr(apd["alpha_A"]))
    m = x6.run_sleeve_on_preds(pred_path, "x16_F4")
    print(f"  trained: {len(apd):,} rows, IC={ic:+.4f}, "
          f"Sharpe={m.get('sharpe', '?'):+.2f} folds={m.get('folds_pos','?')} "
          f"conc={m.get('concentration','?')} [{time.time()-tf:.0f}s]", flush=True)
    results.append({"variant": "F4_lgbm_pool_aggT_drop_nan",
                    "desc": "LGBM Pool+symid +aggT FULL univ, drop NaN training rows",
                    "n_syms": 50, "train_ic": round(ic, 4), **m})
    del panel_full, apd; gc.collect()

    keys = ["variant", "desc", "n_syms", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "error"]
    out_csv = OUT / "X16_subset_universe.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} variants → {out_csv} [{time.time()-t0:.0f}s]")
    print("\nReference baselines:")
    print("  LGBM Pool+symid +aggT (X6, HL-50): -0.63")
    print("  Ridge Pool+symid +crossX (X6, HL-50): +0.43")
    print("  Ridge Pool+symid +crossX C1 (X10, HL-50): -0.03")


if __name__ == "__main__":
    main()
