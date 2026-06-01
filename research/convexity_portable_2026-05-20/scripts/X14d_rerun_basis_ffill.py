"""X14b — Rerun +crossX cells with NEW 5m-granular cross_exchange_features.

Compare 4h-granular (X7/X10) vs 5m-granular (X14) crossX feature performance.

3 cells to re-run:
  - Ridge Pool+symid +crossX (X10 C1 norm sym_id): X10 = -0.03 (Δ vs X6 +0.43 = -0.46)
  - Ridge Per-sym +crossX: X7 = +1.12
  - LGBM Per-sym +crossX: X7 = -0.17 (biggest LGBM lift in matrix)

Expect: if 4h NaN problem was real, 5m features should improve Sharpe.
Memory-lite (float32 + gc + per-fold).
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

LGB_PARAMS_PERSYM = dict(
    objective="regression", metric="rmse", learning_rate=0.05,
    num_leaves=15, max_depth=4, min_data_in_leaf=30,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, verbose=-1, n_estimators=200,
)


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)


def load_panel_with_5m_crossX():
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + x6.BASE)
    p = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                        columns=list(set(needed)))
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p[p["symbol"].isin(HL_SYMS) & (p["symbol"] != "BTCUSDT")].copy()

    # NEW: 5m crossX features
    cross_path = REPO / "data/ml/cache/cross_exchange_features_5m_v2.parquet"
    cross_df = pd.read_parquet(cross_path)
    cross_df["open_time"] = pd.to_datetime(cross_df["open_time"], utc=True)
    crossX_z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    print(f"  5m crossX features: {crossX_z_cols}", flush=True)
    p = p.merge(cross_df[["symbol", "open_time"] + crossX_z_cols],
                 on=["symbol", "open_time"], how="left")
    print(f"  panel rows: {len(p):,}, crossX non-null rate (5m): "
          f"{p[crossX_z_cols[0]].notna().mean()*100:.1f}%", flush=True)
    p = x6.build_target_z(p)
    for c in p.columns:
        if p[c].dtype in ("float64",): p[c] = p[c].astype("float32")
    for c in crossX_z_cols:
        x6.HEAVY_TAIL.add(c)
    return p, crossX_z_cols


def train_ridge_c1(panel, folds, feats):
    """Ridge Pool+symid with C1 (normalized sym_id, wider α grid)."""
    syms_sorted = sorted(panel["symbol"].unique())
    sym_idx = {s: i for i, s in enumerate(syms_sorted)}
    sym_codes = panel["symbol"].map(sym_idx).to_numpy(np.int32)
    n_rows = len(panel)
    n_dum = len(syms_sorted) - 1
    dum_all = np.zeros((n_rows, n_dum), dtype=np.float32)
    mask = sym_codes > 0
    dum_all[mask, sym_codes[mask] - 1] = 1.0

    all_preds = []
    sstats0 = hstats0 = None
    for f, ts, te, ec in folds:
        train_mask = ((panel["exit_time"] < ec).to_numpy()
                      & panel["target_z"].notna().to_numpy())
        test_mask = ((panel["open_time"] >= ts) & (panel["open_time"] <= te)).to_numpy()
        train = panel.iloc[train_mask]
        test = panel.iloc[test_mask]
        if len(train) < 5000 or len(test) < 1000: continue
        if sstats0 is None:
            sstats0, hstats0 = x6.fit_preproc(train, feats)
        Xtr = x6.apply_preproc(train, feats, sstats0, hstats0).astype(np.float32)
        Xte = x6.apply_preproc(test, feats, sstats0, hstats0).astype(np.float32)
        ytr = train["target_z"].to_numpy(np.float32)

        dum_tr = dum_all[train_mask]; dum_te = dum_all[test_mask]
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


def train_ridge_persym(panel, folds, feats):
    all_preds = []
    for f, ts, te, ec in folds:
        train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        out_frames = []
        for sym, gtr in train_all.groupby("symbol"):
            if len(gtr) < 300: continue
            gte = test_all[test_all["symbol"] == sym]
            if len(gte) < 30: continue
            sstats, hstats = x6.fit_preproc(gtr, feats)
            Xtr = x6.apply_preproc(gtr, feats, sstats, hstats).astype(np.float32)
            Xte = x6.apply_preproc(gte, feats, sstats, hstats).astype(np.float32)
            ytr = gtr["target_z"].to_numpy(np.float32)
            try:
                m = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(Xtr, ytr)
                pred = m.predict(Xte).astype(np.float32)
            except Exception: continue
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                     "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred; o["fold"] = f
            out_frames.append(o)
            del Xtr, Xte, ytr, m
        if out_frames: all_preds.append(pd.concat(out_frames, ignore_index=True))
        gc.collect()
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])


def train_lgbm_persym(panel, folds, feats):
    all_preds = []
    for f, ts, te, ec in folds:
        train_all = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test_all = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        out_frames = []
        for sym, gtr in train_all.groupby("symbol"):
            if len(gtr) < 200: continue
            gte = test_all[test_all["symbol"] == sym]
            if len(gte) < 30: continue
            try:
                m = lgb.LGBMRegressor(random_state=20260520, **LGB_PARAMS_PERSYM)
                m.fit(gtr[feats], gtr["target_z"].to_numpy(np.float32))
                pred = m.predict(gte[feats]).astype(np.float32)
            except Exception: continue
            o = gte[["symbol", "open_time", "alpha_vs_btc_realized",
                     "return_pct", "exit_time"]].copy()
            o.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
            o["pred"] = pred; o["fold"] = f
            out_frames.append(o)
            del m
        if out_frames: all_preds.append(pd.concat(out_frames, ignore_index=True))
        gc.collect()
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])


def main():
    t0 = time.time()
    print("=== X14b rerun crossX cells with 5m-granular features ===\n", flush=True)
    log_mem("start")
    panel, crossX_z_cols = load_panel_with_5m_crossX()
    folds = x6.get_folds(panel)
    log_mem("after_panel")

    feats_crossX = x6.BASE + crossX_z_cols

    variants = [
        ("Ridge_pool+symid_crossX_5m_C1", "Ridge Pool+symid +crossX 5m (C1 norm)", "ridge_c1"),
        ("Ridge_per-sym_crossX_5m",       "Ridge Per-sym +crossX 5m",              "ridge_persym"),
        ("LGBM_per-sym_crossX_5m",        "LGBM Per-sym +crossX 5m",               "lgbm_persym"),
    ]
    refs = {
        "Ridge_pool+symid_crossX_5m_C1": ("X10 -0.03", "vs +aggT C1 +1.38"),
        "Ridge_per-sym_crossX_5m": ("X7 +1.12", "best Per-sym crossX cell"),
        "LGBM_per-sym_crossX_5m": ("X7 -0.17", "best LGBM cell"),
    }

    results = []
    for v_name, desc, fn_key in variants:
        tf = time.time()
        log_mem(f"before {v_name}")
        print(f"\n[{v_name}] {desc}", flush=True)
        print(f"  reference: {refs[v_name]}", flush=True)
        try:
            if fn_key == "ridge_c1":
                apd = train_ridge_c1(panel, folds, feats_crossX)
            elif fn_key == "ridge_persym":
                apd = train_ridge_persym(panel, folds, feats_crossX)
            elif fn_key == "lgbm_persym":
                apd = train_lgbm_persym(panel, folds, feats_crossX)
            pred_path = CACHE / f"x14d_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {type(e).__name__}: {e}"); import traceback; traceback.print_exc()
            results.append({"variant": v_name, "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x14d_{v_name}")
        row = {"variant": v_name, "desc": desc, "ref": str(refs[v_name]),
               "train_ic": round(ic, 4), "train_time_s": round(time.time()-tf, 0), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del apd; gc.collect()
        log_mem(f"after {v_name}")

    keys = ["variant", "desc", "ref", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "train_time_s", "error"]
    out_csv = OUT / "X14d_basis_ffill_rerun.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} → {out_csv} [{time.time()-t0:.0f}s]")
    log_mem("end")


if __name__ == "__main__":
    main()
