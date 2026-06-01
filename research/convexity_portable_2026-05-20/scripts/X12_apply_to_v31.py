"""X12 — Apply best matrix config to V3.1 production stack.

Goal: take BEST cell (Ridge Per-sym +cohort = +2.01 on HL-50 portable BASE)
and add V3.1's WINNER_21 basket-frame features (`dom_level_vs_bk`,
`dom_change_288b_vs_bk`, `bk_ema_slope_4h`, `idio_ret_48b_vs_bk`,
`corr_change_3d_vs_bk`, `idio_vol_1d_vs_bk_xs_rank`, mfi, price_volume_corr_20,
funding_streak_pos) to compare to V3.1 production +3.00 absolute Sharpe.

Tests:
  V1: Ridge Per-sym on (BASE + cohort + basket-frame) — apples-to-apples vs V3.1
  V2: LGBM Pool+symid on (WINNER_21 + cohort) — adds cohort to production LGBM
  V3: LGBM Pool+symid on (WINNER_21 + crossX) — adds cross-exchange to production LGBM
"""
from __future__ import annotations
import csv, sys, time, warnings, importlib.util, gc, resource
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.linear_model import RidgeCV
import lightgbm as lgb


def log_mem(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"  [MEM {label}] peak_rss={rss_mb:.0f}MB", flush=True)
    return rss_mb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"

spec = importlib.util.spec_from_file_location("x6",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6_controlled_matrix.py")
x6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(x6)
spec_b = importlib.util.spec_from_file_location("x6b",
    REPO / "research/convexity_portable_2026-05-20/scripts/X6b_cohort_fill.py")
x6b = importlib.util.module_from_spec(spec_b); spec_b.loader.exec_module(x6b)

# V3.1 WINNER_21 features
WINNER_21 = [
    "return_1d", "atr_pct", "dom_level_vs_bk", "dom_change_288b_vs_bk",
    "bk_ema_slope_4h", "idio_ret_48b_vs_bk", "corr_change_3d_vs_bk",
    "obv_z_1d", "vwap_slope_96", "price_volume_corr_20", "mfi",
    "bars_since_high_xs_rank", "idio_vol_1d_vs_bk_xs_rank",
    "funding_rate", "funding_rate_z_7d", "corr_to_btc_1d",
    "idio_vol_to_btc_1h", "beta_to_btc_change_5d",
    "funding_rate_1d_change", "funding_streak_pos",
    # sym_id added separately
]

LGB_PARAMS = dict(
    objective="regression", metric="rmse", learning_rate=0.03,
    num_leaves=31, max_depth=6, min_data_in_leaf=300,
    feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, verbose=-1, n_estimators=400,
)
HL_MAP = pd.read_csv(REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv")
HL_SYMS = set(HL_MAP[HL_MAP["on_hl"] == True]["symbol"].tolist())


def load_panel_v3_universe(include_btc=False):
    """Load 51-panel + cohort + crossX features. include_btc=True for V3.1 native."""
    needed = (["symbol", "open_time", "exit_time", "alpha_vs_btc_realized", "return_pct"]
              + WINNER_21 + x6.BASE)
    panel = pd.read_parquet(REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet",
                            columns=list(set(needed)))
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel["exit_time"] = pd.to_datetime(panel["exit_time"], utc=True)
    if not include_btc:
        panel = panel[panel["symbol"].isin(HL_SYMS) & (panel["symbol"] != "BTCUSDT")].copy()
    else:
        panel = panel[panel["symbol"].isin(HL_SYMS | {"BTCUSDT"})].copy()
    panel = x6b.build_cohort_fixed(panel)
    cross_path = REPO / "data/ml/cache/cross_exchange_features.parquet"
    cross_df = pd.read_parquet(cross_path)
    cross_z_cols = [c for c in cross_df.columns if c.endswith("_basis_z")]
    panel = panel.merge(cross_df[["symbol", "open_time"] + cross_z_cols],
                        on=["symbol", "open_time"], how="left")
    panel = x6.build_target_z(panel)
    for c in panel.columns:
        if panel[c].dtype in ("float64",): panel[c] = panel[c].astype("float32")
    for c in x6.COHORT_EXTRAS + cross_z_cols:
        x6.HEAVY_TAIL.add(c)
    return panel, cross_z_cols


def train_persym_ridge(panel, folds, feats):
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


def train_lgbm_pool(panel, folds, feats_no_symid):
    syms_sorted = sorted(panel["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(syms_sorted)}
    panel = panel.copy()
    panel["sym_id"] = panel["symbol"].map(sym_map).astype("int32")
    feats = feats_no_symid + ["sym_id"]
    all_preds = []
    for f, ts, te, ec in folds:
        train = panel[(panel["exit_time"] < ec) & panel["target_z"].notna()]
        test = panel[(panel["open_time"] >= ts) & (panel["open_time"] <= te)]
        if len(train) < 5000 or len(test) < 1000: continue
        m = lgb.LGBMRegressor(random_state=20260520, **LGB_PARAMS)
        m.fit(train[feats], train["target_z"].to_numpy(np.float32), categorical_feature=["sym_id"])
        pred = m.predict(test[feats]).astype(np.float32)
        out = test[["symbol", "open_time", "alpha_vs_btc_realized", "return_pct", "exit_time"]].copy()
        out.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out["pred"] = pred; out["fold"] = f
        all_preds.append(out)
        print(f"      fold {f}: n_tr={len(train):,}", flush=True)
        del m; gc.collect()
    return pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])


def main():
    t0 = time.time()
    print("=== X12 apply best to V3.1 production stack ===\n", flush=True)

    panel, cross_z_cols = load_panel_v3_universe(include_btc=False)
    folds = x6.get_folds(panel)
    print(f"  panel: {len(panel):,} rows × {panel['symbol'].nunique()} syms", flush=True)

    variants = [
        ("V1_persym_ridge_W21_cohort",
         lambda: train_persym_ridge(panel, folds, WINNER_21 + x6.COHORT_EXTRAS),
         "Ridge Per-sym on WINNER_21 + cohort"),
        ("V2_lgbm_pool_W21_cohort",
         lambda: train_lgbm_pool(panel, folds, WINNER_21 + x6.COHORT_EXTRAS),
         "LGBM Pool+symid on WINNER_21 + cohort (add cohort to production)"),
        ("V3_lgbm_pool_W21_crossX",
         lambda: train_lgbm_pool(panel, folds, WINNER_21 + cross_z_cols),
         "LGBM Pool+symid on WINNER_21 + crossX (add cross-exchange to production)"),
    ]

    results = []
    for v_name, fn, desc in variants:
        tf = time.time()
        log_mem(f"before {v_name}")
        print(f"\n[{v_name}] {desc}", flush=True)
        try:
            apd = fn()
            pred_path = CACHE / f"x12_{v_name}_preds.parquet"
            apd.to_parquet(pred_path, index=False)
            ic = float(apd["pred"].corr(apd["alpha_A"]))
            print(f"  trained: {len(apd):,} rows, IC={ic:+.4f} [{time.time()-tf:.0f}s]", flush=True)
        except Exception as e:
            print(f"  TRAIN ERR: {type(e).__name__}: {e}"); import traceback; traceback.print_exc()
            results.append({"variant": v_name, "error": str(e)}); continue
        m = x6.run_sleeve_on_preds(pred_path, f"x12_{v_name}")
        row = {"variant": v_name, "desc": desc, "train_ic": round(ic, 4),
               "train_time_s": round(time.time()-tf, 0), **m}
        results.append(row)
        if "sharpe" in m:
            print(f"  sleeve: Sharpe {m['sharpe']:+.2f} folds {m.get('folds_pos','?')} "
                  f"conc {m.get('concentration','?')} PnL {m.get('totPnL','?')}", flush=True)
        del apd
        gc.collect()
        log_mem(f"after {v_name}")

    keys = ["variant", "desc", "train_ic", "sharpe", "ci_lo", "ci_hi",
            "totPnL", "maxDD", "folds_pos", "concentration", "train_time_s", "error"]
    out_csv = OUT / "X12_apply_to_v31.csv"
    with open(out_csv, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results: w.writerow(r)
    print(f"\nSaved {len(results)} variants → {out_csv} [{time.time()-t0:.0f}s]")
    print(f"\nReference: V3.1 production LGBM Pool+symid WINNER_21 on HL-50 = +3.00")
    for r in results:
        if "sharpe" in r:
            print(f"  {r['variant']:<32} Sharpe={r['sharpe']:+.2f}  IC={r['train_ic']:+.4f}")


if __name__ == "__main__":
    main()
