"""Loop Phase 5: validate +btc+fund (21 features) with 10-seed ensemble.

Phase 4 found +btc+fund @ Sharpe +4.40, CI [+1.09, +7.46], all 5 folds positive.
This is the BEST result of the investigation. Need to confirm it's not seed luck.

Compare:
  baseline_16          — current dedup_16_fund (Sharpe +2.52 5-seed, +1.52 10-seed Phase1)
  +btc+fund            — winner, 5-seed Sharpe +4.40, expected ~+3.5 with 10-seed
  +xs+btc+fund         — combined-all, 5-seed +3.72, sanity check
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train
from ml.research.alpha_v4_xs import block_bootstrap_ci
from ml.research.alpha_v9_pred_momentum_stack import evaluate_stacked

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
IC_PATH = REPO / "outputs/vBTC_universe_noleak/calibration_ic.csv"
OUT_DIR = REPO / "outputs/vBTC_loop_phase5"
OUT_DIR.mkdir(parents=True, exist_ok=True)
HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
CYCLES_PER_YEAR = (288 * 365) / HORIZON
PROD_FOLDS = [5, 6, 7, 8, 9]
SEEDS = (42, 1337, 7, 19, 2718, 99, 777, 123, 456, 789)   # 10 seeds
TOP_N = 15
K = 3

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
ALL_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
    "atr_pct_xs_rank", "dom_z_7d_vs_bk", "obv_z_1d_xs_rank",
    "obv_signal", "price_volume_corr_10",
    "hour_cos", "hour_sin",
]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
DEDUP_16_FUND = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN
ADD_XS_AGGS       = ["xs_alpha_dispersion_48b", "xs_alpha_mean_48b"]
ADD_CROSS_BTC     = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING  = ["funding_rate_1d_change", "funding_streak_pos"]


def _sharpe(x):
    x = np.asarray(x)
    return float(x.mean() / x.std() * np.sqrt(CYCLES_PER_YEAR)) if x.std() > 0 else 0.0


def _per_cycle_ic(test_df, pred, target_col):
    df = test_df.copy(); df["pred"] = pred
    ics = []
    for t, g in df.groupby("open_time"):
        sub = g[[target_col, "pred"]].dropna()
        if len(sub) < 5: continue
        ic = sub["pred"].rank().corr(sub[target_col].rank())
        if not pd.isna(ic): ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def run_config(panel, folds, feat_set, universe, label):
    print(f"\n=== {label} ({len(feat_set)} features, 10 seeds) ===", flush=True)
    cycles = []; ics = []
    for fold in folds:
        t0 = time.time()
        train, cal, test = _slice(panel, fold)
        tr = train[train["autocorr_pctile_7d"] >= THRESHOLD]
        ca = cal[cal["autocorr_pctile_7d"] >= THRESHOLD]
        if len(tr) < 1000 or len(ca) < 200: continue
        Xt = tr[feat_set].to_numpy(np.float32)
        Xc = ca[feat_set].to_numpy(np.float32)
        Xtest = test[feat_set].to_numpy(np.float32)
        yt = tr["target_A"].to_numpy(np.float32)
        yc = ca["target_A"].to_numpy(np.float32)
        mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
        if mask_t.sum() < 1000 or mask_c.sum() < 200: continue
        preds = []
        for s in SEEDS:
            m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
            preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
        ensemble = np.mean(preds, axis=0)
        keep_mask = test["symbol"].isin(universe).to_numpy()
        test_f = test[keep_mask].copy()
        pred_f = ensemble[keep_mask]
        ic = _per_cycle_ic(test_f, pred_f, "alpha_A")
        ics.append(ic)
        df_eval = evaluate_stacked(test_f, pred_f, use_conv_gate=True, use_pm_gate=True, top_k=K)
        for _, r in df_eval.iterrows():
            cycles.append({"fold": fold["fid"], "time": r["time"],
                            "net": r["net_bps"], "cost": r["cost_bps"]})
        n_f = pd.DataFrame(cycles)
        n_f = n_f[n_f["fold"] == fold["fid"]]["net"].to_numpy() if "fold" in n_f.columns else np.array([])
        sh = _sharpe(n_f) if len(n_f) else 0
        print(f"  fold {fold['fid']}: Sh={sh:+.2f} IC={ic:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

    df_v = pd.DataFrame(cycles)
    if df_v.empty: return {}
    net = df_v["net"].to_numpy()
    sh, lo, hi = block_bootstrap_ci(net, statistic=_sharpe, block_size=7, n_boot=2000)
    per_fold = {fid: _sharpe(df_v[df_v["fold"] == fid]["net"].to_numpy())
                  for fid in sorted(df_v["fold"].unique())}
    return {"label": label, "n_features": len(feat_set), "n_cycles": len(net),
            "mean_net": net.mean(), "cost": df_v["cost"].mean(),
            "sharpe": sh, "ci_lo": lo, "ci_hi": hi,
            "avg_ic": np.mean(ics) if ics else 0,
            "per_fold": per_fold}


def main():
    print(f"Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    df_ic_calib = pd.read_csv(IC_PATH).sort_values("ic", ascending=False)
    universe = set(df_ic_calib["symbol"].head(TOP_N).tolist())
    folds_all = _multi_oos_splits(panel)
    folds = [folds_all[i] for i in PROD_FOLDS if i < len(folds_all)]

    configs = [
        ("baseline_16",          DEDUP_16_FUND),
        ("+btc+fund",            DEDUP_16_FUND + ADD_CROSS_BTC + ADD_MORE_FUNDING),
        ("+xs+btc+fund",         DEDUP_16_FUND + ADD_XS_AGGS + ADD_CROSS_BTC + ADD_MORE_FUNDING),
    ]
    results = []
    for label, feats in configs:
        feats = [f for f in feats if f in panel.columns]
        r = run_config(panel, folds, feats, universe, label)
        if r: results.append(r)

    print(f"\n{'=' * 100}", flush=True)
    print(f"PHASE 5 — 10-SEED VALIDATION", flush=True)
    print(f"{'=' * 100}", flush=True)
    print(f"  {'config':<18} {'n_feat':>6}  {'avg_IC':>8}  {'Sharpe':>7}  {'CI_lo':>7}  {'CI_hi':>7}  {'mean':>6}",
          flush=True)
    for r in results:
        print(f"  {r['label']:<18} {r['n_features']:>6}  {r['avg_ic']:>+8.4f}  "
              f"{r['sharpe']:>+7.2f}  {r['ci_lo']:>+7.2f}  {r['ci_hi']:>+7.2f}  "
              f"{r['mean_net']:>+6.2f}", flush=True)
    print(f"\n  Per-fold:", flush=True)
    for r in results:
        cells = " ".join(f"{r['per_fold'].get(f, 0):+5.2f}" for f in PROD_FOLDS)
        print(f"  {r['label']:<18}  " + cells, flush=True)

    pd.DataFrame(results).to_csv(OUT_DIR / "phase5_validation.csv", index=False)
    print(f"\n  saved → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
