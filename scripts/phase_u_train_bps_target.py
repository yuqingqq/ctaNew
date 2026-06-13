"""Phase U: retrain LGBM on bps-scale target (alpha_A * 1e4) instead of z-scored target_A.

Production target_A = (alpha - β·basket_fwd) / rstd_per_symbol — z-scored per
symbol. Model learns rank order, not magnitude.

This script retrains with target = alpha_A * 1e4 (bps). The cross-sectional
relative magnitudes are preserved (since z-score is a monotone transform per
symbol, ranks roughly match BUT magnitude information is preserved across
symbols).

Output: outputs/vBTC_audit_bps_target/all_predictions.parquet
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
OUT = REPO / "outputs/vBTC_audit_bps_target"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
MIN_HISTORY_DAYS = 60

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
ADD_CROSS_BTC = ["corr_to_btc_1d", "idio_vol_to_btc_1h", "beta_to_btc_change_5d"]
ADD_MORE_FUNDING = ["funding_rate_1d_change", "funding_streak_pos"]
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try:
            L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception:
            continue
    return L


def train_fold(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100:
        return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    # *** KEY CHANGE: target is alpha_A * 1e4 (bps) instead of target_A ***
    yt = (tr["alpha_A"] * 1e4).to_numpy(np.float32)
    yc = (ca["alpha_A"] * 1e4).to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200:
        return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


def main():
    print("=== Phase U: train on bps-scale target ===\n", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"  Panel: {len(panel):,} rows, {panel.symbol.nunique()} syms", flush=True)
    print(f"  Features: {len(feat_set)}", flush=True)
    print(f"  Target: alpha_A * 1e4 (bps scale)", flush=True)

    folds_all = _multi_oos_splits(panel)
    print(f"  Folds: {len(folds_all)}", flush=True)

    listings = get_listings()
    pfo = panel.groupby("symbol")["open_time"].min()
    for sym, t in pfo.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    panel_syms = set(panel["symbol"].unique())

    def eligibility_at(ts):
        if isinstance(ts, (int, np.integer)):
            t = pd.Timestamp(ts, unit="ms", tz="UTC")
        else:
            t = pd.Timestamp(ts)
            if t.tz is None: t = t.tz_localize("UTC")
        cutoff = t - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold(panel, folds_all[fid], feat_set, eligible)
        if td is None:
            print(f"  fold {fid}: skipped", flush=True)
            continue
        cols = ["symbol", "open_time", "alpha_A", "return_pct"]
        if "exit_time" in td.columns:
            cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p  # bps-scale predictions
        df["fold"] = fid
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: eligible={len(eligible)}, n_test={len(td):,} "
              f"pred_mean={p.mean():+.2f}bps pred_std={p.std():.2f}bps "
              f"({time.time()-t0:.0f}s)", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    apd.to_parquet(OUT / "all_predictions.parquet", index=False)
    print(f"\n  saved → {OUT}/all_predictions.parquet  ({len(apd):,} rows)", flush=True)
    print(f"  Final pred distribution (all folds):", flush=True)
    print(f"    mean: {apd['pred'].mean():+.3f} bps", flush=True)
    print(f"    std:  {apd['pred'].std():.3f} bps", flush=True)
    print(f"    p1:   {apd['pred'].quantile(0.01):+.2f} bps", flush=True)
    print(f"    p99:  {apd['pred'].quantile(0.99):+.2f} bps", flush=True)


if __name__ == "__main__":
    main()
