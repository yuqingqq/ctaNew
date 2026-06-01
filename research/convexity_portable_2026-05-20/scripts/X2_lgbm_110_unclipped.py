"""X2 — LGBM with sym_id on 110-panel, target_A unclipped.

The key test: does V3.1's LGBM architecture (with sym_id) extract more signal
on the 110-panel WHEN THE TARGET ISN'T CLIPPED at ±5? Phase UNI-111 gave -1.48
on the clipped 111-panel — the diagnostic showed the clip removes alpha-bearing
rows of meme/rotation names.

This test:
- Builds target_A FRESH per-symbol PIT z without any clip
- Trains LGBM with sym_id + 19 BTC-frame features (110-panel)
- Single-seed for speed (5-seed = 5× compute; will scale if 1-seed promising)
- Walk-forward 9 folds, embargo, label purging via exit_time
- Outputs predictions schema-compatible with phase_ah_sleeve
"""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"; CACHE.mkdir(parents=True, exist_ok=True)
PANEL = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"

FEATS = [
    "return_1d", "return_8h", "atr_pct", "obv_z_1d", "vwap_slope_96",
    "bars_since_high", "bars_since_high_xs_rank", "autocorr_pctile_7d",
    "corr_to_btc_1d", "corr_to_btc_change_3d", "beta_to_btc_change_5d",
    "dom_btc_z_1d", "dom_btc_change_288b",
    "idio_vol_to_btc_1h", "idio_vol_to_btc_1d",
    "vol_zscore_4h_over_7d",
    "funding_rate", "funding_rate_z_7d", "funding_rate_1d_change",
]

HORIZON = 48           # 4h
TRAIL_BARS = 288 * 7
MIN_TRAIL = 288
N_FOLDS = 9
WINSOR_TARGET = 50.0   # mild cap to prevent extreme z from numerical issues — NOT the ±5 hack

# V3.1 LGBM hyperparameters (pinned from production per CLAUDE.md)
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=31,
    max_depth=6,
    min_data_in_leaf=300,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    verbose=-1,
    n_estimators=400,
)


def main():
    t0 = time.time()
    print("=== X2 LGBM-with-sym_id on 110-panel, unclipped target_A ===\n", flush=True)
    cols = ["symbol", "open_time", "alpha_beta", "return_pct", "exit_time"] + FEATS
    cols = list(dict.fromkeys(cols))
    p = pd.read_parquet(PANEL, columns=cols)
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    p["exit_time"] = pd.to_datetime(p["exit_time"], utc=True)
    p = p.dropna(subset=["alpha_beta"]).sort_values(["symbol", "open_time"]).reset_index(drop=True)
    print(f"  panel: {len(p):,} rows × {p['symbol'].nunique()} syms", flush=True)

    # Build target_A per-symbol PIT z, no clip
    print("  building per-symbol PIT-z target_A (no clip)...", flush=True)
    p["rmean"] = p.groupby("symbol")["alpha_beta"].transform(
        lambda s: s.expanding(min_periods=MIN_TRAIL).mean().shift(HORIZON))
    p["rstd"] = p.groupby("symbol")["alpha_beta"].transform(
        lambda s: s.rolling(TRAIL_BARS, min_periods=MIN_TRAIL).std().shift(HORIZON))
    p["target_A"] = (p["alpha_beta"] - p["rmean"]) / p["rstd"].replace(0, np.nan)
    # mild winsorization at ±50 (NOT the ±5 production hack) to prevent extreme tail rows
    # dominating LGBM's MSE
    p["target_A"] = p["target_A"].clip(-WINSOR_TARGET, WINSOR_TARGET)
    nv = p["target_A"].notna().sum()
    print(f"  target_A non-null: {nv:,} ({nv/len(p)*100:.1f}%)", flush=True)
    print(f"  target_A: min={p['target_A'].min():.2f}, max={p['target_A'].max():.2f}, "
          f"std={p['target_A'].std():.3f}, "
          f"frac|t|>5: {(p['target_A'].abs() > 5).mean()*100:.3f}%, "
          f"frac|t|>10: {(p['target_A'].abs() > 10).mean()*100:.3f}%", flush=True)

    # Build sym_id (categorical integer per symbol)
    syms_sorted = sorted(p["symbol"].unique())
    sym_map = {s: i for i, s in enumerate(syms_sorted)}
    p["sym_id"] = p["symbol"].map(sym_map).astype("int32")

    feats_with_symid = FEATS + ["sym_id"]
    print(f"  features (incl sym_id): {len(feats_with_symid)}", flush=True)

    # Walk-forward 9 folds
    times = sorted(p["open_time"].unique())
    n_times = len(times); fold_size = n_times // N_FOLDS

    all_preds = []
    for fid in range(N_FOLDS):
        tf = time.time()
        i0 = fid * fold_size
        i1 = min((fid+1)*fold_size, n_times-1) if fid < N_FOLDS-1 else n_times
        oos_start = pd.Timestamp(times[i0])
        oos_end = pd.Timestamp(times[i1-1])
        embargo_cut = oos_start - pd.Timedelta(days=1)

        # Label purging via exit_time: training labels must have exit_time < embargo_cut
        train_mask = (p["exit_time"] < embargo_cut) & p["target_A"].notna()
        train = p[train_mask]
        test = p[(p["open_time"] >= oos_start) & (p["open_time"] <= oos_end)]

        if len(train) < 5000 or len(test) < 1000:
            print(f"  fold {fid}: skipped (n_train={len(train)}, n_test={len(test)})", flush=True)
            continue

        Xtr = train[feats_with_symid]
        ytr = train["target_A"].to_numpy()
        Xte = test[feats_with_symid]

        m = lgb.LGBMRegressor(random_state=42, **LGB_PARAMS)
        m.fit(Xtr, ytr, categorical_feature=["sym_id"])
        pred = m.predict(Xte)

        out_df = test[["symbol", "open_time", "alpha_beta", "return_pct", "exit_time"]].copy()
        out_df.columns = ["symbol", "open_time", "alpha_A", "return_pct", "exit_time"]
        out_df["pred"] = pred
        out_df["fold"] = fid
        all_preds.append(out_df)

        # IC on test
        valid = test["alpha_beta"].notna()
        ic = float(np.corrcoef(pred[valid.values], test.loc[valid, "alpha_beta"])[0, 1]) if valid.sum() > 100 else np.nan
        print(f"  fold {fid}: n_tr={len(train):>7,} n_te={len(test):>7,} "
              f"IC={ic:+.4f} best_iter={m.best_iteration_ if hasattr(m,'best_iteration_') else '—'} "
              f"({time.time()-tf:.0f}s)", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    out_path = CACHE / "all_predictions_X2_lgbm.parquet"
    apd.to_parquet(out_path, index=False)
    print(f"\nPredictions saved: {len(apd):,} rows -> {out_path}", flush=True)
    print(f"[total {time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
