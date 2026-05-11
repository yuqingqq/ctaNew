"""Train and persist the vBTC production model artifact.

Produces models/vBTC_production.pkl containing:
  - 5-seed LGBM ensemble (latest fold's model, trained on data up to cal_start)
  - Calibration metadata (cal_start, cal_end, features, hyperparams)
  - PIT-eligibility config (min_history_days=60)
  - Universe artifact: rolling-IC top-15 at most-recent boundary

Run periodically (e.g., monthly via cron):
  python -m live.train_vBTC_artifact

The artifact is loaded by live/vBTC_paper_bot.py for live/paper trading.
"""
from __future__ import annotations
import json, pickle, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
MODEL_DIR = REPO / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = MODEL_DIR / "vBTC_production.pkl"

# Validated production hyperparameters
HORIZON = 48                         # 48 5-min bars = 4 hours
RC = 0.50                            # autocorr percentile filter
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)      # 5-seed ensemble
TARGET_COL = "target_A"
MIN_OBS_PER_SYM = 100
TARGET_N = 15                        # universe size
IC_WINDOW_DAYS = 180
MIN_HISTORY_DAYS = 60                # PIT eligibility

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


def get_listing_dates_from_klines():
    listings = {}
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        try:
            ts = pd.Timestamp(files[0].stem, tz="UTC")
            listings[sym_dir.name] = ts
        except Exception:
            continue
    return listings


def compute_rolling_ic_universe(panel, train_pred_panel, boundary_ts, listings,
                                  ic_window_days=IC_WINDOW_DAYS,
                                  min_history_days=MIN_HISTORY_DAYS):
    """Compute the rolling-IC top-15 universe at boundary_ts.

    panel: full feature panel (for symbol list).
    train_pred_panel: prior OOS predictions (must have 'pred', 'alpha_A', 'open_time', 'symbol').
    boundary_ts: pd.Timestamp (UTC).
    """
    cutoff = boundary_ts - pd.Timedelta(days=min_history_days)
    eligible = {s for s in panel["symbol"].unique()
                  if listings.get(s) and listings[s] <= cutoff}

    window_start = boundary_ts - pd.Timedelta(days=ic_window_days)
    past = train_pred_panel[
        (train_pred_panel["open_time"] >= window_start) &
        (train_pred_panel["open_time"] < boundary_ts) &
        (train_pred_panel["symbol"].isin(eligible))
    ].dropna(subset=["alpha_A"])
    if len(past) < 1000:
        return set()
    ics = past.groupby("symbol").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= MIN_OBS_PER_SYM else np.nan
    )
    ics_sorted = ics.dropna().sort_values(ascending=False)
    return set(ics_sorted.head(TARGET_N).index.tolist())


def main():
    print(f"[train_vBTC_artifact] Loading panel...", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_21 if f in panel.columns]
    listings = get_listing_dates_from_klines()
    # Fallback first_obs
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    folds = _multi_oos_splits(panel)
    latest_fold = folds[-1]
    cal_start = latest_fold["cal_start"]
    cal_end = latest_fold["cal_end"]
    print(f"  Panel: {panel['symbol'].nunique()} symbols, {len(panel):,} rows", flush=True)
    print(f"  Latest fold: cal {cal_start.date()} → {cal_end.date()}", flush=True)

    # PIT-eligible symbols at cal_start (for training)
    cutoff = cal_start - pd.Timedelta(days=MIN_HISTORY_DAYS)
    eligible = {s for s in panel["symbol"].unique()
                  if listings.get(s) and listings[s] <= cutoff}
    print(f"  PIT-eligible at cal_start ({MIN_HISTORY_DAYS}d filter): {len(eligible)} symbols",
          flush=True)

    # Train on latest available data
    print(f"\n[train_vBTC_artifact] Training {len(SEEDS)}-seed ensemble on latest fold...",
          flush=True)
    train, cal, _ = _slice(panel, latest_fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) &
                 (train["symbol"].isin(eligible))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) &
                (cal["symbol"].isin(eligible))]
    print(f"  Train rows: {len(tr):,}, Cal rows: {len(ca):,}", flush=True)
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    yt = tr[TARGET_COL].to_numpy(np.float32)
    yc = ca[TARGET_COL].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    Xt = Xt[mask_t]; yt = yt[mask_t]
    Xc = Xc[mask_c]; yc = yc[mask_c]

    models = []
    for seed in SEEDS:
        t0 = time.time()
        m = _train(Xt, yt, Xc, yc, seed=seed)
        models.append(m)
        print(f"  seed={seed}: best_iter={m.best_iteration} ({time.time()-t0:.0f}s)",
              flush=True)

    # Compute universe artifact:
    # The "current" universe at deployment time = most-recent IC boundary's top-15.
    # We use all OOS-test predictions from prior folds to construct IC.
    print(f"\n[train_vBTC_artifact] Computing universe artifact...", flush=True)
    print(f"  Building prior-OOS-pred panel from all folds...", flush=True)
    prior_oos_preds = []
    for f in folds[:-1]:  # exclude latest (which we just trained)
        eligible_f = {s for s in panel["symbol"].unique()
                        if listings.get(s) and listings[s] <= (f["cal_start"] - pd.Timedelta(days=MIN_HISTORY_DAYS))}
        train_f, cal_f, test_f = _slice(panel, f)
        tr_f = train_f[(train_f["autocorr_pctile_7d"] >= THRESHOLD) &
                         (train_f["symbol"].isin(eligible_f))]
        ca_f = cal_f[(cal_f["autocorr_pctile_7d"] >= THRESHOLD) &
                        (cal_f["symbol"].isin(eligible_f))]
        test_r = test_f[test_f["symbol"].isin(eligible_f)].copy()
        if len(tr_f) < 1000 or len(ca_f) < 200 or len(test_r) < 100:
            continue
        Xt_f = tr_f[feat_set].to_numpy(np.float32)
        Xc_f = ca_f[feat_set].to_numpy(np.float32)
        Xtest_f = test_r[feat_set].to_numpy(np.float32)
        yt_f = tr_f[TARGET_COL].to_numpy(np.float32)
        yc_f = ca_f[TARGET_COL].to_numpy(np.float32)
        mt_f = ~np.isnan(yt_f); mc_f = ~np.isnan(yc_f)
        preds = []
        for s in SEEDS:
            m = _train(Xt_f[mt_f], yt_f[mt_f], Xc_f[mc_f], yc_f[mc_f], seed=s)
            preds.append(m.predict(Xtest_f, num_iteration=m.best_iteration))
        df_f = test_r[["symbol", "open_time", "alpha_A"]].copy()
        df_f["pred"] = np.mean(preds, axis=0)
        df_f["fold"] = f["fid"]
        prior_oos_preds.append(df_f)
        print(f"    fold {f['fid']}: {len(df_f):,} preds", flush=True)
    train_pred_panel = pd.concat(prior_oos_preds, ignore_index=True)

    # Universe at most-recent boundary
    deployment_boundary = cal_end  # start trading at cal_end + embargo
    universe = compute_rolling_ic_universe(panel, train_pred_panel, deployment_boundary,
                                              listings)
    print(f"  Universe at boundary {deployment_boundary.date()}: {sorted(universe)}",
          flush=True)

    # Save artifact
    artifact = {
        "version": "vBTC_production_v1",
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "panel_path": str(PANEL_PATH),
        "feature_set": feat_set,
        "target_col": TARGET_COL,
        "horizon_bars": HORIZON,
        "min_history_days": MIN_HISTORY_DAYS,
        "ic_window_days": IC_WINDOW_DAYS,
        "target_n": TARGET_N,
        "top_k": 4,
        "seeds": list(SEEDS),
        "ensemble_models": models,
        "ensemble_best_iters": [int(m.best_iteration) for m in models],
        "deployment_boundary": str(deployment_boundary),
        "deployment_universe": sorted(universe),
        "cal_start": str(cal_start),
        "cal_end": str(cal_end),
        "n_train_rows": int(len(tr)),
        "n_cal_rows": int(len(ca)),
        "listings": {s: t.isoformat() for s, t in listings.items()},
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(artifact, f)
    # Also save a JSON of metadata (no models)
    meta = {k: v for k, v in artifact.items() if k != "ensemble_models"}
    meta_path = OUT_PATH.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n[train_vBTC_artifact] saved → {OUT_PATH}", flush=True)
    print(f"  metadata → {meta_path}", flush=True)
    print(f"  size: {OUT_PATH.stat().st_size / 1024:.0f} KB", flush=True)


if __name__ == "__main__":
    main()
