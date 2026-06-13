"""Phase Q: full WINNER_23 retrain and audit panel rebuild.

WINNER_23 = WINNER_21 + [ethbtc_change_24h, xs_ret_disp_1d]

Pipeline:
  Step 1: build augmented panel (existing panel + 2 new features)
  Step 2: train 9 OOS folds with WINNER_23 → all_predictions_w23.parquet
  Step 3: build new sleeves with same machinery (rolling-IC + filter_refill +
          conv_gate + flat_real + K=3) → production_sleeves_w23.parquet
  Step 4: V3.1 aggregation + paired bootstrap vs current V3.1
  Step 5: report verdict

This is the full retrain — expected runtime ~30-60 min for steps 1-2.
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

OUT = REPO / "outputs/vBTC_phase_Q"
OUT.mkdir(parents=True, exist_ok=True)
SLEEVE_OUT = REPO / "outputs/vBTC_sleeve_horizon"
HORIZON = 48
SEEDS = (42, 1337, 7, 19, 2718)
TARGET_COL = "target_A"
MIN_HISTORY_DAYS = 60
RC = 0.50
THRESHOLD = 1 - RC

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
ADD_PHASE_Q = ["ethbtc_change_24h", "xs_ret_disp_1d"]  # NEW
WINNER_21 = [f for f in V6_CLEAN_28 if f not in ALL_DROPS] + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING
WINNER_23 = WINNER_21 + ADD_PHASE_Q


def load_eth_btc_ratio():
    bdir = REPO / "data/ml/test/parquet/klines/BTCUSDT/5m"
    edir = REPO / "data/ml/test/parquet/klines/ETHUSDT/5m"
    bf = sorted(bdir.glob("*.parquet"))
    ef = sorted(edir.glob("*.parquet"))
    bdfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in bf]
    edfs = [pd.read_parquet(f, columns=["open_time", "close"]) for f in ef]
    btc = pd.concat(bdfs, ignore_index=True).rename(columns={"close": "btc_close"})
    eth = pd.concat(edfs, ignore_index=True).rename(columns={"close": "eth_close"})
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    eth["open_time"] = pd.to_datetime(eth["open_time"], utc=True)
    btc = btc.drop_duplicates("open_time")
    eth = eth.drop_duplicates("open_time")
    df = btc.merge(eth, on="open_time", how="inner").sort_values("open_time")
    df["ethbtc"] = df["eth_close"] / df["btc_close"]
    df["ethbtc_change_24h"] = df["ethbtc"].pct_change(288)
    return df[["open_time", "ethbtc_change_24h"]]


def build_augmented_panel(out_path):
    print(f"\n[step 1] Building augmented panel...", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"  loaded base panel {len(panel):,} rows ({time.time()-t0:.0f}s)", flush=True)

    # 1. ethbtc_change_24h
    t0 = time.time()
    eth = load_eth_btc_ratio()
    panel = panel.merge(eth, on="open_time", how="left")
    print(f"  attached ethbtc_change_24h ({time.time()-t0:.0f}s)", flush=True)

    # 2. xs_ret_disp_1d (cross-symbol std of return_1d at each timestamp)
    t0 = time.time()
    xs_disp = panel.groupby("open_time")["return_1d"].std().reset_index()
    xs_disp.columns = ["open_time", "xs_ret_disp_1d"]
    panel = panel.merge(xs_disp, on="open_time", how="left")
    print(f"  attached xs_ret_disp_1d ({time.time()-t0:.0f}s)", flush=True)

    panel.to_parquet(out_path, index=False)
    print(f"  saved {len(panel):,} rows × {panel.shape[1]} cols to {out_path}",
          flush=True)
    return panel


def train_fold_w23(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) &
                 (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) &
                (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100:
        return None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr[TARGET_COL].to_numpy(np.float32)
    yc = ca[TARGET_COL].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    preds = []
    for s in SEEDS:
        m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    pred_cols = ["symbol", "open_time", "alpha_A"]
    if "exit_time" in test_r.columns:
        pred_cols.append("exit_time")
    df_f = test_r[pred_cols].copy()
    df_f["pred"] = np.mean(preds, axis=0)
    df_f["fold"] = fold["fid"]
    return df_f


def get_listings():
    listings = {}
    klines_dir = REPO / "data/ml/test/parquet/klines"
    for sym_dir in klines_dir.iterdir():
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


def train_all_oos(panel):
    print(f"\n[step 2] Training all OOS folds with WINNER_23...", flush=True)
    feat_set = [f for f in WINNER_23 if f in panel.columns]
    missing = [f for f in WINNER_23 if f not in panel.columns]
    print(f"  features used: {len(feat_set)}", flush=True)
    if missing:
        print(f"  MISSING from panel: {missing}", flush=True)
        return None

    listings = get_listings()
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    folds = _multi_oos_splits(panel)
    print(f"  total folds: {len(folds)}", flush=True)

    all_preds = []
    t0_total = time.time()
    for fold in folds:
        t0 = time.time()
        cutoff = fold["cal_start"] - pd.Timedelta(days=MIN_HISTORY_DAYS)
        eligible = {s for s in panel["symbol"].unique()
                      if listings.get(s) and listings[s] <= cutoff}
        df_f = train_fold_w23(panel, fold, feat_set, eligible)
        if df_f is None:
            print(f"  fold {fold['fid']}: SKIP (insufficient data)", flush=True)
            continue
        all_preds.append(df_f)
        print(f"  fold {fold['fid']}: {len(df_f):,} preds  "
              f"(elapsed={time.time()-t0:.0f}s, total={time.time()-t0_total:.0f}s)",
              flush=True)

    all_pred = pd.concat(all_preds, ignore_index=True)
    if "exit_time" not in all_pred.columns:
        all_pred["exit_time"] = all_pred["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
    out_path = OUT / "all_predictions_w23.parquet"
    all_pred.to_parquet(out_path, index=False)
    print(f"\n  saved {len(all_pred):,} predictions to {out_path}", flush=True)
    return all_pred


def main():
    print("=== Phase Q: WINNER_23 full retrain ===\n", flush=True)
    print(f"  WINNER_21 features: {len(WINNER_21)}", flush=True)
    print(f"  Phase Q additions: {ADD_PHASE_Q}", flush=True)
    print(f"  WINNER_23 features: {len(WINNER_23)}\n", flush=True)

    panel_path = OUT / "panel_w23.parquet"
    if panel_path.exists():
        print(f"[step 1] Augmented panel already exists at {panel_path}", flush=True)
        panel = pd.read_parquet(panel_path)
        print(f"  {len(panel):,} rows × {panel.shape[1]} cols", flush=True)
    else:
        panel = build_augmented_panel(panel_path)

    pred_path = OUT / "all_predictions_w23.parquet"
    if pred_path.exists():
        print(f"\n[step 2] Predictions already exist at {pred_path}", flush=True)
        all_pred = pd.read_parquet(pred_path)
        print(f"  {len(all_pred):,} predictions, folds: {sorted(all_pred['fold'].unique())}",
              flush=True)
    else:
        all_pred = train_all_oos(panel)
        if all_pred is None:
            print("  ABORT: training failed", flush=True)
            return

    print(f"\n[step 2 complete] WINNER_23 predictions ready. Run step 3-5 next "
          f"(rebuild sleeves + V3.1 aggregation + validation).", flush=True)


if __name__ == "__main__":
    main()
