"""Phase SEG: symbol-segmented LGBM (A2 from ranking-optimization review).

Pre-registered:
  - For each fold, split universe by per-symbol median `corr_to_btc_1d` over
    the TRAINING data (PIT: only data with open_time < fold.train_end).
  - Threshold = median of per-symbol medians → high-corr group and low-corr
    group, each ~25 symbols.
  - Train 2 independent LGBM 5-seed ensembles per fold (same WINNER_21
    features, same hyperparameters as production).
  - At inference, predict each symbol using its group's model.
  - Save combined all_predictions_segmented.parquet, then rebuild sleeves +
    run V3.1 + 6-gate validation.

Hypothesis: cross-sectional alpha-generating processes differ for BTC-driven
symbols vs idiosyncratic symbols. Segmenting lets the model specialize per
group, which should lift per-symbol IC on the 11 negative-IC symbols.
"""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))

from features_ml.cross_sectional import XS_FEATURE_COLS_V6_CLEAN
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

OUT = REPO / "outputs/vBTC_phase_SEG"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 48
SEEDS = (42, 1337, 7, 19, 2718)
TARGET_COL = "target_A"
MIN_HISTORY_DAYS = 60
RC = 0.50
THRESHOLD = 1 - RC
SEGMENT_FEATURE = "corr_to_btc_1d"

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


def split_universe_at_fold(train_slice, feat=SEGMENT_FEATURE):
    """Compute per-symbol median of feat over training slice, split at panel median."""
    per_sym_median = train_slice.groupby("symbol")[feat].median()
    threshold = per_sym_median.median()
    high_group = set(per_sym_median[per_sym_median > threshold].index)
    low_group = set(per_sym_median[per_sym_median <= threshold].index)
    return high_group, low_group, threshold


def train_segmented_fold(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr_full = train[(train["autocorr_pctile_7d"] >= THRESHOLD) &
                       (train["symbol"].isin(eligible_syms))]
    ca_full = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) &
                       (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr_full) < 1000 or len(ca_full) < 200 or len(test_r) < 100:
        return None

    high_group, low_group, threshold = split_universe_at_fold(tr_full)

    out_frames = []
    for group_name, group_syms in [("high", high_group), ("low", low_group)]:
        tr_g = tr_full[tr_full["symbol"].isin(group_syms)]
        ca_g = ca_full[ca_full["symbol"].isin(group_syms)]
        test_g = test_r[test_r["symbol"].isin(group_syms)].copy()
        if len(tr_g) < 500 or len(ca_g) < 100 or len(test_g) < 10:
            continue
        Xt = tr_g[feat_set].to_numpy(np.float32)
        Xc = ca_g[feat_set].to_numpy(np.float32)
        Xtest = test_g[feat_set].to_numpy(np.float32)
        yt = tr_g[TARGET_COL].to_numpy(np.float32)
        yc = ca_g[TARGET_COL].to_numpy(np.float32)
        mt = ~np.isnan(yt); mc = ~np.isnan(yc)
        preds = []
        for s in SEEDS:
            m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
            preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
        pred_cols = ["symbol", "open_time", "alpha_A"]
        if "exit_time" in test_g.columns:
            pred_cols.append("exit_time")
        df_g = test_g[pred_cols].copy()
        df_g["pred"] = np.mean(preds, axis=0)
        df_g["fold"] = fold["fid"]
        df_g["segment"] = group_name
        out_frames.append(df_g)
    if not out_frames:
        return None
    return pd.concat(out_frames, ignore_index=True), threshold, len(high_group), len(low_group)


def main():
    print("=== Phase SEG: symbol-segmented LGBM ===\n", flush=True)
    print(f"  Pre-registered split: per-symbol median of {SEGMENT_FEATURE}",
          flush=True)
    print(f"  Threshold: median across symbols (PIT, per fold)", flush=True)
    print(f"  2 independent LGBM 5-seed ensembles per fold\n", flush=True)

    print("  loading panel...", flush=True)
    t0 = time.time()
    panel = pd.read_parquet(
        REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet")
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    print(f"  panel {len(panel):,} rows ({time.time()-t0:.0f}s)", flush=True)

    feat_set = [f for f in WINNER_21 if f in panel.columns]
    print(f"  features: {len(feat_set)}", flush=True)

    listings = get_listings()
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    folds = _multi_oos_splits(panel)
    print(f"  total folds: {len(folds)}\n", flush=True)

    all_preds = []
    t0_total = time.time()
    for fold in folds:
        t0 = time.time()
        cutoff = fold["cal_start"] - pd.Timedelta(days=MIN_HISTORY_DAYS)
        eligible = {s for s in panel["symbol"].unique()
                      if listings.get(s) and listings[s] <= cutoff}
        result = train_segmented_fold(panel, fold, feat_set, eligible)
        if result is None:
            print(f"  fold {fold['fid']}: SKIP", flush=True)
            continue
        df_f, thr, n_hi, n_lo = result
        all_preds.append(df_f)
        print(f"  fold {fold['fid']}: {len(df_f):,} preds  "
              f"(high={n_hi} syms, low={n_lo} syms, thr={thr:.3f})  "
              f"elapsed={time.time()-t0:.0f}s, total={time.time()-t0_total:.0f}s",
              flush=True)

    all_pred = pd.concat(all_preds, ignore_index=True)
    if "exit_time" not in all_pred.columns:
        all_pred["exit_time"] = all_pred["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
    out_path = OUT / "all_predictions_segmented.parquet"
    all_pred.to_parquet(out_path, index=False)
    print(f"\n  saved {len(all_pred):,} predictions to {out_path}", flush=True)
    print(f"  total time: {time.time()-t0_total:.0f}s", flush=True)

    # Quick IC diagnostic
    print(f"\n  Per-cycle IC comparison (Segmented vs WINNER_21 MSE baseline):",
          flush=True)
    base = pd.read_parquet(REPO / "outputs/vBTC_audit_panel/all_predictions.parquet")
    base = base.dropna(subset=["pred", "alpha_A"])
    base = base[base["fold"].isin(range(1, 10))]
    seg_oos = all_pred[all_pred["fold"].isin(range(1, 10))].dropna(
        subset=["pred", "alpha_A"])

    base_ics = []
    for t, g in base.groupby("open_time"):
        if len(g) < 10: continue
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        if not pd.isna(ic): base_ics.append(ic)
    seg_ics = []
    for t, g in seg_oos.groupby("open_time"):
        if len(g) < 10: continue
        ic = g["pred"].rank().corr(g["alpha_A"].rank())
        if not pd.isna(ic): seg_ics.append(ic)

    base_ics = np.array(base_ics); seg_ics = np.array(seg_ics)
    print(f"    Baseline (1 model):  mean IC = {base_ics.mean():+.4f}  "
          f"median = {np.median(base_ics):+.4f}  pct_pos = {(base_ics>0).mean()*100:.1f}%",
          flush=True)
    print(f"    Segmented (2 models): mean IC = {seg_ics.mean():+.4f}  "
          f"median = {np.median(seg_ics):+.4f}  pct_pos = {(seg_ics>0).mean()*100:.1f}%",
          flush=True)
    print(f"    Δ mean IC: {seg_ics.mean() - base_ics.mean():+.4f}", flush=True)

    # Per-symbol IC comparison
    print(f"\n  Per-symbol IC comparison (focus on previously negative-IC symbols):",
          flush=True)
    neg_syms = ["ETHUSDT", "BIOUSDT", "JTOUSDT", "BNBUSDT", "PENGUUSDT",
                 "ZECUSDT", "ENAUSDT", "BCHUSDT", "PUMPUSDT", "JUPUSDT", "ONDOUSDT"]
    print(f"    {'symbol':<14}  {'baseline':>10}  {'segmented':>10}  {'Δ':>8}",
          flush=True)
    for sym in neg_syms:
        b = base[base["symbol"] == sym]
        s = seg_oos[seg_oos["symbol"] == sym]
        if len(b) < 100 or len(s) < 100: continue
        ic_b = b["pred"].rank().corr(b["alpha_A"].rank())
        ic_s = s["pred"].rank().corr(s["alpha_A"].rank())
        print(f"    {sym:<14}  {ic_b:>+10.4f}  {ic_s:>+10.4f}  {ic_s-ic_b:>+8.4f}",
              flush=True)


if __name__ == "__main__":
    main()
