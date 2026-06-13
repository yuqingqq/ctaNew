"""Retrain WINNER_21 without sym_id, build a separate audit panel, run V3.1.

Goal: test whether sym_id is doing real work in WINNER_21 or just memorizing
alphabetical-rank intercepts. If V3.1 Sharpe survives sym_id removal, the model
becomes structurally more universe-portable for free.

Pipeline (mirrors alpha_vBTC_build_audit_panel.py exactly, only feat_set changes):
  1. Drop sym_id from WINNER_21 → 20-feature set
  2. Train 10 folds × 5 seeds with PIT 60d eligibility (same as production)
  3. Save predictions to outputs/vBTC_audit_panel_no_sym_id/all_predictions.parquet
  4. Re-use scripts/phase_ah_sleeve.py + diag_v31_no_ic_filter.py machinery to
     run V3.1 head-to-head with original sym_id-having predictions.
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
OUT_DIR = REPO / "outputs/vBTC_audit_panel_no_sym_id"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
OOS_FOLDS = list(range(1, 10))
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
WINNER_21 = ([f for f in V6_CLEAN_28 if f not in ALL_DROPS]
             + FUNDING_LEAN + ADD_CROSS_BTC + ADD_MORE_FUNDING)

# THE ONE CHANGE
WINNER_NO_SYMID = [f for f in WINNER_21 if f != "sym_id"]


def get_listings():
    L = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        f = sorted(m5.glob("*.parquet"))
        if not f: continue
        try: L[d.name] = pd.Timestamp(f[0].stem, tz="UTC")
        except Exception: pass
    return L


def train_fold_restricted(panel, fold, feat_set, eligible_syms):
    train, cal, test = _slice(panel, fold)
    tr = train[(train["autocorr_pctile_7d"] >= THRESHOLD) & (train["symbol"].isin(eligible_syms))]
    ca = cal[(cal["autocorr_pctile_7d"] >= THRESHOLD) & (cal["symbol"].isin(eligible_syms))]
    test_r = test[test["symbol"].isin(eligible_syms)].copy()
    if len(tr) < 1000 or len(ca) < 200 or len(test_r) < 100:
        return None, None
    Xt = tr[feat_set].to_numpy(np.float32)
    Xc = ca[feat_set].to_numpy(np.float32)
    Xtest = test_r[feat_set].to_numpy(np.float32)
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mask_t = ~np.isnan(yt); mask_c = ~np.isnan(yc)
    if mask_t.sum() < 1000 or mask_c.sum() < 200: return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


def main():
    print("=== Retrain WINNER_21 WITHOUT sym_id ===\n", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    feat_set = [f for f in WINNER_NO_SYMID if f in panel.columns]
    print(f"Feature set ({len(feat_set)}): {feat_set}\n", flush=True)
    assert "sym_id" not in feat_set, "sym_id should NOT be in feature set"

    folds_all = _multi_oos_splits(panel)
    listings = get_listings()
    panel_first = panel.groupby("symbol")["open_time"].min()
    for s, t in panel_first.items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t
    panel_syms = set(panel["symbol"].unique())

    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    print("--- Train 10 folds × 5 seeds ---", flush=True)
    all_preds = []
    t_start = time.time()
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold_restricted(panel, folds_all[fid], feat_set, eligible)
        if td is None: continue
        cols = ["symbol", "open_time", "alpha_A", "return_pct"]
        if "exit_time" in td.columns: cols.append("exit_time")
        df = td[cols].copy()
        df["pred"] = p; df["fold"] = fid
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: n={len(td):,} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  total train time: {time.time()-t_start:.0f}s\n", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    out = OUT_DIR / "all_predictions.parquet"
    apd.to_parquet(out, index=False)
    print(f"Saved {len(apd):,} prediction rows → {out}", flush=True)

    # Quick sanity: per-symbol pred std should still vary (model should learn)
    pred_std_per_sym = apd.groupby("symbol")["pred"].std().describe()
    print(f"\nPer-symbol pred std distribution:\n{pred_std_per_sym}", flush=True)

    # Per-cycle IC sanity
    cyc_ic = apd.dropna(subset=["alpha_A"]).groupby("open_time").apply(
        lambda g: g["pred"].rank().corr(g["alpha_A"].rank()) if len(g) >= 5 else np.nan
    ).dropna()
    print(f"\nPer-cycle IC: mean={cyc_ic.mean():+.4f}, median={cyc_ic.median():+.4f}, "
          f"std={cyc_ic.std():.4f}, n={len(cyc_ic)}", flush=True)


if __name__ == "__main__":
    main()
