"""Phase G2: train WINNER_21 + data-driven sector features on 111-panel.

Reuses the WINNER_21 baseline already trained in E5a v2 at
outputs/vBTC_audit_panel_expanded/all_predictions.parquet (no retrain needed).
Trains ONLY the treatment (WINNER_21 + 3 data-driven sector features).
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
from features_ml.sector_features import add_sector_features
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice, _train

PANEL_PATH = REPO / "outputs/vBTC_features_expanded/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
CLUSTERS_DD = REPO / "config/clusters_data_driven_v1.json"
OUT_DIR = REPO / "outputs/vBTC_audit_panel_expanded_sector_dd"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

SECTOR_DD_COLS = ["own_cluster_ret_1d_dd", "relative_to_cluster_1d_dd", "cluster_dispersion_1d_dd"]


def get_listing_dates():
    listings = {}
    for d in KLINES_DIR.iterdir():
        if not d.is_dir(): continue
        m5 = d / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        try:
            listings[d.name] = pd.Timestamp(files[0].stem, tz="UTC")
        except Exception:
            continue
    return listings


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
    yt = tr["target_A"].to_numpy(np.float32)
    yc = ca["target_A"].to_numpy(np.float32)
    mt = ~np.isnan(yt); mc = ~np.isnan(yc)
    if mt.sum() < 1000 or mc.sum() < 200:
        return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mt], yt[mt], Xc[mc], yc[mc], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


def main():
    print("=== Phase G2: WINNER_21 + sector_dd on 111-panel ===\n", flush=True)

    panel = pd.read_parquet(PANEL_PATH)
    print(f"  panel: {len(panel):,} rows, {panel.symbol.nunique()} syms", flush=True)

    # Build sector_dd features with rename so column names are explicit
    t0 = time.time()
    panel_sect, sect_cols = add_sector_features(panel, CLUSTERS_DD)
    rename_map = {
        "own_cluster_ret_1d": "own_cluster_ret_1d_dd",
        "relative_to_cluster_1d": "relative_to_cluster_1d_dd",
        "cluster_dispersion_1d": "cluster_dispersion_1d_dd",
    }
    panel = panel_sect.rename(columns=rename_map)
    print(f"  sector_dd features added in {time.time()-t0:.0f}s: {SECTOR_DD_COLS}", flush=True)

    if "sym_id" in panel.columns:
        usyms = sorted(panel["symbol"].unique())
        m = {s: i for i, s in enumerate(usyms)}
        panel["sym_id"] = panel["symbol"].map(m).astype("int32")

    feat_set = WINNER_21 + SECTOR_DD_COLS
    missing = [f for f in feat_set if f not in panel.columns]
    if missing:
        print(f"  WARNING missing: {missing}", flush=True)
        feat_set = [f for f in feat_set if f in panel.columns]

    folds_all = _multi_oos_splits(panel)
    print(f"  Folds: {len(folds_all)}", flush=True)

    listings = get_listing_dates()
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
        df["pred"] = p; df["fold"] = fid
        if "exit_time" not in df.columns:
            df["exit_time"] = df["open_time"] + pd.Timedelta(minutes=HORIZON * 5)
        all_preds.append(df)
        print(f"  fold {fid}: eligible={len(eligible)}, n_test={len(td):,} "
              f"({time.time()-t0:.0f}s)", flush=True)

    apd = pd.concat(all_preds, ignore_index=True).sort_values(["open_time", "symbol"])
    apd.to_parquet(OUT_DIR / "all_predictions.parquet", index=False)
    print(f"\n  saved → {OUT_DIR}/all_predictions.parquet  ({len(apd):,} rows)",
          flush=True)


if __name__ == "__main__":
    main()
