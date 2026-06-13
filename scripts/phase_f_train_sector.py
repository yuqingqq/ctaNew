"""Phase F4: train LGBM on dedup_23_fund (baseline) vs dedup_26_fund_sector.

Mirrors the existing audit-panel build pattern (folds 0-9, 5-seed ensemble,
kline-listing PIT eligibility), but for two feature sets side-by-side.

Outputs:
  outputs/vBTC_sector_features/audit_dedup_23_fund/all_predictions.parquet
  outputs/vBTC_sector_features/audit_dedup_26_fund_sector/all_predictions.parquet
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

PANEL_PATH = REPO / "outputs/vBTC_features/panel_variants_with_funding.parquet"
KLINES_DIR = REPO / "data/ml/test/parquet/klines"
CLUSTERS_PATH = REPO / "config/clusters_v1.json"
OUT_DIR = REPO / "outputs/vBTC_sector_features"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 48
RC = 0.50
THRESHOLD = 1 - RC
SEEDS = (42, 1337, 7, 19, 2718)
ALL_FOLDS = list(range(10))
MIN_HISTORY_DAYS = 60

V6_CLEAN_28 = list(XS_FEATURE_COLS_V6_CLEAN)
DEDUP_DROPS = [
    "return_1d_xs_rank", "bk_ret_48b", "volume_ma_50",
    "ema_slope_20_1h", "ema_slope_20_1h_xs_rank",
    "vwap_zscore_xs_rank", "vwap_zscore",
]
DEDUP_21 = [f for f in V6_CLEAN_28 if f not in DEDUP_DROPS]
FUNDING_LEAN = ["funding_rate", "funding_rate_z_7d"]
DEDUP_23_FUND = DEDUP_21 + FUNDING_LEAN
SECTOR_FEATS = ["own_cluster_ret_1d", "relative_to_cluster_1d", "cluster_dispersion_1d"]
DEDUP_26_FUND_SECTOR = DEDUP_23_FUND + SECTOR_FEATS


def get_listing_dates_from_klines():
    listings = {}
    for sym_dir in KLINES_DIR.iterdir():
        if not sym_dir.is_dir(): continue
        m5 = sym_dir / "5m"
        if not m5.exists(): continue
        files = sorted(m5.glob("*.parquet"))
        if not files: continue
        try:
            listings[sym_dir.name] = pd.Timestamp(files[0].stem, tz="UTC")
        except Exception:
            continue
    return listings


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
    if mask_t.sum() < 1000 or mask_c.sum() < 200:
        return None, None
    preds = []
    for s in SEEDS:
        m = _train(Xt[mask_t], yt[mask_t], Xc[mask_c], yc[mask_c], seed=s)
        preds.append(m.predict(Xtest, num_iteration=m.best_iteration))
    return test_r, np.mean(preds, axis=0)


def run_one_feature_set(panel, folds_all, feat_set, eligibility_at, label):
    out_sub = OUT_DIR / f"audit_{label}"
    out_sub.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Training {label} ({len(feat_set)} features) ===", flush=True)
    missing = [f for f in feat_set if f not in panel.columns]
    if missing:
        print(f"  WARNING: missing features: {missing}", flush=True)
        feat_set = [f for f in feat_set if f in panel.columns]

    all_preds = []
    for fid in ALL_FOLDS:
        if fid >= len(folds_all): continue
        t0 = time.time()
        eligible = eligibility_at(folds_all[fid]["cal_start"])
        td, p = train_fold_restricted(panel, folds_all[fid], feat_set, eligible)
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
    apd.to_parquet(out_sub / "all_predictions.parquet", index=False)
    print(f"  saved {label} all_predictions ({len(apd):,} rows)", flush=True)


def main():
    print(f"=== Phase F4: dedup_23_fund vs dedup_26_fund_sector ===\n", flush=True)
    panel = pd.read_parquet(PANEL_PATH)
    print(f"  Panel: {len(panel):,} rows, {panel.symbol.nunique()} syms", flush=True)

    t0 = time.time()
    panel, sect_cols = add_sector_features(panel, CLUSTERS_PATH)
    print(f"  Sector features added: {sect_cols} in {time.time()-t0:.0f}s", flush=True)

    folds_all = _multi_oos_splits(panel)
    print(f"  Folds: {len(folds_all)}", flush=True)

    listings = get_listing_dates_from_klines()
    panel_first_obs = panel.groupby("symbol")["open_time"].min()
    for sym, t in panel_first_obs.items():
        if sym not in listings:
            ts = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[sym] = ts

    panel_syms = set(panel["symbol"].unique())

    def eligibility_at(timestamp):
        if isinstance(timestamp, (int, np.integer)):
            ts = pd.Timestamp(timestamp, unit="ms", tz="UTC")
        else:
            ts = pd.Timestamp(timestamp)
            if ts.tz is None: ts = ts.tz_localize("UTC")
        cutoff = ts - pd.Timedelta(days=MIN_HISTORY_DAYS)
        return {s for s in panel_syms if listings.get(s) and listings[s] <= cutoff}

    run_one_feature_set(panel, folds_all, DEDUP_23_FUND, eligibility_at, "dedup_23_fund")
    run_one_feature_set(panel, folds_all, DEDUP_26_FUND_SECTOR, eligibility_at, "dedup_26_fund_sector")


if __name__ == "__main__":
    main()
