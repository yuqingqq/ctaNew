"""Step 62: V2 Ridge retrained on the HL-executable blue-chip universe
(on_hl & hl_day_vol_usd >= $2M => 44 symbols). Predictions producer for the
mean-reversion-exit backtest (Step 63). Mirrors Step 56/58 machinery.

Saves predictions.parquet with beta_pit added (needed for the dynamic BTC-hedge
sizing diagnostic in Step 63).
"""
from __future__ import annotations
import sys, time, importlib.util, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
REPO = Path("/home/yuqing/ctaNew")
sys.path.insert(0, str(REPO))


def _imp(name, rel):
    sp = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m

psl = _imp("psl", "scripts/phase_ah_sleeve.py")
s58 = _imp("s58", "linear_model/scripts/58_clean108_train.py")
from ml.research.alpha_v4_xs_1d import _multi_oos_splits, _slice

PANEL_111 = REPO / "outputs/vBTC_features_btc_only_111_full_pit/panel_btc_only_111.parquet"
HL_MAP = REPO / "outputs/vBTC_check_universe/panel110_hl_map.csv"
OUT = REPO / "linear_model/results/step62_bluechip44"
OUT.mkdir(parents=True, exist_ok=True)
OOS_FOLDS = list(range(1, 10))
MIN_HISTORY_DAYS = 60
VOL_THRESH = 2e6


def main():
    print("=" * 100, flush=True)
    print("  STEP 62: V2 Ridge on HL blue-chip (on_hl & vol>=$2M = 44 syms)", flush=True)
    print("=" * 100, flush=True)
    t0 = time.time()
    listings = s58.get_listings()

    hl = pd.read_csv(HL_MAP)
    keep = set(hl[(hl.on_hl) & (hl.hl_day_vol_usd >= VOL_THRESH)]["symbol"])
    panel = pd.read_parquet(PANEL_111)
    panel["open_time"] = pd.to_datetime(panel["open_time"], utc=True)
    panel = panel[panel["symbol"].isin(keep) & (panel["symbol"] != "BTCUSDT")].copy()
    syms = sorted(panel["symbol"].unique())
    print(f"  universe: {len(syms)} symbols, {len(panel):,} rows", flush=True)
    print(f"  {syms}", flush=True)

    folds_all = _multi_oos_splits(panel)
    fold0_train_idx = _slice(panel, folds_all[0])[0].index
    tr0 = panel.loc[fold0_train_idx]
    sig = tr0.groupby("symbol")["alpha_beta"].std()
    med = float(sig.dropna().median())
    panel["sigma_idio"] = panel["symbol"].map(sig).fillna(med).clip(lower=1e-6)
    panel = s58.build_target_z(panel, fold0_train_idx)
    print(f"  σ_idio fold-0 median {med:.5f}; target_z std {panel['target_z'].std():.3f}",
          flush=True)

    train_mask = panel["open_time"].between(
        _slice(panel, folds_all[0])[0].open_time.min(),
        _slice(panel, folds_all[0])[0].open_time.max())
    for s, t in panel.groupby("symbol")["open_time"].min().items():
        if s not in listings:
            t = t.tz_convert("UTC") if t.tz is not None else t.tz_localize("UTC")
            listings[s] = t

    X, feat_cols = s58.build_v2_features(panel, train_mask)
    print(f"  V2 features: {len(feat_cols)}", flush=True)
    panel_x = panel[["symbol", "open_time", "alpha_beta", "target_z",
                      "autocorr_pctile_7d"]].merge(
        X.drop(columns=["alpha_beta", "target_z", "autocorr_pctile_7d"]),
        on=["symbol", "open_time"], how="left")
    apd = s58.train_ridge(panel_x, folds_all, feat_cols)
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd["alpha_A"] = apd["alpha_beta"]

    cols_extra = ["symbol", "open_time", "return_pct", "exit_time"]
    if "beta_pit" in panel.columns:
        cols_extra.append("beta_pit")
    extra = panel[cols_extra].copy()
    extra["exit_time"] = pd.to_datetime(extra["exit_time"], utc=True)
    apd = apd.merge(extra, on=["symbol", "open_time"], how="left")

    target_t = sorted(apd[apd["fold"].isin(OOS_FOLDS)]["open_time"].unique())
    sampled_t = target_t[::psl.HORIZON_ENTRY]
    df_ic = s58.compute_trailing_ic(apd, sampled_t, 90)
    apd = apd.merge(df_ic, on=["symbol", "open_time"], how="left")
    apd["trail_ic"] = apd["trail_ic"].fillna(0)
    apd["pred_B"] = apd["pred_z"] * apd["trail_ic"]
    apd["pred"] = apd["pred_B"]

    cyc_ic = apd.dropna(subset=["alpha_beta"]).groupby("open_time").apply(
        lambda g: g["pred_z"].rank().corr(g["alpha_beta"].rank())
        if len(g) >= 5 else np.nan).dropna()
    print(f"  overall per-cycle IC: {cyc_ic.mean():+.4f}", flush=True)
    apd.to_parquet(OUT / "predictions.parquet", index=False)
    print(f"  saved predictions.parquet ({len(apd):,} rows, "
          f"{apd['symbol'].nunique()} syms)\nTotal: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
