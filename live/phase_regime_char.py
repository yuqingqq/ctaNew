"""Characterize the REGIME difference between in-sample (2025-10+) and 2024 (+ 2023, 2025-H1), in the SIDE regime.
Goal: find the PIT market-structure feature that explains WHY mean-reversion (the long leg esp.) died in 2024.
Metrics per period (side cycles only):
  reversion signal (the crux):
    * ac1_alpha  = mean per-symbol lag-1 autocorr of 4h residual alpha  (NEG = reverts=good; POS = trends=bad)
    * rev_ic_1d  = mean per-cycle xs corr( -return_1d , fwd 24h alpha )  (does "buy yesterday's loser" pay?)
  market structure:
    * btc_rvol   = mean btc_rvol_7d          (vol regime)
    * xs_corr    = mean corr_to_btc_1d       (beta-driven co-movement; high = trending together)
    * xs_disp    = mean per-cycle std(return_1d)   (dispersion)
    * fund_z     = mean |funding_rate_z_7d|  (crowding)
    * btc30_abs  = mean |btc_ret_30d|        (trend magnitude within the side band)
All PIT (features known at decision time). Uses full-history regime.csv.
"""
import pandas as pd, numpy as np
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
R = "/home/yuqing/ctaNew"; K = 6

reg = pd.read_csv(f"{R}/live/state/longtail/full_regime/state/regime.csv")
reg["open_time"] = pd.to_datetime(reg["open_time"], utc=True)
reg = reg[["open_time", "regime", "btc_ret_30d"]].drop_duplicates("open_time")

cols = ["symbol", "open_time", "alpha_vs_btc_realized", "return_1d", "corr_to_btc_1d", "rvol_7d",
        "btc_rvol_7d", "funding_rate_z_7d"]
pan = pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet", columns=cols)
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
pan["fa24"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s: s.shift(-1).rolling(K).sum().shift(-(K-1)))
pan = pan.merge(reg, on="open_time", how="left")

PERIODS = {
    "IN-SAMPLE 2025-10+": lambda d: d.open_time >= pd.Timestamp("2025-10-04", tz="UTC"),
    "2023":              lambda d: (d.open_time.dt.year == 2023),
    "2024":              lambda d: (d.open_time.dt.year == 2024),
    "2025-H1":           lambda d: (d.open_time.dt.year == 2025) & (d.open_time < pd.Timestamp("2025-10-04", tz="UTC")),
}

def ac1(g):
    x = g["alpha_vs_btc_realized"].dropna().to_numpy()
    if len(x) < 30: return np.nan
    return np.corrcoef(x[:-1], x[1:])[0, 1]

print(f"  {'period':>20} {'ac1_alpha':>9} {'rev_ic1d':>8} {'btc_rvol':>8} {'xs_corr':>8} {'xs_disp':>8} {'|fund_z|':>8} {'|btc30|':>8} {'n_cyc':>6}")
for name, f in PERIODS.items():
    d = pan[f(pan) & (pan.regime == "side")].copy()
    if not len(d): continue
    ac = d.groupby("symbol").apply(ac1).dropna()
    # per-cycle reversion IC: xs corr(-return_1d, fwd24) — does buying the recent loser pay over 24h?
    rics = []
    for _, g in d.groupby("open_time"):
        g2 = g.dropna(subset=["return_1d", "fa24"])
        if len(g2) < 6: continue
        rics.append(spearmanr(-g2["return_1d"], g2["fa24"]).correlation)
    rics = np.array([x for x in rics if np.isfinite(x)])
    xs_disp = d.groupby("open_time")["return_1d"].std().mean() * 1e4
    print(f"  {name:>20} {ac.mean():+9.4f} {np.nanmean(rics):+8.4f} {d['btc_rvol_7d'].mean():8.4f} "
          f"{d['corr_to_btc_1d'].mean():8.3f} {xs_disp:8.0f} {d['funding_rate_z_7d'].abs().mean():8.3f} "
          f"{d['btc_ret_30d'].abs().mean():8.4f} {d['open_time'].dt.floor('4h').nunique():6d}")
