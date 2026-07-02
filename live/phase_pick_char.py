"""What kind of symbols did the strategy PICK in 2024 vs in-sample, and why did BOTH momentum & reversion fail?
Uses the actual selected_long/selected_short flags from a run's predictions.parquet. Filters to SIDE regime.
For picks (long & short) compares 2024 vs in-sample on:
  - feature profile: rvol_7d, idio_vol_to_btc_1d, corr_to_btc_1d, atr_pct, |funding_z|, ret_3d, autocorr_pctile_7d
  - the NAMES most frequently picked
  - forward 24h residual alpha of picks + its lag-1 autocorr (structure vs noise: ~0 => neither mom nor rev works)
"""
import pandas as pd, numpy as np
import warnings; warnings.filterwarnings("ignore")
R = "/home/yuqing/ctaNew"; K = 6
FEATS = ["rvol_7d", "idio_vol_to_btc_1d", "corr_to_btc_1d", "atr_pct", "funding_rate_z_7d", "ret_3d", "autocorr_pctile_7d"]

pan = pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",
                      columns=["symbol", "open_time", "alpha_vs_btc_realized"] + FEATS)
pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True); pan = pan.sort_values(["symbol", "open_time"])
pan["fa24"] = pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s: s.shift(-1).rolling(K).sum().shift(-(K-1)))

def load(run):
    p = pd.read_parquet(f"{R}/live/state/longtail/{run}/state/predictions.parquet")
    p["open_time"] = pd.to_datetime(p["open_time"], utc=True)
    reg = pd.read_csv(f"{R}/live/state/longtail/{run}/state/regime.csv"); reg["open_time"] = pd.to_datetime(reg["open_time"], utc=True)
    p = p.merge(reg[["open_time", "regime"]].drop_duplicates("open_time"), on="open_time", how="left")
    return p.merge(pan, on=["symbol", "open_time"], how="left")

def prof(df, label):
    d = df[df.regime == "side"]
    uni = d  # universe this cycle
    L = d[d.get("selected_long", False) == True]
    S = d[d.get("selected_short", False) == True]
    print(f"\n=== {label}  side cycles={d['open_time'].nunique()}, longs={len(L)}, shorts={len(S)} ===")
    print(f"  {'feature':>20} {'universe':>9} {'LONG pick':>10} {'SHORT pick':>10}")
    for f in FEATS:
        if f not in d.columns: continue
        u = d[f].median(); lv = L[f].median() if len(L) else np.nan; sv = S[f].median() if len(S) else np.nan
        print(f"  {f:>20} {u:9.3f} {lv:10.3f} {sv:10.3f}")
    # forward alpha of picks + autocorr structure
    for nm, g in [("LONG", L), ("SHORT", S)]:
        fa = g["fa24"].dropna()
        # lag-1 autocorr of picked names' fwd alpha (per symbol then avg)
        acs = []
        for sym, gs in g.dropna(subset=["alpha_vs_btc_realized"]).groupby("symbol"):
            x = gs.sort_values("open_time")["alpha_vs_btc_realized"].to_numpy()
            if len(x) >= 20: acs.append(np.corrcoef(x[:-1], x[1:])[0, 1])
        print(f"  {nm} picks: fwd24 alpha mean {fa.mean()*1e4:+.1f}bps | resid ac1 {np.nanmean(acs):+.4f} (n_sym {len(acs)})")
    # top names
    print(f"  top LONG names: {L['symbol'].value_counts().head(8).to_dict()}")
    print(f"  top SHORT names: {S['symbol'].value_counts().head(8).to_dict()}")

IS = load("is_pick")
OOS = load("oos_pick")
prof(IS, "IN-SAMPLE 2025-10+")
prof(OOS[OOS.open_time.dt.year == 2024], "OOS 2024")
prof(OOS[OOS.open_time.dt.year == 2023], "OOS 2023")
