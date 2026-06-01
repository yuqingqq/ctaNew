"""X73 — PIT regime verification on 3-year V0 predictions.

Uses ONLY trailing (point-in-time) BTC features to define regimes:
  - trend: BTC trailing 30d return  (bull if > +10%)
  - vol:   BTC trailing 30d realized vol, percentile over trailing 1yr

Then buckets the 3-year V0 K=3 basket performance by PIT regime to verify
which regimes the strategy actually has edge in (tradeable, no hindsight).

Metric per regime: per-cycle K=3 basket spread = mean(top3 alpha) - mean(bot3 alpha),
plus per-cycle IC. These approximate the sleeve's long/short edge without
needing continuous time.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO / "research/convexity_portable_2026-05-20/results"
CACHE = OUT / "_cache"
KLINES = REPO / "data/ml/test/parquet/klines"


def main():
    print("=== X73 PIT regime verification (3-year V0) ===\n")
    # PIT BTC features (trailing only)
    files = sorted((KLINES/"BTCUSDT"/"5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float64)
    ret_30d = btc/btc.shift(8640) - 1                       # trailing 30d return (PIT)
    logret = np.log(btc/btc.shift(1))
    rv_30d = logret.rolling(8640, min_periods=2880).std() * np.sqrt(8640)   # trailing 30d vol
    rv_pctile = rv_30d.rolling(8640*12, min_periods=8640).rank(pct=True)    # vs trailing 1yr
    ma200 = btc.rolling(8640*200//30, min_periods=2880).mean()  # ~200d MA proxy (use 57600 bars)
    feat = pd.DataFrame({
        "btc_ret_30d": ret_30d, "btc_rv_pctile": rv_pctile,
        "btc_above_ma": (btc > ma200).astype(float),
    }).reset_index()
    feat["open_time"] = pd.to_datetime(feat["open_time"], utc=True)

    apd = pd.read_parquet(CACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    # restrict to 4h-aligned cycles
    apd = apd[(apd["open_time"].dt.hour % 4 == 0) & (apd["open_time"].dt.minute == 0)]
    m = apd.merge(feat, on="open_time", how="left").dropna(subset=["btc_ret_30d","btc_rv_pctile"])
    print(f"Cycles: {m['open_time'].nunique():,}, rows: {len(m):,}")

    # PIT regime labels
    m["trend_pit"] = np.where(m["btc_ret_30d"] > 0.10, "bull",
                       np.where(m["btc_ret_30d"] < -0.10, "bear", "side"))
    m["vol_pit"] = np.where(m["btc_rv_pctile"] > 0.67, "highvol",
                     np.where(m["btc_rv_pctile"] < 0.33, "lowvol", "midvol"))

    # Per-cycle K=3 basket spread + IC
    def cycle_stats(g):
        if len(g) < 8 or g["pred"].std() == 0:
            return pd.Series({"spread": np.nan, "ic": np.nan})
        gg = g.sort_values("pred")
        bot = gg.head(3)["alpha_A"].mean()
        top = gg.tail(3)["alpha_A"].mean()
        ic = g["pred"].corr(g["alpha_A"])
        return pd.Series({"spread": (top - bot)*10000, "ic": ic})  # spread in bps

    cyc = m.groupby("open_time").apply(cycle_stats).reset_index()
    # attach regime (per cycle — take first)
    reg = m.groupby("open_time")[["trend_pit","vol_pit","btc_ret_30d","btc_rv_pctile"]].first().reset_index()
    cyc = cyc.merge(reg, on="open_time")
    cyc = cyc.dropna(subset=["spread"])

    def sharpe(x):
        x = x.dropna()
        return x.mean()/x.std()*np.sqrt(len(x)/ (len(cyc)/ (365*1.0)) ) if len(x)>2 and x.std()>0 else np.nan

    def ann_sharpe(x):
        # cycles are 4h → 6/day → annualize
        x = x.dropna()
        if len(x) < 3 or x.std() == 0: return np.nan
        return x.mean()/x.std() * np.sqrt(6*365)

    print(f"\n=== By PIT TREND regime (trailing 30d return) ===")
    print(f"  {'regime':<8} {'cycles':>8} {'mean_spread_bps':>16} {'spread_Sharpe':>14} {'mean_IC':>9}")
    for r in ["bull","side","bear"]:
        sub = cyc[cyc["trend_pit"]==r]
        if len(sub)==0: continue
        print(f"  {r:<8} {len(sub):>8} {sub['spread'].mean():>+16.2f} {ann_sharpe(sub['spread']):>+14.2f} {sub['ic'].mean():>+9.4f}")

    print(f"\n=== By PIT VOL regime (trailing 30d vol percentile) ===")
    print(f"  {'regime':<8} {'cycles':>8} {'mean_spread_bps':>16} {'spread_Sharpe':>14} {'mean_IC':>9}")
    for r in ["lowvol","midvol","highvol"]:
        sub = cyc[cyc["vol_pit"]==r]
        if len(sub)==0: continue
        print(f"  {r:<8} {len(sub):>8} {sub['spread'].mean():>+16.2f} {ann_sharpe(sub['spread']):>+14.2f} {sub['ic'].mean():>+9.4f}")

    print(f"\n=== 2D: TREND × VOL ===")
    print(f"  {'trend':<6} {'vol':<8} {'cycles':>7} {'spread_bps':>11} {'Sharpe':>8} {'IC':>8}")
    for t in ["bull","side","bear"]:
        for v in ["lowvol","midvol","highvol"]:
            sub = cyc[(cyc["trend_pit"]==t)&(cyc["vol_pit"]==v)]
            if len(sub)<10: continue
            print(f"  {t:<6} {v:<8} {len(sub):>7} {sub['spread'].mean():>+11.2f} {ann_sharpe(sub['spread']):>+8.2f} {sub['ic'].mean():>+8.4f}")

    cyc.to_parquet(OUT/"X73_pit_regime_cycles.parquet", index=False)
    print(f"\nSaved cycle-level stats. (PIT = tradeable, no hindsight.)")


if __name__ == "__main__":
    main()
