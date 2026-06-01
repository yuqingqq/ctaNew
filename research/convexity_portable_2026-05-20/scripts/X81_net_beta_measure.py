"""X81 — Measure the K=3 basket's net BTC-beta per regime (3-year V0 preds).

Confirms (with data) the hypothesis: is the traded basket net-short-beta in bull,
and does that net-beta drag realized PnL?

Method (PIT):
  1. Per-sym trailing 30d beta vs BTC (cov/var, shifted) from klines.
  2. For each 4h cycle: long = top-3 by pred, short = bottom-3 by pred.
     net_beta = mean(long betas) - mean(short betas).
  3. PIT regime per cycle: btc trailing 30d return (bull/side/bear).
  4. Report: mean net_beta by regime, % cycles net-short, and corr(net_beta,
     realized basket PnL) to see if net-beta hurts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
KLINES = REPO/"data/ml/test/parquet/klines"


def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def main():
    print("=== X81 net-beta of K=3 basket per regime (3yr V0) ===\n", flush=True)
    apd = pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"] = pd.to_datetime(apd["open_time"], utc=True)
    apd = apd[(apd["open_time"].dt.hour % 4 == 0) & (apd["open_time"].dt.minute == 0)]
    syms = sorted(apd["symbol"].unique())
    print(f"Preds: {len(apd):,} rows, {apd['open_time'].nunique():,} cycles, {len(syms)} syms")

    # BTC + per-sym trailing 30d beta (4h bars), PIT shifted
    btc = load_close("BTCUSDT")
    btc4 = btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    btc_ret = np.log(btc4/btc4.shift(1))
    btc_var = btc_ret.rolling(180, min_periods=42).var()  # 30d at 4h bars

    beta_rows = []
    for i, sym in enumerate(syms, 1):
        c = load_close(sym)
        if c is None: continue
        c4 = c[(c.index.hour%4==0)&(c.index.minute==0)]
        r = np.log(c4/c4.shift(1))
        ri, bi = r.align(btc_ret, join="inner")
        cov = ri.rolling(180, min_periods=42).cov(bi)
        var = btc_var.reindex(cov.index)
        beta = (cov/var.replace(0,np.nan)).shift(1)  # PIT
        beta_rows.append(pd.DataFrame({"symbol":sym,"open_time":beta.index,"beta":beta.values}))
        if i%10==0: print(f"  beta {i}/{len(syms)}", flush=True)
    betas = pd.concat(beta_rows, ignore_index=True)
    betas["open_time"] = pd.to_datetime(betas["open_time"], utc=True)

    m = apd.merge(betas, on=["symbol","open_time"], how="left")

    # BTC 30d return regime (PIT)
    btc_30d = (btc4/btc4.shift(180)-1)
    reg = btc_30d.to_frame("btc_ret_30d").reset_index()
    reg["open_time"] = pd.to_datetime(reg["open_time"], utc=True)

    # Per-cycle basket net beta + realized basket PnL
    def cyc(g):
        g = g.dropna(subset=["beta","pred"])
        if len(g) < 8: return pd.Series({"net_beta":np.nan,"pnl":np.nan})
        gg = g.sort_values("pred")
        longs = gg.tail(3); shorts = gg.head(3)
        net_beta = longs["beta"].mean() - shorts["beta"].mean()
        # realized basket alpha (long alpha - short alpha), bps
        pnl = (longs["alpha_A"].mean() - shorts["alpha_A"].mean())*10000
        return pd.Series({"net_beta":net_beta, "pnl":pnl})
    cs = m.groupby("open_time").apply(cyc).reset_index().dropna()
    cs = cs.merge(reg, on="open_time", how="left").dropna(subset=["btc_ret_30d"])
    cs["regime"] = np.where(cs["btc_ret_30d"]>0.10,"bull",np.where(cs["btc_ret_30d"]<-0.10,"bear","side"))

    print(f"\n=== Basket net-beta by regime ===")
    print(f"  {'regime':<6} {'cycles':>7} {'mean_net_beta':>14} {'%net_short':>11} {'mean_pnl_bps':>13}")
    for r in ["bull","side","bear"]:
        sub = cs[cs["regime"]==r]
        if len(sub)==0: continue
        print(f"  {r:<6} {len(sub):>7} {sub['net_beta'].mean():>+14.3f} "
              f"{(sub['net_beta']<0).mean()*100:>10.1f}% {sub['pnl'].mean():>+13.2f}")
    print(f"\n  ALL    {len(cs):>7} {cs['net_beta'].mean():>+14.3f} "
          f"{(cs['net_beta']<0).mean()*100:>10.1f}% {cs['pnl'].mean():>+13.2f}")

    # Does net-beta drag PnL? corr, and PnL when net-beta extreme
    print(f"\n=== Does net-beta drag PnL? ===")
    print(f"  corr(net_beta, basket_pnl) overall: {cs['net_beta'].corr(cs['pnl']):+.4f}")
    for r in ["bull","side","bear"]:
        sub = cs[cs["regime"]==r]
        if len(sub)<30: continue
        print(f"  corr in {r}: {sub['net_beta'].corr(sub['pnl']):+.4f}")
    # PnL by net-beta sign in bull
    bull = cs[cs["regime"]=="bull"]
    if len(bull)>30:
        print(f"\n  BULL: mean PnL when net-beta<0: {bull[bull['net_beta']<0]['pnl'].mean():+.2f} bps")
        print(f"  BULL: mean PnL when net-beta>0: {bull[bull['net_beta']>0]['pnl'].mean():+.2f} bps")

    cs.to_parquet(OUT/"X81_net_beta_cycles.parquet", index=False)
    print(f"\nSaved. (Confirms direction/magnitude of net-beta before X79 neutralization.)")


if __name__ == "__main__":
    main()
