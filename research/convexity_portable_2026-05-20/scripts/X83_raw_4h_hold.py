"""X83 — Raw 4h-hold backtest (construction matching the signal's native horizon).

The V3.1 sleeve holds 24h with 6 overlapping sleeves, but the target is 4h-forward
alpha. That horizon mismatch + concentration may be why V5 (IC +0.016, good signal)
gives negative sleeve Sharpe. Test the SIMPLEST construction:
  - Every 4h cycle: rank syms by pred, long top-K / short bottom-K, equal weight.
  - Hold exactly 4h (= target horizon), non-overlapping → cycles independent.
  - Per-cycle gross = mean(long alpha_A) - mean(short alpha_A).
  - Cost: 2*K legs rotated each cycle (full turnover) → 2*cost_per_leg per side.
  - Annualize with sqrt(6*365) (valid: non-overlapping 4h bars are independent).

Run on V0 3yr + V5 3yr preds. Sweep K. Overall + by PIT trend regime.
This isolates SIGNAL HARVEST from sleeve/overlap/concentration artifacts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
COST_PER_LEG = 4.5e-4  # 4.5 bps


def btc_30d():
    files = sorted((KLINES/"BTCUSDT"/"5m").glob("*.parquet"))
    btc = pd.concat([pd.read_parquet(f, columns=["open_time","close"]) for f in files],
                     ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc = btc.set_index("open_time")["close"].astype(np.float64)
    b4 = btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    r = (b4/b4.shift(180)-1)
    return r.to_frame("btc_ret_30d").reset_index()


def raw_hold(apd, K, cost=COST_PER_LEG):
    """Non-overlapping 4h L/S basket; returns per-cycle net return series indexed by open_time."""
    a = apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)].copy()
    rows=[]
    for ot, g in a.groupby("open_time"):
        g = g.dropna(subset=["pred","alpha_A"])
        if len(g) < 2*K: continue
        gg = g.sort_values("pred")
        longs = gg.tail(K)["alpha_A"].mean()
        shorts = gg.head(K)["alpha_A"].mean()
        gross = longs - shorts
        # full rotation each cycle: 2K legs entered + 2K exited; cost per leg both sides
        net = gross - 2*cost*2  # round-trip on long+short legs (approx)
        rows.append((ot, gross, net))
    df = pd.DataFrame(rows, columns=["open_time","gross","net"])
    return df


def ann_sharpe(x):
    x = pd.Series(x).dropna()
    if len(x)<3 or x.std()==0: return np.nan
    return x.mean()/x.std()*np.sqrt(6*365)


def main():
    print("=== X83 raw 4h-hold backtest ===\n", flush=True)
    reg = btc_30d()
    cases = {
        "V0_3yr": RCACHE/"x70_v0_3yr_preds.parquet",
        "V5_3yr": RCACHE/"x78_v5_single_preds.parquet",
    }
    for name, pth in cases.items():
        if not Path(pth).exists():
            print(f"{name}: missing"); continue
        apd = pd.read_parquet(pth); apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
        print(f"\n=== {name} ({apd['symbol'].nunique()} syms) ===")
        print(f"  {'K':<4}{'gross_Sh':>10}{'net_Sh':>9}{'mean_net_bps':>14}{'cycles':>9}")
        for K in [1,2,3,5]:
            bt = raw_hold(apd, K)
            print(f"  {K:<4}{ann_sharpe(bt['gross']):>+10.2f}{ann_sharpe(bt['net']):>+9.2f}"
                  f"{bt['net'].mean()*1e4:>+13.2f} {len(bt):>9,}", flush=True)
        # by regime at K=3
        bt = raw_hold(apd, 3).merge(reg, on="open_time", how="left").dropna(subset=["btc_ret_30d"])
        bt["regime"] = np.where(bt["btc_ret_30d"]>0.10,"bull",np.where(bt["btc_ret_30d"]<-0.10,"bear","side"))
        print(f"  K=3 by regime (net Sharpe):")
        for r in ["bull","side","bear"]:
            s=bt[bt["regime"]==r]
            if len(s)<30: continue
            print(f"    {r:<6} n={len(s):>5} net_Sh={ann_sharpe(s['net']):>+6.2f} mean_net={s['net'].mean()*1e4:>+6.2f}bps", flush=True)

    print(f"\nReference: V0 sleeve 3yr +0.12; V5 sleeve 3yr -0.36 (conc 100%).")
    print(f"If raw-4h-hold Sharpe >> sleeve, the 24h-overlap/concentration is the problem, not the signal.")


if __name__ == "__main__":
    main()
