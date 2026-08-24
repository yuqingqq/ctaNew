"""Validate (or kill) the one survivor from the sub-cost recheck.

The recheck found sell_to_bid_5min at |z|>=3-4 with a 30-minute hold nets +2.1 to +4.3 bps/trade after a
4 bps round trip, t=2.0-2.9. Two caveats make that far weaker than it looks and BOTH must be resolved:

  1. 2 of 36 cells at t>2 is CHANCE LEVEL (36 x 5% = 1.8 expected). The monotone structure — net improving
     with both hold and threshold, on both mirror signals — is the only reason not to dismiss it outright.
  2. The t-stats treat 16,643 trades as independent. They are not: a market-wide flow event fires many
     symbols in the same 5-minute bar, so trades cluster hard in time. The effective N is far smaller and the
     true t-stat much lower.

This test therefore:
  A  recomputes the t-stat with DAY-CLUSTERED standard errors (the honest unit of independence)
  B  splits by era — OOS 2023-06..2025-09 vs RECENT 2025-10..2026-06
  C  checks concentration: how much of the P&L comes from the top few days and the top few symbols
  D  reports the realistic capacity, since simultaneous triggers across symbols compete for the same capital

Gate: day-clustered t > 2 in BOTH eras, with no single day contributing >20% of P&L.
Falsifier: fails -> chance-level finding, the sub-cost rejection stands, and the 5-15min lead closes.
Run: python3 -u -m live.sc_subcost_validate
"""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
OB = REPO / "data/ml/cache/research/bookdepth_flow_all_5min_v3_recovered"
SIG = "sell_to_bid_5min"
MIRROR = "buy_to_ask_5min"
HOLD = 6                 # 30 minutes
THRESH = [3.0, 4.0]
RT_COST_BPS = 4.0
SAMPLE_EVERY = 4
ZWIN = 288


def sym_job(sym: str):
    days = sorted((OB / sym).glob("*.parquet"))[::SAMPLE_EVERY]
    if len(days) < 60:
        return None
    cols = ["bar_time", "return_5min", "quality_valid_5min", SIG, MIRROR]
    parts = []
    for fp in days:
        try:
            parts.append(pd.read_parquet(fp, columns=cols))
        except Exception:
            continue
    if not parts:
        return None
    D = pd.concat(parts, ignore_index=True)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    D = D.sort_values("bar_time").drop_duplicates("bar_time")
    if "quality_valid_5min" in D.columns:
        D = D[D["quality_valid_5min"].fillna(True).astype(bool)]
    r = pd.to_numeric(D["return_5min"], errors="coerce")
    fwd = r.shift(-1).rolling(HOLD, min_periods=HOLD).sum().shift(-(HOLD - 1)) * 1e4
    out = []
    for s in (SIG, MIRROR):
        x = pd.to_numeric(D[s], errors="coerce")
        mu = x.rolling(ZWIN, min_periods=ZWIN // 2).mean().shift(1)
        sd = x.rolling(ZWIN, min_periods=ZWIN // 2).std().shift(1)
        z = (x - mu) / sd.replace(0, np.nan)
        for th in THRESH:
            m = (z.abs() >= th) & fwd.notna() & z.notna()
            if m.sum() < 20:
                continue
            out.append(pd.DataFrame({
                "symbol": sym, "signal": s, "thresh": th,
                "t": D.loc[m, "bar_time"].values,
                "pnl": (np.sign(z[m]) * fwd[m]).astype(float).values - RT_COST_BPS}))
    if not out:
        return None
    print(f"  {sym}", flush=True)
    return pd.concat(out, ignore_index=True)


def clustered_t(df):
    """Day-clustered t-stat on the mean net P&L per trade."""
    g = df.groupby(df["t"].dt.floor("1D"))["pnl"]
    s = g.sum(); n = g.size()
    N = int(n.sum()); mean = float(df["pnl"].mean())
    if N < 30 or len(s) < 10:
        return mean, np.nan, N, len(s)
    resid = s - mean * n
    se = float(np.sqrt((resid ** 2).sum())) / N
    return mean, (mean / se if se > 0 else np.nan), N, len(s)


def main():
    syms = sorted(d.name for d in OB.iterdir() if d.is_dir())[:25]
    with mp.Pool(5) as pool:
        parts = pool.map(sym_job, syms)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    D["t"] = pd.to_datetime(D["t"], utc=True)
    print(f"\n{len(D):,} trades across {D.symbol.nunique()} symbols\n", flush=True)

    print("=== A/B — day-clustered t-stat, pooled and by era ===", flush=True)
    print(f"  {'signal':<20}{'|z|':>5}{'era':<10}{'trades':>9}{'days':>7}{'net bps':>10}"
          f"{'naive t':>9}{'CLUSTERED t':>13}", flush=True)
    eras = [("pooled", None, None),
            ("OOS", "2023-06-01", "2025-10-01"),
            ("RECENT", "2025-10-01", "2026-06-01")]
    res = {}
    for s in (SIG, MIRROR):
        for th in THRESH:
            for nm, t0, t1 in eras:
                d = D[(D.signal == s) & (D.thresh == th)]
                if t0:
                    d = d[(d.t >= pd.Timestamp(t0, tz="UTC")) & (d.t < pd.Timestamp(t1, tz="UTC"))]
                if len(d) < 100:
                    continue
                mean, ct, N, nd = clustered_t(d)
                naive = mean / (d["pnl"].std() / np.sqrt(N)) if d["pnl"].std() > 0 else np.nan
                res[(s, th, nm)] = (mean, ct)
                print(f"  {s:<20}{th:>5.1f}{nm:<10}{N:>9,}{nd:>7}{mean:>10.2f}{naive:>9.2f}"
                      f"{ct:>13.2f}", flush=True)

    print("\n=== C — concentration (best cell by clustered t) ===", flush=True)
    cand = [(v[1], k) for k, v in res.items() if k[2] == "pooled" and np.isfinite(v[1])]
    if cand:
        cand.sort(reverse=True)
        _, (s, th, _) = cand[0]
        d = D[(D.signal == s) & (D.thresh == th)]
        byday = d.groupby(d["t"].dt.floor("1D"))["pnl"].sum().sort_values()
        bysym = d.groupby("symbol")["pnl"].sum().sort_values()
        tot = byday.sum()
        print(f"  best cell: {s} |z|>={th}  total {tot:,.0f} bps over {len(byday)} days", flush=True)
        print(f"    top 1 day = {100*byday.iloc[-1]/tot:.0f}% of P&L | "
              f"top 5 days = {100*byday.tail(5).sum()/tot:.0f}% | "
              f"top 5 symbols = {100*bysym.tail(5).sum()/tot:.0f}%", flush=True)
        print(f"    worst day {byday.iloc[0]:,.0f} bps, best day {byday.iloc[-1]:,.0f} bps", flush=True)
        conc_ok = (byday.tail(1).sum() / tot) < 0.20
        print(f"\n=== D — capacity: simultaneous triggers ===", flush=True)
        per_bar = d.groupby(d["t"])["symbol"].nunique()
        print(f"    triggers per 5-min bar: mean {per_bar.mean():.2f}, p95 {per_bar.quantile(.95):.0f}, "
              f"max {per_bar.max()}", flush=True)
        print(f"    bars with >=2 simultaneous: {100*(per_bar >= 2).mean():.1f}%", flush=True)
        print("\n=== GATE ===", flush=True)
        o = res.get((s, th, "OOS"), (np.nan, np.nan))[1]
        rr = res.get((s, th, "RECENT"), (np.nan, np.nan))[1]
        g = (np.isfinite(o) and o > 2) and (np.isfinite(rr) and rr > 2)
        print(f"  clustered t>2 in BOTH eras: OOS {o:.2f}, RECENT {rr:.2f} -> {'PASS' if g else 'FAIL'}",
              flush=True)
        print(f"  no single day >20% of P&L: {'PASS' if conc_ok else 'FAIL'}", flush=True)
    print("\nSUBVALDONE", flush=True)


if __name__ == "__main__":
    main()
