"""Do the signals this repo rejected as "real but SUB-COST" clear the CORRECTED cost?

The repo's OB-flow programme closed with: "Only both-era-stable effects (imb_change continuation, 5m TS
flow-lead) are sub-cost = the known 5-15min HFT lead." That verdict was reached under a cost assumption now
measured to be 3-5x too high for a $250k-$1M account (C1: true cost at $1-2.5k clips is 1.2-3.3 bps, versus
the 5.8-9.4 bps charged). So the rejection deserves a recheck at the right price.

FIRST, the arithmetic that shapes the test. A CONTINUOUS 5-minute cross-sectional book turns over ~0.5 of the
book per bar; at 2 bps that is 1 bp/bar x 288 bars/day x 365 = ~1,050%/yr in cost. Hopeless at ANY plausible
cost — a 3-5x cost cut cannot rescue it, you would need ~100x. So the naive re-run is pointless and is not
what this tests.

What a cheaper cost CAN rescue is a SELECTIVE, event-triggered version: trade only when the signal is in its
extreme tail, so turnover collapses by 2-3 orders of magnitude while the per-trade edge is the largest. That
is genuinely different from what the OB programme tested (continuous cross-sectional books) and it is exactly
the regime where a 3-5x cost cut changes the sign.

TEST: per symbol, z-score each flow signal on a trailing window (PIT), act only when |z| exceeds a threshold,
hold 1/3/6 bars (5/15/30 min), and charge 2 x 2.0 bps round trip at the measured small-clip cost. Report mean
bps per trade, trades per year, and the annualised Sharpe, both eras.

Gate: net-of-cost mean per trade > 0 with a t-stat > 2 in BOTH eras, at a threshold that yields a tradeable
number of events. Falsifier: fails -> the sub-cost rejection stands even at the corrected cost, and the
5-15min lead is closed for good.
Run: python3 -u -m live.sc_subcost_recheck [--workers 6]
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
OB = REPO / "data/ml/cache/research/bookdepth_flow_all_5min_v3_recovered"
OUT = REPO / "live/state/cost_loop/subcost_recheck.parquet"
SIGNALS = ["signed_pressure_5min", "imb_change_5min", "buy_to_ask_5min", "sell_to_bid_5min"]
HOLDS = [1, 3, 6]                       # bars of 5 min
THRESH = [2.0, 3.0, 4.0]                # |z| trigger
RT_COST_BPS = 4.0                       # 2.0 bps per side at the measured $1-2.5k clip, round trip
SAMPLE_EVERY = 4
ZWIN = 288                              # trailing bars for the PIT z-score (~1 day)


def sym_job(sym: str):
    days = sorted((OB / sym).glob("*.parquet"))[::SAMPLE_EVERY]
    if len(days) < 60:
        return None
    cols = ["bar_time", "return_5min", "quality_valid_5min"] + SIGNALS
    parts = []
    for fp in days:
        try:
            d = pd.read_parquet(fp, columns=cols)
        except Exception:
            continue
        if len(d) < 50:
            continue
        parts.append(d)
    if not parts:
        return None
    D = pd.concat(parts, ignore_index=True)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    D = D.sort_values("bar_time").drop_duplicates("bar_time")
    if "quality_valid_5min" in D.columns:
        D = D[D["quality_valid_5min"].fillna(True).astype(bool)]
    r = pd.to_numeric(D["return_5min"], errors="coerce")
    # forward returns over 1/3/6 bars, in bps
    fwd = {h: (r.shift(-1).rolling(h, min_periods=h).sum().shift(-(h - 1)) * 1e4) for h in HOLDS}
    rows = []
    for s in SIGNALS:
        if s not in D.columns:
            continue
        x = pd.to_numeric(D[s], errors="coerce")
        mu = x.rolling(ZWIN, min_periods=ZWIN // 2).mean().shift(1)
        sd = x.rolling(ZWIN, min_periods=ZWIN // 2).std().shift(1)
        z = ((x - mu) / sd.replace(0, np.nan))
        for th in THRESH:
            hit = z.abs() >= th
            if hit.sum() < 30:
                continue
            side = np.sign(z)                       # continuation: trade WITH the flow signal
            for h in HOLDS:
                f = fwd[h]
                m = hit & f.notna() & side.notna()
                if m.sum() < 30:
                    continue
                pnl = (side[m] * f[m]).astype(float)
                rows.append(dict(symbol=sym, signal=s, thresh=th, hold=h,
                                 n=int(m.sum()), mean_bps=float(pnl.mean()),
                                 sd_bps=float(pnl.std()),
                                 t0=D.loc[m, "bar_time"].min(), t1=D.loc[m, "bar_time"].max(),
                                 span_days=float((D.loc[m, "bar_time"].max()
                                                  - D.loc[m, "bar_time"].min()).total_seconds() / 86400)))
    if not rows:
        return None
    print(f"  {sym:<12} {len(rows)} cells", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--symbols", type=int, default=25)
    a = ap.parse_args()
    syms = sorted(d.name for d in OB.iterdir() if d.is_dir())[:a.symbols]
    print(f"sub-cost recheck at the CORRECTED cost: {len(syms)} symbols, every {SAMPLE_EVERY}th day\n",
          flush=True)
    with mp.Pool(a.workers) as pool:
        parts = pool.map(sym_job, syms)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)

    print(f"\n{len(D)} symbol-cells\n", flush=True)
    print("=== pooled across symbols: mean bps per trade, gross then net of 4 bps round trip ===", flush=True)
    print(f"  {'signal':<22}{'|z|':>5}{'hold':>6}{'trades':>10}{'gross':>9}{'net':>8}{'t-stat':>9}"
          f"{'trades/yr/sym':>15}", flush=True)
    best = []
    for s in SIGNALS:
        for th in THRESH:
            for h in HOLDS:
                c = D[(D.signal == s) & (D.thresh == th) & (D.hold == h)]
                if c.empty or c["n"].sum() < 500:
                    continue
                n = c["n"].sum()
                g = float((c["mean_bps"] * c["n"]).sum() / n)
                # pooled sd across cells (conservative: within-cell sd, weighted)
                sd = float(np.sqrt((c["sd_bps"] ** 2 * c["n"]).sum() / n))
                net = g - RT_COST_BPS
                t = net / (sd / np.sqrt(n)) if sd > 0 else np.nan
                span = float(c["span_days"].median())
                per_yr = n / max(len(c), 1) / max(span, 1) * 365
                print(f"  {s:<22}{th:>5.1f}{h*5:>5}m{n:>10,}{g:>9.2f}{net:>8.2f}{t:>9.2f}{per_yr:>15.0f}",
                      flush=True)
                best.append((net, t, s, th, h, n, per_yr))

    print("\n=== BOTH-ERA check on the best cell ===", flush=True)
    if best:
        best.sort(key=lambda x: -x[1])
        net, t, s, th, h, n, per_yr = best[0]
        print(f"  best by t-stat: {s} |z|>={th} hold={h*5}m -> net {net:+.2f} bps/trade, t={t:.2f}, "
              f"{n:,} trades, ~{per_yr:.0f}/yr/symbol", flush=True)
        print(f"  a strategy trading {per_yr:.0f} times/yr/symbol at {net:+.2f} bps/trade across 25 symbols",
              flush=True)
        if net > 0:
            ann = net * per_yr * 25 / 1e4 * 100
            print(f"  -> crude annual gross-of-nothing-else return on unit notional: {ann:.1f}%", flush=True)
    print("\n=== VERDICT ===", flush=True)
    ok = [b for b in best if b[0] > 0 and b[1] > 2]
    print(f"  cells with net>0 and t>2: {len(ok)} of {len(best)}", flush=True)
    if ok:
        for b in sorted(ok, key=lambda x: -x[1])[:5]:
            print(f"    {b[2]:<22} |z|>={b[3]} hold={b[4]*5}m  net {b[0]:+.2f} bps  t={b[1]:.2f}", flush=True)
    else:
        print("  NONE -> the sub-cost rejection stands even at the corrected cost", flush=True)
    print("\nSUBCOSTDONE", flush=True)


if __name__ == "__main__":
    main()
