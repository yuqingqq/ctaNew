"""Small-capital research — step 0: what does trading ACTUALLY cost at the clip sizes a $250k-$1M book uses?

Every net number in this repo charged `cost_10k` — modelled slippage for $10,000 clips. A $250k-$1M account
at a 10% vol target holds positions of roughly $1.8k-$18k, so at the lower end we have been charging
ourselves ~5x the right cost. This matters far beyond a rescale: the repo repeatedly found edges and then
dismissed them as CAPACITY-WALLED (the both-era Amihud illiquidity premium "dies >=$500k, collapses >=$2M,
lives in thin names"). Those walls are not binding at this size, so the dismissals may not hold.

MEASUREMENT (empirical, not a depth model). Binance aggTrades aggregates same-price fills of one taker order,
so a taker order that walks the book appears as consecutive rows with the SAME aggressor side and near-
identical timestamps. Group those into a "sweep":

    sweep notional = sum(p*q)
    slippage_bps   = side * (VWAP_sweep / first_price - 1) * 1e4

That is precisely what a market order of that size pays relative to the touch it hit. Bucketing sweeps by
notional gives the realised cost-vs-clip-size curve, per symbol and pooled — the thing the entire
capacity/universe argument in this repo rests on and which has never been measured directly.

Also reports the curve for LIQUID (top-40 ADV) vs THIN names separately, because the whole question is
whether thin names are tradeable at small size.
Run: python3 -u -m live.sc_cost_curve [--workers 6]
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/yuqing/ctaNew")
AGG = REPO / "data/ml/test/parquet/aggTrades"
OUT = REPO / "live/state/cost_loop/cost_curve.parquet"
GAP_MS = 50                       # trades within 50ms on the same side = one taker sweep
BUCKETS = [0, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, np.inf]
LABELS = ["<0.5k", "0.5-1k", "1-2.5k", "2.5-5k", "5-10k", "10-25k", "25-50k", "50-100k", ">100k"]
SAMPLE_EVERY = 11
MIN_FILES = 1100


def sweeps_for_day(fp: Path) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(fp, columns=["transact_time", "price", "quantity", "is_buyer_maker"])
    except Exception:
        return None
    if len(d) < 500:
        return None
    t = pd.to_datetime(d["transact_time"], utc=True).to_numpy("datetime64[ms]").astype("int64")
    p = d["price"].to_numpy(float); q = d["quantity"].to_numpy(float)
    side = np.where(d["is_buyer_maker"].to_numpy(), -1.0, 1.0)
    # new sweep when the side changes or the time gap exceeds GAP_MS
    newgrp = np.empty(len(t), bool); newgrp[0] = True
    newgrp[1:] = (side[1:] != side[:-1]) | ((t[1:] - t[:-1]) > GAP_MS)
    gid = np.cumsum(newgrp) - 1
    notional = p * q
    # vectorised with reduceat over sweep boundaries — a groupby.apply here is ~100x slower and the
    # per-symbol concat of raw sweeps (tens of millions of rows) exhausts memory
    starts = np.flatnonzero(newgrp)
    tot = np.add.reduceat(notional, starts)
    vwap = np.add.reduceat(p * notional, starts) / np.where(tot > 0, tot, np.nan)
    first = p[starts]
    sd = side[starts]
    cnt = np.diff(np.append(starts, len(p)))
    slip = sd * (vwap / first - 1.0) * 1e4
    ok = np.isfinite(slip) & (tot > 0)
    S = pd.DataFrame({"notional": tot[ok], "slip_bps": slip[ok], "legs": cnt[ok]})
    S["bucket"] = pd.cut(S["notional"], BUCKETS, labels=LABELS, right=False)
    return S.groupby("bucket", observed=True).agg(          # aggregate WITHIN the day
        n_sweeps=("slip_bps", "size"), sum_slip=("slip_bps", "sum"),
        med_slip=("slip_bps", "median"), notional=("notional", "sum"),
        sum_legs=("legs", "sum")).reset_index()


def sym_job(sym: str) -> pd.DataFrame | None:
    files = sorted((AGG / sym).glob("*.parquet"))[::SAMPLE_EVERY]
    parts = []
    for fp in files:
        s = sweeps_for_day(fp)
        if s is not None and len(s):
            parts.append(s)
    if not parts:
        return None
    S = pd.concat(parts, ignore_index=True)
    agg = S.groupby("bucket", observed=True).agg(
        n_sweeps=("n_sweeps", "sum"), sum_slip=("sum_slip", "sum"),
        med_slip=("med_slip", "median"), notional=("notional", "sum"),
        sum_legs=("sum_legs", "sum")).reset_index()
    agg["mean_slip"] = agg["sum_slip"] / agg["n_sweeps"]
    agg["mean_legs"] = agg["sum_legs"] / agg["n_sweeps"]
    agg["symbol"] = sym
    print(f"  {sym:<12} {int(agg['n_sweeps'].sum()):>10,} sweeps", flush=True)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--symbols", type=int, default=0)
    a = ap.parse_args()
    syms = sorted(d.name for d in AGG.iterdir()
                  if d.is_dir() and len(list(d.glob("*.parquet"))) >= MIN_FILES)
    if a.symbols:
        syms = syms[:a.symbols]
    print(f"realised cost curve: {len(syms)} symbols, every {SAMPLE_EVERY}th day, "
          f"sweeps grouped at {GAP_MS}ms\n", flush=True)
    with mp.Pool(a.workers) as pool:
        parts = pool.map(sym_job, syms)
    D = pd.concat([p for p in parts if p is not None], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    D.to_parquet(OUT, index=False)

    print("\n=== POOLED realised taker slippage vs clip size (notional-weighted) ===", flush=True)
    print(f"  {'clip':<10}{'sweeps':>12}{'mean bps':>11}{'median bps':>12}{'avg levels':>12}", flush=True)
    for lab in LABELS:
        s = D[D["bucket"] == lab]
        if s.empty:
            continue
        w = s["n_sweeps"]
        print(f"  {lab:<10}{int(w.sum()):>12,}{float((s['mean_slip']*w).sum()/w.sum()):>11.3f}"
              f"{float((s['med_slip']*w).sum()/w.sum()):>12.3f}"
              f"{float((s['mean_legs']*w).sum()/w.sum()):>12.2f}", flush=True)

    print("\n=== the number that matters: cost at a $250k-$1M book's clip vs what we charged ===", flush=True)
    def wm(lab):
        s = D[D["bucket"] == lab]
        return float((s["mean_slip"] * s["n_sweeps"]).sum() / s["n_sweeps"].sum()) if len(s) else np.nan
    small = np.nanmean([wm("1-2.5k"), wm("2.5-5k")])
    mid = np.nanmean([wm("5-10k"), wm("10-25k")])
    print(f"  realised slippage at $1-5k clips   : {small:.3f} bps", flush=True)
    print(f"  realised slippage at $5-25k clips  : {mid:.3f} bps", flush=True)
    cal = pd.read_csv(REPO / "live/state/v3loop/persym_cost_cal.csv")
    print(f"  cost_10k model used all session    : {cal['cost_10k'].mean():.3f} bps (mean over symbols)",
          flush=True)
    HALF_SPREAD = 1.83     # measured on 31 syms x 754 days in live/bx_iter1_markout.py
    print(f"\n  CORRECTION: the sweep VWAP is measured FROM THE FIRST FILL, i.e. from the touch, so the",
          flush=True)
    print(f"  half-spread paid to cross is NOT in the numbers above. Total taker cost = walk + half-spread:",
          flush=True)
    print(f"    total at $1-5k clips  : {small + HALF_SPREAD:.2f} bps  ({small:.2f} walk + {HALF_SPREAD} spread)",
          flush=True)
    print(f"    total at $5-25k clips : {mid + HALF_SPREAD:.2f} bps  ({mid:.2f} walk + {HALF_SPREAD} spread)",
          flush=True)
    print(f"    cost_10k model        : {cal['cost_10k'].mean():.2f} bps (pure slippage; the bot adds "
          f"FEE_BPS_FILL separately)", flush=True)
    print(f"  -> overcharge at a $1-5k clip: {cal['cost_10k'].mean()/max(small+HALF_SPREAD,1e-9):.1f}x "
          f"(NOT the {cal['cost_10k'].mean()/max(small,1e-9):.0f}x the walk alone would suggest)", flush=True)
    print(f"  CAVEAT: we observe only sweeps that WERE sent. Traders size to available liquidity, so any",
          flush=True)
    print(f"  notional bucket is populated by orders placed when conditions suited them — a downward bias",
          flush=True)
    print(f"  on measured slippage. Treat this as a lower bound.", flush=True)

    print("\n=== LIQUID vs THIN: are thin names tradeable at small size? ===", flush=True)
    tot = D.groupby("symbol")["notional"].sum().sort_values(ascending=False)
    k = max(3, len(tot) // 4)
    liquid = set(tot.head(k).index); thin = set(tot.tail(k).index)
    for nm, grp in ((f"top-{k} by volume", liquid), (f"bottom-{k} by volume", thin)):
        print(f"  --- {nm} ---", flush=True)
        sub = D[D["symbol"].isin(grp)]
        for lab in ("1-2.5k", "2.5-5k", "5-10k", "10-25k", "50-100k"):
            s = sub[sub["bucket"] == lab]
            if s.empty:
                continue
            w = s["n_sweeps"]
            print(f"    {lab:<10}{float((s['mean_slip']*w).sum()/w.sum()):>8.3f} bps  "
                  f"({int(w.sum()):,} sweeps)", flush=True)
    print("\nCOSTCURVEDONE", flush=True)


if __name__ == "__main__":
    main()
