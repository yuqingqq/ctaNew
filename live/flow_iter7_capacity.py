"""iter7 — earn the NO for the RIGHT reason (adversarial reviewer's owed test).

The reviewer found a real crack: the UNTESTED daily illiquidity family (impact_bps_per_pressure
= Amihud) has both-era partial-IC beyond price+daily-vol+size at the MULTI-DAY horizon I never
tested. Two questions decide whether it is USABLE alpha or a capacity-walled textbook premium:
  (1) VOL REPACKAGING: does it survive an INTRADAY-realized-vol control (C1)? (illiq numerator is
      |return|, so its power may be intraday vol = a price feature).
  (2) CAPACITY: does the both-era signal survive restricting to TRADEABLE-DEPTH names?
Signals: illiq=mean log1p|impact|, amihud=mean|ret5|/mean vol, apress=mean|signed_pressure| (pure flow).
partial-IC vs fwd_3d/5d, both-era, day-clustered, on ALL vs depth-filtered subsamples.
"""
import os, glob, numpy as np, pandas as pd
os.environ.setdefault("MALLOC_ARENA_MAX", "2"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys; sys.path.insert(0, "/home/yuqing/ctaNew")
from live.flow_harness import SRC, ci, partial_xsic, fmt

CUT = pd.Timestamp("2025-10-01", tz="UTC")


def daily_sym(sym):
    sf = sorted(glob.glob(f"{SRC}/{sym}/*.parquet"))
    if not sf:
        return None
    d = pd.read_parquet(sf, columns=["bar_time", "price", "quality_valid_5min", "return_5min",
                                     "impact_bps_per_pressure_5min", "signed_pressure_5min",
                                     "buy_quote_5min", "sell_quote_5min", "bid1", "ask1"])
    d = d[d["quality_valid_5min"].fillna(False)]
    if d.empty:
        return None
    d["bar_time"] = pd.to_datetime(d["bar_time"], utc=True)
    d["date"] = d["bar_time"].dt.floor("1D")
    d["aimp"] = np.log1p(d["impact_bps_per_pressure_5min"].abs())
    d["apress"] = d["signed_pressure_5min"].abs()
    d["ar5"] = d["return_5min"].abs()
    d["dvol"] = d["buy_quote_5min"].fillna(0) + d["sell_quote_5min"].fillna(0)
    d["depth"] = (d["bid1"] + d["ask1"])  # total displayed +/-1% notional ($)
    g = d.groupby("date")
    daily = pd.DataFrame({
        "illiq": g["aimp"].mean(),
        "amihud": g["ar5"].mean() / g["dvol"].mean().replace(0, np.nan),
        "apress": g["apress"].mean(),
        "rv_intra": g["return_5min"].std(),
        "logvol": np.log1p(g["dvol"].sum()),
        "depth_1s": g["depth"].mean() / 2.0,   # one-side $ depth
        "close": g["price"].last(), "nb": g.size(),
    })
    daily = daily[daily["nb"] >= 100]
    if len(daily) < 30:
        return None
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="1D")
    c = daily["close"].reindex(full)
    ret = c.pct_change(fill_method=None)
    add = pd.DataFrame({
        "fwd_3d": c.shift(-3) / c - 1, "fwd_5d": c.shift(-5) / c - 1,
        "tr_1d": c / c.shift(1) - 1, "tr_3d": c / c.shift(3) - 1, "tr_5d": c / c.shift(5) - 1,
        "rv_10d": ret.rolling(10, min_periods=6).std(),
    }, index=full)
    out = daily.join(add, how="left").reset_index(names="date")
    out["bar_time"] = out["date"]; out["symbol"] = sym
    out["ldepth"] = np.log1p(out["depth_1s"])
    for cc in out.columns:
        if out[cc].dtype == np.float64:
            out[cc] = out[cc].astype(np.float32)
    return out


def be(D, feat, tgt, controls, mask):
    o = {}
    for era, m in [("OOS", (D["bar_time"] < CUT).to_numpy() & mask),
                   ("REC", (D["bar_time"] >= CUT).to_numpy() & mask)]:
        o[era] = ci(partial_xsic(D, feat, controls, tgt, min_n=12, row_mask=m))
    (oa, ol, ou), (ra, rl, ru) = o["OOS"], o["REC"]
    o["both"] = bool(np.sign(oa) == np.sign(ra) and (ol > 0 or ou < 0) and (rl > 0 or ru < 0))
    return o


def main():
    syms = sorted(p.split("/")[-1] for p in glob.glob(f"{SRC}/*") if os.path.isdir(p))
    D = pd.concat([x for x in (daily_sym(s) for s in syms) if x is not None], ignore_index=True)
    D["bar_time"] = pd.to_datetime(D["bar_time"], utc=True)
    dep = D["depth_1s"].to_numpy()
    print(f"DAILY panel {len(D):,} name-days | {D.symbol.nunique()} syms")
    print(f"one-side depth $: p10 {np.nanpercentile(dep,10):,.0f} | median {np.nanmedian(dep):,.0f} "
          f"| p90 {np.nanpercentile(dep,90):,.0f}\n")

    C0 = ["tr_1d", "tr_3d", "tr_5d", "rv_10d", "logvol"]          # price + daily-vol + size
    C1 = C0 + ["rv_intra"]                                         # + intraday realized vol
    allrows = np.ones(len(D), bool)

    print("=== (1) VOL REPACKAGING: illiq/amihud/apress partial-IC vs fwd, C0 vs C1(+intraday vol), ALL names ===")
    for feat in ["illiq", "amihud", "apress"]:
        print(f"### {feat}")
        for k in ["3d", "5d"]:
            print(f"  fwd_{k} C0        : {fmt(be(D, feat, f'fwd_{k}', C0, allrows))}")
            print(f"        C1(+rvintra): {fmt(be(D, feat, f'fwd_{k}', C1, allrows))}", flush=True)
        print()

    print("=== (2) CAPACITY: illiq partial-IC vs fwd_5d (control C1) on depth-filtered subsamples ===")
    for thr in [0, 100_000, 500_000, 2_000_000]:
        mask = dep >= thr
        kept = mask.mean()
        o = be(D, "illiq", "fwd_5d", C1, mask)
        print(f"  depth_1s >= ${thr:>10,} ({kept:5.1%} kept): {fmt(o)}  {'<<< BOTH' if o['both'] else ''}",
              flush=True)
    print("\nread: if illiq loses both-era going C0->C1 it was intraday vol (a price feature); if it loses")
    print("both-era as the depth filter tightens, the residual premium is capacity-walled (thin names).")
    print("ITER7DONE", flush=True)


if __name__ == "__main__":
    main()
