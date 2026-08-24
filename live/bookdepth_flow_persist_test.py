"""Test the flow-PERSISTENCE (TWAP/VWAP / informed-order) hypothesis at 4h, cross-sectional, done with the rigor.
Distinguish LEVEL from PERSISTENCE:
  per 5-min bin i in [T-4h, T):  f_i = buy_5min - sell_5min (net flow),  g_i = buy_5min + sell_5min (gross)
  ofi_4h   = sum(f_i)/sum(g_i)          NET FLOW LEVEL      (already tested -> redundant with price)
  persist  = sum(f_i)/sum(|f_i|)  in [-1,1]  SIGNED STEADINESS  (do the bins add up = one order working, or cancel)
persist is the NEW thing: high |persist| = steady one-sided flow across the window (algo/informed footprint), NOT just
a big net number. Hypothesis: persistent flow CONTINUES (IC>0), unlike the reversal-dominated cross-section.

Tests (fixed-64, both eras, honest day-CI):
  1. raw IC(persist -> fwd 4h)                                     does steady flow predict continuation?
  2. PARTIAL IC(persist | ofi_4h, trailing_ret)                   does steadiness add beyond net-flow LEVEL + momentum?
  3. coiled-spring: IC(persist -> fwd) split by ABSORBED vs SPENT (did the flow move price in its own direction?)
"""
import glob
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from live.bookdepth_timing_corrected import fixed_universe
FLOW = "/home/yuqing/ctaNew/data/ml/cache/research/bookdepth_flow_full_5min"
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")
rng = np.random.default_rng(303)

def agg(sym):
    files = sorted(glob.glob(f"{FLOW}/{sym}/*.parquet"))
    if not files: return None
    d = pd.concat([pd.read_parquet(f, columns=["snapshot_time", "buy_quote_5min", "sell_quote_5min", "gap_interval"])
                   for f in files], ignore_index=True)
    d = d[~d["gap_interval"].fillna(False)]
    d["snapshot_time"] = pd.to_datetime(d["snapshot_time"], utc=True)
    d["t4"] = d["snapshot_time"].dt.floor("4h")
    d["f"] = d["buy_quote_5min"] - d["sell_quote_5min"]
    d["g"] = d["buy_quote_5min"] + d["sell_quote_5min"]
    d["absf"] = d["f"].abs()
    g = d.groupby("t4")
    o = pd.DataFrame({"sf": g["f"].sum(), "sg": g["g"].sum(), "saf": g["absf"].sum(), "n": g.size()})
    o = o[o["n"] >= 24]
    o["ofi_4h"] = o["sf"] / o["sg"].replace(0, np.nan)
    o["persist"] = o["sf"] / o["saf"].replace(0, np.nan)
    o.index = o.index + pd.Timedelta("4h")
    o["symbol"] = sym
    return o.reset_index(names="open_time")[["symbol", "open_time", "ofi_4h", "persist"]]

def ci(ic):
    if len(ic) < 5: return (np.nan, np.nan, np.nan)
    s = pd.DataFrame({"v": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    gg = [x["v"].values for _, x in s.groupby("d")]
    b = [np.concatenate([gg[i] for i in rng.integers(0, len(gg), len(gg))]).mean() for _ in range(2000)]
    return (ic.mean(), *np.nanpercentile(b, [2.5, 97.5]))

def xic(df, feat, tgt, ctrls=None):
    def pb(g):
        cc = ctrls or []
        gg = g[[feat, tgt] + cc].dropna()
        if len(gg) < 10 + len(cc): return np.nan
        y = gg[feat].values
        if cc:
            X = np.column_stack([np.ones(len(gg))] + [gg[c].values for c in cc])
            y = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        return spearmanr(y, gg[tgt].values).correlation
    return df.groupby("open_time").apply(pb).dropna()

def rf(df, feat, tgt, ctrls=None):
    o = {}
    for era, m in [("OOS", df.open_time < CUT), ("REC", df.open_time >= CUT)]:
        o[era] = ci(xic(df[m], feat, tgt, ctrls))
    (ra, rl, ru), (oa, ol, ou) = o["OOS"], o["REC"]
    both = "BOTH✓" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
    return f"{ra:+.4f}[{rl:+.4f},{ru:+.4f}] | {oa:+.4f}[{ol:+.4f},{ou:+.4f}] | {both}"

def main():
    syms = fixed_universe()
    P = pd.concat([x for x in (agg(s) for s in syms) if x is not None], ignore_index=True)
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct"]).sort_values(["symbol", "open_time"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan["trail_ret"] = pan.groupby("symbol")["return_pct"].shift(1)
    D = P.merge(pan, on=["symbol", "open_time"], how="inner").dropna(subset=["return_pct", "persist"])
    D = D[(D.open_time.dt.hour % 4 == 0) & (D.open_time.dt.minute == 0)]
    print(f"panel {len(D)} rows | {D.symbol.nunique()} syms | corr(persist,ofi)={D['persist'].corr(D['ofi_4h']):.2f}\n")
    print("target = forward 4h return.  IC>0 = CONTINUATION.   OOS [CI] | RECENT [CI] | both?\n")
    print(f"  persist   raw                         : {rf(D, 'persist', 'return_pct')}")
    print(f"  ofi_4h    raw (level, reference)       : {rf(D, 'ofi_4h', 'return_pct')}")
    print(f"  trail_ret raw (momentum reference)     : {rf(D, 'trail_ret', 'return_pct')}")
    print(f"  persist | ofi_4h + trail_ret (PARTIAL) : {rf(D, 'persist', 'return_pct', ['ofi_4h', 'trail_ret'])}   <- adds beyond level+momentum?")
    print()
    # coiled-spring: absorbed (flow did NOT move price its way) vs spent
    D["moved"] = D["trail_ret"] * np.sign(D["persist"])       # >0 = flow moved price its own way (spent)
    lo = D[D["moved"] <= D["moved"].median()]; hi = D[D["moved"] > D["moved"].median()]
    print("coiled-spring: IC(persist -> fwd) when flow was ABSORBED (price didn't move its way) vs SPENT:")
    print(f"  ABSORBED (low move): {rf(lo, 'persist', 'return_pct')}")
    print(f"  SPENT    (high move): {rf(hi, 'persist', 'return_pct')}")
    print("\nread: persist ADDS if its PARTIAL IC is +continuation, same-sign CI-off-zero BOTH eras; coiled-spring")
    print("confirmed if continuation is stronger in the ABSORBED bucket. PERSISTDONE")

if __name__ == "__main__":
    main()
