"""Do the NEW flow-based metrics (from the 5-min flow-dynamics dataset) work when aggregated to the 4h decision grid?
This is UNTESTED: prior 4h OB tests used book STATE (imbalance) or a price-PROXY absorption (absorp_net). Here we use
the actual aggressive TRADE FLOW.

Aggregate the 5-min flow rows over each [T-4h, T) window (PIT; snapshot_time < T), decision at T:
  ofi_4h       = (sum buy - sum sell)/(sum buy + sum sell)   net aggressive order-flow imbalance  [CLEAN, no band confound]
  sp_4h        = mean signed_pressure_5min                    flow normalized by displayed depth
  absorb_net   = mean(bid_depth_residual - ask_depth_residual) absorption proxy  [band-confounded]
  cand_net     = sum(bid_absorption_candidate - ask_absorption_candidate)  failed-liquidity flags
Target = forward 4h return (panel return_pct at T). Control = trailing 4h return (return_pct at T-4h) = the completed
price move = momentum + the band-confound. Cross-sectional rank-IC + PARTIAL-IC (vs trailing return), both eras, day-CI.
Fixed-64 (survivor-caveat). If a flow metric's PARTIAL-IC is same-sign CI-off-zero BOTH eras -> it adds beyond momentum.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from live.bookdepth_timing_corrected import fixed_universe
FLOW = "/home/yuqing/ctaNew/data/ml/cache/research/bookdepth_flow_full_5min"
PANEL = "/home/yuqing/ctaNew/outputs/vBTC_features/panel_expanded_v0_clean.parquet"
CUT = pd.Timestamp("2025-10-01", tz="UTC")
rng = np.random.default_rng(202)
COLS = ["snapshot_time", "buy_quote_5min", "sell_quote_5min", "signed_pressure_5min",
        "ask_depth_residual_5min", "bid_depth_residual_5min",
        "ask_absorption_candidate_5min", "bid_absorption_candidate_5min", "gap_interval"]

def agg_4h(sym):
    files = sorted(glob.glob(f"{FLOW}/{sym}/*.parquet"))
    if not files: return None
    d = pd.concat([pd.read_parquet(f, columns=COLS) for f in files], ignore_index=True)
    d["snapshot_time"] = pd.to_datetime(d["snapshot_time"], utc=True)
    d = d[~d["gap_interval"].fillna(False)]
    d["t4"] = d["snapshot_time"].dt.floor("4h")
    g = d.groupby("t4")
    o = pd.DataFrame({
        "buy": g["buy_quote_5min"].sum(), "sell": g["sell_quote_5min"].sum(),
        "sp_4h": g["signed_pressure_5min"].mean(),
        "ares": g["ask_depth_residual_5min"].mean(), "bres": g["bid_depth_residual_5min"].mean(),
        "acand": g["ask_absorption_candidate_5min"].sum(), "bcand": g["bid_absorption_candidate_5min"].sum(),
        "n": g.size(),
    })
    o = o[o["n"] >= 24]                                   # require >=24 of ~48 5-min bins present in the 4h
    o["ofi_4h"] = (o["buy"] - o["sell"]) / (o["buy"] + o["sell"]).replace(0, np.nan)
    o["absorb_net"] = o["bres"] - o["ares"]
    o["cand_net"] = o["bcand"] - o["acand"]
    o.index = o.index + pd.Timedelta("4h")               # decision bar = window end
    o["symbol"] = sym
    return o.reset_index(names="open_time")[["symbol", "open_time", "ofi_4h", "sp_4h", "absorb_net", "cand_net"]]

def xic(df, feat, tgt):
    return df.groupby("open_time").apply(lambda g: spearmanr(g[feat], g[tgt]).correlation
                                         if g[[feat, tgt]].dropna().shape[0] >= 8 else np.nan).dropna()

def pxic(df, feat, ctrl, tgt):
    def pb(g):
        gg = g[[feat, ctrl, tgt]].dropna()
        if len(gg) < 10: return np.nan
        X = np.column_stack([np.ones(len(gg)), gg[ctrl].values])
        r = gg[feat].values - X @ np.linalg.lstsq(X, gg[feat].values, rcond=None)[0]
        return spearmanr(r, gg[tgt].values).correlation
    return df.groupby("open_time").apply(pb).dropna()

def ci(ic):
    if len(ic) < 5: return (np.nan, np.nan, np.nan)
    s = pd.DataFrame({"v": ic.values}, index=pd.to_datetime(ic.index, utc=True)); s["d"] = s.index.floor("1D")
    g = [x["v"].values for _, x in s.groupby("d")]
    b = [np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(2000)]
    return (ic.mean(), *np.nanpercentile(b, [2.5, 97.5]))

def rowfmt(df, feat, tgt, ctrl=None):
    o = {}
    for era, m in [("OOS", df.open_time < CUT), ("REC", df.open_time >= CUT)]:
        icv = pxic(df[m], feat, ctrl, tgt) if ctrl else xic(df[m], feat, tgt)
        o[era] = ci(icv)
    (ra, rl, ru), (oa, ol, ou) = o["OOS"], o["REC"]
    both = "BOTH✓" if (np.sign(ra) == np.sign(oa) and (rl > 0 or ru < 0) and (ol > 0 or ou < 0)) else "no"
    return f"{ra:+.4f}[{rl:+.4f},{ru:+.4f}] | {oa:+.4f}[{ol:+.4f},{ou:+.4f}] | {both}"

def main():
    syms = fixed_universe()
    P = pd.concat([x for x in (agg_4h(s) for s in syms) if x is not None], ignore_index=True)
    pan = pd.read_parquet(PANEL, columns=["symbol", "open_time", "return_pct"])
    pan["open_time"] = pd.to_datetime(pan["open_time"], utc=True)
    pan = pan.sort_values(["symbol", "open_time"])
    pan["trail_ret"] = pan.groupby("symbol")["return_pct"].shift(1)     # completed prior-4h return (momentum/band ctrl)
    D = P.merge(pan, on=["symbol", "open_time"], how="inner").dropna(subset=["return_pct"])
    D = D[(D.open_time.dt.hour % 4 == 0) & (D.open_time.dt.minute == 0)]
    print(f"panel {len(D)} rows | {D.symbol.nunique()} syms | {D.open_time.min().date()}..{D.open_time.max().date()}\n")
    print("target = forward 4h return.  rank-IC: OOS [CI] | RECENT [CI] | both-era?\n")
    for feat in ["ofi_4h", "sp_4h", "absorb_net", "cand_net"]:
        print(f"### {feat} ###")
        print(f"  raw                  : {rowfmt(D, feat, 'return_pct')}")
        print(f"  | trailing-ret (PART): {rowfmt(D, feat, 'return_pct', ctrl='trail_ret')}   <- adds beyond momentum?")
    print("\n  (reference) trail_ret  :", rowfmt(D, "trail_ret", "return_pct"))
    print("\nread: a flow metric ADDS only if its PARTIAL-IC (vs trailing return) is same-sign CI-off-zero BOTH eras.")
    print("ofi_4h is the clean (band-confound-free) one to watch. FLOW4HDONE")

if __name__ == "__main__":
    main()
