"""X86 — Beta-neutral leg sizing in a HELD-BOOK (24h overlap) backtest on V0.

X84 showed clean position-level beta-neutral helps at the 4h-COHORT level
(+1.72 vs +1.67). This confirms it in a held-book that mirrors the sleeve's cost
amortization (enter every 4h, hold 24h = 6 overlapping baskets, net position =
sum of active sleeves), comparing:
  EQUAL weight   : each leg member ±1/K
  BETA-NEUTRAL   : scale long/short leg sizes so basket net-beta = 0 (ranking unchanged)

This isolates the beta-neutral CONSTRUCTION effect on the V0 base model with
realistic (amortized) turnover. 12mo + 3yr. Net of cost.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import pandas as pd, numpy as np

REPO = Path("/home/yuqing/ctaNew")
OUT = REPO/"research/convexity_portable_2026-05-20/results"; RCACHE = OUT/"_cache"
KLINES = REPO/"data/ml/test/parquet/klines"
COST_PER_LEG = 4.5e-4
K = 3
HOLD = 6  # 6 × 4h = 24h


def load_close(sym):
    sd = KLINES/sym/"5m"
    if not sd.exists(): return None
    dfs=[pd.read_parquet(f,columns=["open_time","close"]) for f in sorted(sd.glob("*.parquet"))]
    df=pd.concat(dfs,ignore_index=True).drop_duplicates("open_time").sort_values("open_time")
    df["open_time"]=pd.to_datetime(df["open_time"],utc=True)
    return df.set_index("open_time")["close"].astype(np.float64)


def ann_sharpe(x):
    x=pd.Series(x).dropna()
    return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan


def heldbook_backtest(cyc_baskets, cyc_times, ret_lookup, beta_neutral):
    """cyc_baskets: list of (long_syms, short_syms, long_betas, short_betas) per 4h cycle.
    Net position per cycle = avg of last HOLD baskets' target weights.
    PnL = sum(net_weight_sym × sym_4h_return) - turnover cost.
    Returns per-cycle net return series."""
    # Build target weights per cycle (dict sym->weight)
    cyc_w = []
    for (L, S, bL, bS) in cyc_baskets:
        w = {}
        if beta_neutral and len(L)>0 and len(S)>0:
            mbL, mbS = np.mean(bL), np.mean(bS)
            if mbL>0 and mbS>0:
                a = 2*mbS/(mbL+mbS); b = 2*mbL/(mbL+mbS)  # a*mbL=b*mbS, a+b=2
            else:
                a=b=1.0
        else:
            a=b=1.0
        for s in L: w[s] = w.get(s,0) + a/max(1,len(L))
        for s in S: w[s] = w.get(s,0) - b/max(1,len(S))
        cyc_w.append(w)

    n = len(cyc_w)
    prev_net = {}
    rets = []
    for t in range(n):
        # net position = avg of baskets entered at t-HOLD+1 .. t
        active = cyc_w[max(0,t-HOLD+1):t+1]
        net = {}
        for w in active:
            for s,wt in w.items(): net[s] = net.get(s,0) + wt/HOLD
        # turnover cost vs prev_net
        allsyms = set(net)|set(prev_net)
        turn = sum(abs(net.get(s,0)-prev_net.get(s,0)) for s in allsyms)
        cost = turn * 0.5 * COST_PER_LEG  # per sleeve doc: |delta| × 0.5 × cost_per_leg
        # realized 4h return at this cycle
        rl = ret_lookup.get(cyc_times[t], {})
        pnl = sum(net.get(s,0)*rl.get(s,0.0) for s in net)
        rets.append(pnl - cost)
        prev_net = net
    return pd.Series(rets, index=cyc_times)


def main():
    t0=time.time()
    print("=== X86 beta-neutral held-book backtest (V0) ===\n", flush=True)
    apd=pd.read_parquet(RCACHE/"x70_v0_3yr_preds.parquet")
    apd["open_time"]=pd.to_datetime(apd["open_time"],utc=True)
    apd=apd[(apd["open_time"].dt.hour%4==0)&(apd["open_time"].dt.minute==0)]
    syms=sorted(apd["symbol"].unique())
    print(f"V0 preds: {apd['open_time'].nunique():,} cycles, {len(syms)} syms")

    # per-sym trailing 30d beta + 4h-fwd return lookup
    print("building betas + return lookup...", flush=True)
    btc=load_close("BTCUSDT"); b4=btc[(btc.index.hour%4==0)&(btc.index.minute==0)]
    br=np.log(b4/b4.shift(1)); bvar=br.rolling(180,min_periods=42).var()
    beta_map={}
    for sym in syms:
        c=load_close(sym)
        if c is None: continue
        c4=c[(c.index.hour%4==0)&(c.index.minute==0)]; r=np.log(c4/c4.shift(1))
        ri,bi=r.align(br,join="inner")
        beta=(ri.rolling(180,min_periods=42).cov(bi)/bvar.reindex(ri.index).replace(0,np.nan)).shift(1)
        beta_map[sym]=beta
    betas=pd.concat([s.rename(k) for k,s in beta_map.items()],axis=1)  # index time, cols syms

    # return lookup: use return_pct (raw 4h fwd) from preds
    ret_lookup = {ot: dict(zip(g["symbol"], g["return_pct"])) for ot,g in apd.groupby("open_time")}

    def build_baskets(sub):
        times=sorted(sub["open_time"].unique()); baskets=[]
        for ot in times:
            g=sub[sub["open_time"]==ot].dropna(subset=["pred"])
            if len(g)<2*K: baskets.append((( ),(),[],[])); continue
            gg=g.sort_values("pred"); L=gg.tail(K)["symbol"].tolist(); S=gg.head(K)["symbol"].tolist()
            brow=betas.loc[ot] if ot in betas.index else None
            bL=[brow[s] for s in L if brow is not None and s in brow and pd.notna(brow[s])]
            bS=[brow[s] for s in S if brow is not None and s in brow and pd.notna(brow[s])]
            if len(bL)<K or len(bS)<K: bL=[1.0]*len(L); bS=[1.0]*len(S)
            baskets.append((L,S,bL,bS))
        return times, baskets

    for tag, sub in [("3yr", apd), ("12mo", apd[apd["open_time"]>=pd.Timestamp("2025-05-01",tz="UTC")])]:
        times, baskets = build_baskets(sub)
        eq = heldbook_backtest(baskets, times, ret_lookup, beta_neutral=False)
        bn = heldbook_backtest(baskets, times, ret_lookup, beta_neutral=True)
        print(f"\n--- {tag} (held-book 24h, net of cost) ---")
        print(f"  EQUAL-weight : Sharpe={ann_sharpe(eq):+.2f}  mean={eq.mean()*1e4:+.2f}bps")
        print(f"  BETA-NEUTRAL : Sharpe={ann_sharpe(bn):+.2f}  mean={bn.mean()*1e4:+.2f}bps", flush=True)

    print(f"\nVERDICT: if BETA-NEUTRAL Sharpe > EQUAL on the held book (both windows),")
    print(f"beta-neutral leg sizing improves the V0 base model in realistic construction.")
    print(f"Done [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
