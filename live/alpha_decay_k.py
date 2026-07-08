"""How long does the convexity alpha persist, and how broad is it across pred-rank -> what K is right.
 (A) DECAY: top-2/bottom-2 forward residual-alpha spread at horizons 1..48 bars (4h..8d). How long the edge lasts.
 (B) BREADTH: at the traded 24h horizon, forward alpha by pred-rank decile, and the top-K minus bottom-K L/S
     spread as K grows 1..12 -> marginal value of the Kth name (tells us whether K>2 captures more or dilutes).
Uses the frozen base pred (short ranker) + residrev pred (long ranker) vs alpha_vs_btc_realized.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"
base=pd.read_parquet(f"{R}/live/state/convexity/hl_lean175/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
lng =pd.read_parquet(f"{R}/live/state/convexity/hl_residrev_lean/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"pred_long"})
for x in (base,lng): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
H=[1,2,3,4,6,8,12,24,48]
for h in H:
    pan[f"fwd{h}"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(h).sum().shift(-(h-1)))
d=base.merge(lng,on=["symbol","open_time"],how="inner").merge(pan,on=["symbol","open_time"],how="inner")
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")]
print(f"rows {len(d)}, cycles {d.open_time.nunique()}\n")

# (A) DECAY: bottom-2 (shorts, base pred) and top-1 (long, long pred) forward alpha by horizon
print("=== (A) DECAY: L/S forward residual alpha (bps) by holding horizon ===")
print(f"  {'horizon':>8s} {'short_edge':>10s} {'long_edge':>10s} {'L-S(bps)':>9s} {'per-bar':>8s}")
for h in H:
    rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<6: continue
        sh=g.nsmallest(2,"pred")[f"fwd{h}"].mean()      # shorts: lowest base pred; we're short -> -edge
        lo=g.nlargest(1,"pred_long")[f"fwd{h}"].mean()    # long: highest long pred
        rows.append((lo,sh))
    lo=np.nanmean([r[0] for r in rows])*1e4; sh=np.nanmean([r[1] for r in rows])*1e4
    ls=lo-sh
    print(f"  {h:6d}b  {-sh:+10.1f} {lo:+10.1f} {ls:+9.1f} {ls/h:+8.1f}")
print("  (short_edge shown as PnL sign: +ve = shorting the low-pred names profits; per-bar = L-S / horizon)")

# (B) BREADTH at 24h (h=6): decile of base pred vs forward alpha, and top-K/bottom-K spread vs K
print("\n=== (B) BREADTH @24h: forward residual alpha by base-pred decile (bps) ===")
d6=d.dropna(subset=["fwd6"]).copy()
d6["dec"]=d6.groupby("open_time")["pred"].transform(lambda s:pd.qcut(s,10,labels=False,duplicates="drop"))
dec=d6.groupby("dec")["fwd6"].mean()*1e4
for k,v in dec.items(): print(f"  decile {int(k)} (pred {'low/SHORT' if k==0 else 'high/LONG' if k==9 else ''}): {v:+6.1f}")
print("\n=== K-tuning: top-K(long book) minus bottom-K(base) L/S spread @24h vs K, and MARGINAL Kth name ===")
print(f"  {'K':>3s} {'L-S(bps)':>9s} {'per-name':>9s} {'marginal Kth':>13s}")
prev=None
for K in [1,2,3,4,5,6,8,10,12]:
    rows=[]
    for ot,g in d6.groupby("open_time"):
        if len(g)<2*K: continue
        lo=g.nlargest(K,"pred_long")["fwd6"].mean(); sh=g.nsmallest(K,"pred")["fwd6"].mean()
        rows.append((lo-sh)*1e4)
    ls=np.nanmean(rows); marg="" if prev is None else f"{ls*K-prev*(K-1):+.1f}"
    print(f"  {K:3d} {ls:+9.1f} {ls:+9.1f} {marg:>13s}")
    prev=ls
print("VDONE")
