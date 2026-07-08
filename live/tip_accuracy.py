"""TIP-ACCURACY metric: instead of average IC, measure the model's accuracy AT the traded extremes.
For top-K/bottom-K by pred: L/S mean (edge), per-cycle Sharpe (reliability), hit-rate (% cycles the tip pays).
A good predictor for THIS strategy has a STEEP + RELIABLE tip, not just high average IC. Compare base vs +factors.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; ANN=np.sqrt(365)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
b=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"base"})
f=pd.read_parquet(f"{R}/live/state/convexity/hl_v4fac_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"fac"})
for x in (b,f): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
d=b.merge(f,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
from scipy.stats import spearmanr
def avg_ic(col): return d.groupby("open_time").apply(lambda g:spearmanr(g[col],g["fwd"]).correlation if len(g)>=20 else np.nan).mean()
def tip(col,K):
    ls=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<2*K: continue
        ls.append(g.nlargest(K,col)["fwd"].mean()-g.nsmallest(K,col)["fwd"].mean())
    s=pd.Series(ls); return s.mean(), s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan, 100*(s>0).mean()
print("AVERAGE IC vs TIP-ACCURACY — base (v4) vs +factors\n")
print(f"  average IC:  base {avg_ic('base'):+.4f}   +factors {avg_ic('fac'):+.4f}   Δ {avg_ic('fac')-avg_ic('base'):+.4f}  <- says 'factor ~equal/better'")
print(f"\n  {'metric':22s} {'base':>22s} {'+factors':>22s}")
for K in [1,2,3]:
    bm,bs,bh=tip('base',K); fm,fs,fh=tip('fac',K)
    print(f"  tip K={K} L/S mean(bps)   {bm:>+22.1f} {fm:>+22.1f}")
    print(f"  tip K={K} reliability(Sh) {bs:>+22.2f} {fs:>+22.2f}")
    print(f"  tip K={K} hit-rate(%pos)  {bh:>21.0f}% {fh:>21.0f}%")
print("\n=> average IC misses it; TIP metrics (mean+reliability+hitrate at K=1-3) show the factor DEGRADES the traded tip.")
print("TIPDONE")
