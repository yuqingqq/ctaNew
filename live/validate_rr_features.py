"""Fair test: do the resid_rev features help? Simple gate-free book (fixed K, equal weight, gross, no gates), only
the ranking pred differs. 4 wirings of the two residual-target books (base=V0_LEAN, rr=V0_LEAN+resid_rev):
  base_both      : long & short by base        (NO resid_rev anywhere)
  rr_both        : long & short by rr           (resid_rev on both legs)
  split (prod)   : long by rr, short by base    (resid_rev on long only)
  inverse        : long by base, short by rr    (resid_rev on short only)
'RR helps' = rr_both/split beat base_both. 'split earns keep' = split beats rr_both. + robustness on rr_both vs base_both.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; ANN=np.sqrt(365)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
b=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_base/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"base"})
l=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"]).rename(columns={"pred":"rr"})
for x in (b,l): x["open_time"]=pd.to_datetime(x["open_time"],utc=True)
d=b.merge(l,on=["symbol","open_time"]).merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
def run(longcol,shortcol,KL,KS):
    rows=[]
    for ot,g in d.groupby("open_time"):
        if len(g)<KL+KS: continue
        rows.append((ot, g.nlargest(KL,longcol)["fwd"].mean()-g.nsmallest(KS,shortcol)["fwd"].mean()))
    s=pd.Series({o:v for o,v in rows}); dd=(s/1e4).resample("1D").sum()
    sh=dd.mean()/dd.std()*ANN if dd.std()>0 else np.nan; eq=dd.cumsum(); mdd=float((eq-eq.cummax()).min()*1e4)
    return sh,s.mean(),s.sum(),mdd,s
WIR=[("base_both","base","base"),("rr_both","rr","rr"),("split(prod) L=rr S=base","rr","base"),("inverse L=base S=rr","base","rr")]
for KL,KS in [(2,2),(1,2)]:
    print(f"\n=== K_long={KL} K_short={KS} — simple gate-free book, gross 24h L/S (bps) ===")
    print(f"  {'wiring':26s} {'Sharpe':>7s} {'L/S':>7s} {'totPnL':>8s} {'maxDD':>8s}")
    base_s=None
    for lbl,lc,sc in WIR:
        sh,mn,tot,mdd,s=run(lc,sc,KL,KS)
        if lbl=="base_both": base_s=s
        print(f"  {lbl:26s} {sh:+7.2f} {mn:+7.1f} {tot:+8.0f} {mdd:+8.0f}")
    # robustness: rr_both - base_both paired per cycle (does RR help broadly?)
    _,_,_,_,rr_s=run("rr","rr",KL,KS)
    idx=base_s.index.intersection(rr_s.index); diff=(rr_s[idx]-base_s[idx]).sort_index()
    h1,h2=np.array_split(diff.values,2)
    print(f"  [robustness rr_both - base_both] mean{diff.mean():+.1f} med{diff.median():+.1f} %pos{100*(diff>0).mean():.0f} "
          f"top3{diff.nlargest(3).sum()/diff.sum()*100 if diff.sum()!=0 else 0:.0f}% half1{h1.mean():+.1f} half2{h2.mean():+.1f}")
print("VRRDONE")
