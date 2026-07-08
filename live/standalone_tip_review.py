"""Review ALL features (V0_LEAN + resid_rev + alpha candidates) on STANDALONE tip-accuracy: rank names by each
feature alone, IC-oriented, measure top-K/bottom-K realized L/S {mean, per-cycle Sharpe=reliability, hit-rate} at
K=2 & K=3. Ranks which features' EXTREME values reliably predict extreme alpha (the property the strategy needs).
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6
import live.train_twobook_models as tt
V0=list(tt.V0_LEAN)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"]+V0)
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
a=pan.groupby("symbol")["alpha_vs_btc_realized"]
pan["resid_rev_2"]=-a.transform(lambda s:s.shift(1).rolling(2).sum()); pan["resid_rev_3"]=-a.transform(lambda s:s.shift(1).rolling(3).sum())
pan["fwd"]=a.transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
ALPHAS=["alpha082","alpha159","alpha095","alpha070","alpha023","alpha010","alpha052","alpha072","alpha088","alpha047"]
fac=pd.read_parquet(f"{R}/data/ml/cache/alpha191_factors_betaneut.parquet",columns=["symbol","open_time"]+ALPHAS)
fac["open_time"]=pd.to_datetime(fac["open_time"],utc=True); pan=pan.merge(fac,on=["symbol","open_time"],how="left")
d=pan[pan.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
FEATS=[("V0_LEAN",c) for c in V0]+[("resid_rev","resid_rev_2"),("resid_rev","resid_rev_3")]+[("alpha191",c) for c in ALPHAS]
def tip(col,K,sign):
    ls=[]
    for ot,g in d.groupby("open_time"):
        gg=g.dropna(subset=[col])
        if len(gg)<2*K: continue
        ls.append(sign*(gg.nlargest(K,col)["fwd"].mean()-gg.nsmallest(K,col)["fwd"].mean()))
    s=pd.Series(ls); return s.mean(), (s.mean()/s.std()*np.sqrt(len(s)) if s.std()>0 else np.nan), 100*(s>0).mean()
rows=[]
for grp,c in FEATS:
    ic=spearmanr(d[c],d["fwd"]).correlation; sgn=1.0 if (ic or 0)>=0 else -1.0
    m2,s2,h2=tip(c,2,sgn); m3,s3,h3=tip(c,3,sgn)
    rows.append((grp,c,ic,m2,s2,h2,m3,s3))
R2=pd.DataFrame(rows,columns=["grp","feat","avgIC","tipK2_mean","tipK2_Sh","tipK2_hit","tipK3_mean","tipK3_Sh"]).sort_values("tipK2_Sh",key=lambda s:s.abs(),ascending=False)
pd.set_option("display.width",160)
print("STANDALONE TIP-ACCURACY per feature (IC-oriented), ranked by K=2 reliability (|Sharpe|)\n")
print(f"{'grp':9s} {'feature':22s} {'avgIC':>7s} {'tipK2_mean':>10s} {'tipK2_Sh':>8s} {'hit%':>5s} {'tipK3_mean':>10s} {'tipK3_Sh':>8s}")
for _,r in R2.iterrows():
    print(f"{r.grp:9s} {r.feat:22s} {r.avgIC:+7.3f} {r.tipK2_mean:+10.1f} {r.tipK2_Sh:+8.2f} {r.tipK2_hit:4.0f}% {r.tipK3_mean:+10.1f} {r.tipK3_Sh:+8.2f}")
print("\n(sign-oriented; tipK2_mean>0 = feature's extremes predict L/S; higher Sh = more reliable at the tip)")
print("STRDONE")
