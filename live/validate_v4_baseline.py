"""Confirm the v4 baseline: single model V0_LEAN+resid_rev, BOTH legs. Check the residual target still beats the
return target under THIS rr_both wiring (target benefit was only shown under the old split wiring). Simple gate-free
book, gross, per K. Both use V0_LEAN+RR for both legs — only the training LABEL differs.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; ANN=np.sqrt(365)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
def load1(book):  # single model used for both legs
    p=pd.read_parquet(f"{R}/live/state/convexity/{book}/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
    p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
    m=p.merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
    return m[m.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
def run(m,KL,KS):
    rows={}
    for ot,g in m.groupby("open_time"):
        if len(g)<KL+KS: continue
        rows[ot]=g.nlargest(KL,"pred")["fwd"].mean()-g.nsmallest(KS,"pred")["fwd"].mean()
    s=pd.Series(rows); dd=(s/1e4).resample("1D").sum(); sh=dd.mean()/dd.std()*ANN if dd.std()>0 else np.nan
    eq=dd.cumsum(); mdd=float((eq-eq.cummax()).min()*1e4); return sh,s.mean(),s.sum(),mdd,s
print("v4 baseline candidate = single model V0_LEAN+resid_rev, BOTH legs. Target: RETURN vs RESIDUAL, simple book, gross\n")
for KL,KS in [(1,2),(2,2),(3,3)]:
    rr,mr,tr,dr,sr=run(load1("hl_tgt_ret_long"),KL,KS)   # return-target V0_LEAN+RR
    re,me,te,de,se=run(load1("hl_tgt_res_long"),KL,KS)   # residual-target V0_LEAN+RR
    idx=sr.index.intersection(se.index); diff=(se[idx]-sr[idx]).sort_index(); h1,h2=np.array_split(diff.values,2)
    print(f"K={KL}/{KS}:  RETURN Sh {rr:+.2f} L/S {mr:+.1f} maxDD {dr:+.0f}  |  RESIDUAL Sh {re:+.2f} L/S {me:+.1f} maxDD {de:+.0f}  |  Δ {re-rr:+.2f}")
    print(f"          [res-ret paired] mean{diff.mean():+.1f} med{diff.median():+.1f} %pos{100*(diff>0).mean():.0f} half1{h1.mean():+.1f} half2{h2.mean():+.1f}")
print("V4BLDONE")
