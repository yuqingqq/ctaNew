"""Add COST to the v4 baseline (simple gate-free book). Measure ACTUAL per-cycle pick turnover and charge
COST_BPS/leg on the legs that changed vs the prior cycle. Report gross vs net at a few cost levels, per K.
v4 model = hl_tgt_res_long (V0_LEAN+resid_rev, residual target), both legs.
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
R="/home/yuqing/ctaNew"; H=6; ANN=np.sqrt(365)
pan=pd.read_parquet(f"{R}/outputs/vBTC_features/panel_expanded_v0.parquet",columns=["symbol","open_time","alpha_vs_btc_realized"])
pan["open_time"]=pd.to_datetime(pan["open_time"],utc=True); pan=pan.sort_values(["symbol","open_time"])
pan["fwd"]=pan.groupby("symbol")["alpha_vs_btc_realized"].transform(lambda s:s.shift(-1).rolling(H).sum().shift(-(H-1)))*1e4
p=pd.read_parquet(f"{R}/live/state/convexity/hl_tgt_res_long/v0full_hl60.parquet",columns=["symbol","open_time","pred"])
p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
d=p.merge(pan[["symbol","open_time","fwd"]],on=["symbol","open_time"])
d=d[d.open_time>=pd.Timestamp("2025-10-04",tz="UTC")].dropna(subset=["fwd"])
def run(KL,KS,cost_bps):
    prev=set(); ts=[]; gs=[]; ns=[]; turn=[]
    for ot,g in sorted(d.groupby("open_time"),key=lambda kv:kv[0]):
        if len(g)<KL+KS: continue
        L=set(g.nlargest(KL,"pred")["symbol"]); S=set(g.nsmallest(KS,"pred")["symbol"]); book=L|S
        gross=g[g.symbol.isin(L)]["fwd"].mean()-g[g.symbol.isin(S)]["fwd"].mean()
        # ROUGH turnover cost only — the overlapping 24h measurement makes the gross↔cost ratio ambiguous;
        # the authoritative net comes from the bot's turnover+depth+funding engine. See note in reply.
        changed=len(book^prev); prev=book
        cost=changed*cost_bps/H          # amortized-over-hold estimate (a FLOOR on cost impact)
        ts.append(ot); gs.append(gross); ns.append(gross-cost); turn.append(changed)
    g=pd.Series(gs,index=ts); n=pd.Series(ns,index=ts)
    def sh(x): dd=(x/1e4).resample("1D").sum(); return dd.mean()/dd.std()*ANN if dd.std()>0 else np.nan
    return sh(g),g.mean(),sh(n),n.mean(),np.mean(turn)
print("v4 baseline (V0_LEAN+resid_rev, residual target, both legs) — GROSS vs NET at cost/leg, cost amortized over 24h hold\n")
for KL,KS in [(1,2),(2,2)]:
    print(f"K={KL}/{KS}:  gross Sh {run(KL,KS,0)[0]:+.2f} (L/S {run(KL,KS,0)[1]:+.0f})  avg legs changed/cyc {run(KL,KS,0)[4]:.1f}")
    for c in [4.5,9,12]:
        gsh,gm,nsh,nm,tn=run(KL,KS,c)
        print(f"          cost {c:>4}bps/leg -> NET Sh {nsh:+.2f}  L/S {nm:+.1f}  (gave back {gm-nm:.0f}bps/cyc)")
print("VCDONE")
