"""iter-002 follow-up — verify the corr7d mechanism is (a) PIT-predictive (lagged),
(b) not just a bear-regime proxy, (c) actionable as a per-cycle gate, and (d) size the
candidate gate's effect honestly before handing off.

Loads the context parquet produced by iter002_hl70_dd_anatomy.py.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np

REPO=Path("/home/yuqing/ctaNew")
OUT=REPO/"research/convexity_portable_2026-05-20/results"
ctx=pd.read_parquet(OUT/"iter002_hl70_context.parquet")
def ann(x):
    x=pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan

print("=== corr7d mechanism verification ===\n")
# corr7d is built strictly-trailing (sub=rt.iloc[lo:i], excludes current). It is PIT for cycle i.
# (a) lag test: does corr7d at t-1 still predict pnl_t? shift corr7d forward by 1 cycle.
c=ctx[["corr7d","pnl","regime"]].dropna().copy()
c["corr_lag1"]=c["corr7d"].shift(1)
cc=c.dropna(subset=["corr_lag1"])
cc["q"]=pd.qcut(cc["corr_lag1"],5,labels=False,duplicates="drop")
print("(a) NEXT-cycle pnl by lagged(t-1) corr7d quintile -- pure out-of-info-set:")
for q,g in cc.groupby("q"):
    print(f"   q{q}: mean {g['pnl'].mean():+6.1f} bps  Sharpe {ann(g['pnl']/1e4):+5.2f}  n={len(g)}")

# (b) regime breakdown: corr7d effect WITHIN side regime (bear is already flat, so gate only bites side/bull)
print("\n(b) within-regime: hi-corr (top 30%) vs lo-corr (bot 30%) book PnL:")
for rg in ["side","bull"]:
    cr=ctx[ctx["regime"]==rg][["corr7d","pnl"]].dropna()
    if len(cr)<30: continue
    hi=cr[cr["corr7d"]>=cr["corr7d"].quantile(0.7)]; lo=cr[cr["corr7d"]<=cr["corr7d"].quantile(0.3)]
    print(f"   {rg:<5} hi(n={len(hi)}): mean {hi['pnl'].mean():+.1f} Sh {ann(hi['pnl']/1e4):+.2f} | lo(n={len(lo)}): mean {lo['pnl'].mean():+.1f} Sh {ann(lo['pnl']/1e4):+.2f}")

# (c) candidate gate: scale book by g_t where corr7d is high. Test a few PIT thresholds.
# Use a PIT trailing-median/quantile reference so the threshold is not a free in-sample param.
print("\n(c) candidate corr-gate (PIT expanding quantile reference, scale to FLOOR when corr in top band):")
corr=ctx["corr7d"]
pnl=ctx["pnl"].copy()
regime=ctx["regime"]
# expanding PIT quantile of corr7d (lagged by 1 to be safe), warmup 100
exp_q=pd.Series(index=ctx.index,dtype=float)
exp_q80=pd.Series(index=ctx.index,dtype=float)
vals=corr.values
for i in range(len(corr)):
    if i<100 or not np.isfinite(vals[i]):
        continue
    hist=vals[:i]; hist=hist[np.isfinite(hist)]  # strictly trailing
    if len(hist)<50: continue
    exp_q.iloc[i]=np.mean(hist<=vals[i])  # PIT percentile rank of current corr vs history
print("   built PIT percentile rank of corr7d (expanding, strictly trailing, warmup 100)")
# gate variants: when corr percentile >= THR, scale book to FLOOR
base_sh=ann(pnl/1e4); base_eq=pnl.cumsum(); base_dd=(base_eq-base_eq.cummax()).min()
base_cal=pnl.mean()*6*365/abs(base_dd)
print(f"   BASE: Sharpe {base_sh:+.2f}  maxDD {base_dd:+.0f}  Calmar {base_cal:+.2f}  totPnL {base_eq.iloc[-1]:+.0f}")
for THR in [0.70,0.80,0.90]:
    for FLOOR in [0.0,0.3,0.5]:
        scale=pd.Series(1.0,index=ctx.index)
        hot=(exp_q>=THR).fillna(False)
        scale[hot]=FLOOR
        gp=pnl*scale
        eq=gp.cumsum(); dd=(eq-eq.cummax()).min(); cal=gp.mean()*6*365/abs(dd) if dd<0 else np.nan
        n_hot=int(hot.sum())
        print(f"   THR={THR:.2f} FLOOR={FLOOR:.1f}: Sharpe {ann(gp/1e4):+.2f} maxDD {dd:+.0f} ({(1-dd/base_dd)*100:+.0f}% vs base) Calmar {cal:+.2f} totPnL {eq.iloc[-1]:+.0f} (n_hot={n_hot}, {n_hot/len(ctx)*100:.0f}%)")

# (d) quick matched-placebo preview for the most promising config: shuffle the SAME number of FLOOR cycles
print("\n(d) matched-placebo preview (shuffle which cycles get throttled, same count), 300 seeds:")
THR,FLOOR=0.80,0.0
hot=(exp_q>=THR).fillna(False); n_hot=int(hot.sum())
gp=pnl*np.where(hot,FLOOR,1.0)
real_eq=gp.cumsum(); real_dd=(real_eq-real_eq.cummax()).min(); real_cal=gp.mean()*6*365/abs(real_dd)
rng=np.random.default_rng(7); pl_cal=[]; pl_dd=[]
pnlv=pnl.values; n=len(pnlv)
for s in range(300):
    idx=rng.choice(n,size=n_hot,replace=False)
    sc=np.ones(n); sc[idx]=FLOOR
    g=pnlv*sc; eq=np.cumsum(g); dd=(eq-np.maximum.accumulate(eq)).min()
    pl_cal.append(g.mean()*6*365/abs(dd)); pl_dd.append(dd)
pl_cal=np.array(pl_cal)
print(f"   real Calmar {real_cal:+.2f}  placebo median {np.median(pl_cal):+.2f}  p95 {np.percentile(pl_cal,95):+.2f}  max {pl_cal.max():+.2f}")
print(f"   real Calmar percentile vs placebo: p{(pl_cal<real_cal).mean()*100:.0f}")
print(f"   real maxDD {real_dd:+.0f}  placebo median maxDD {np.median(pl_dd):+.0f}  real pctile p{(np.array(pl_dd)<real_dd).mean()*100:.0f}")
