"""iter-002 — CONTINUOUS corr-conditioned sizing vs matched placebo.

Signal: PIT expanding-percentile rank of corr7d (avg trailing-7d pairwise alt-return
correlation), strictly trailing. The anatomy shows this rank is monotone-predictive of
next book PnL (q0 +4.8 Sh ... q4 -0.4 Sh) and the deep DD lives in its top band.

Continuous sizing law (SIDE regime only; bull/bear untouched):
    g_t = clip(1 - LAMBDA*(pr_t - 0.5)_+ , FLOOR, 1.0)
i.e. de-lever progressively as correlation rises above its own median; NEVER lever up.
Structural (parameter-light): only LAMBDA (slope) and FLOOR. Reference (median 0.5) is
intrinsic to the percentile transform, not tuned.

Matched placebo for a CONTINUOUS overlay: keep the SAME multiset of g_t values but
permute WHICH side-eligible cycle each is applied to (matched average-gross + matched
size-distribution, random timing). >=p95 required.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd, numpy as np

REPO=Path("/home/yuqing/ctaNew")
OUT=REPO/"research/convexity_portable_2026-05-20/results"
ctx=pd.read_parquet(OUT/"iter002_hl70_context.parquet")
def ann(x):
    x=pd.Series(x).dropna(); return x.mean()/x.std()*np.sqrt(6*365) if len(x)>2 and x.std()>0 else np.nan
def cal_dd(g):
    g=np.asarray(g); eq=np.cumsum(g); dd=(eq-np.maximum.accumulate(eq)).min()
    return (g.mean()*6*365/abs(dd) if dd<0 else np.nan), dd

corr=ctx["corr7d"].values; n=len(ctx); base=ctx["pnl"].values
side=(ctx["regime"]=="side").values
pr=np.full(n,np.nan)
for i in range(n):
    if i<100 or not np.isfinite(corr[i]): continue
    hist=corr[:i]; hist=hist[np.isfinite(hist)]
    if len(hist)<50: continue
    pr[i]=np.mean(hist<=corr[i])

base_cal,base_dd=cal_dd(base); base_sh=ann(pd.Series(base)/1e4)
print(f"BASE: Sharpe {base_sh:+.2f} maxDD {base_dd:+.0f} Calmar {base_cal:+.2f} totPnL {base.sum():+.0f}\n")

def gscale(lam,floor):
    g=np.ones(n)
    elig=np.isfinite(pr)&side
    over=np.clip(pr-0.5,0,None)
    g[elig]=np.clip(1-lam*over[elig],floor,1.0)
    return g,elig

print(f"{'LAMBDA':>7}{'FLOOR':>6}{'avg_g':>7}{'Sharpe':>8}{'maxDD':>8}{'DDcut':>7}{'Calmar':>8}{'totPnL':>9}")
configs=[(1.0,0.3),(1.5,0.3),(2.0,0.0),(1.5,0.0),(2.0,0.3)]
for lam,floor in configs:
    g,elig=gscale(lam,floor)
    gp=g*base; cal,dd=cal_dd(gp); sh=ann(pd.Series(gp)/1e4)
    print(f"{lam:>7.1f}{floor:>6.1f}{g[elig].mean():>7.3f}{sh:>+8.2f}{dd:>+8.0f}{(1-dd/base_dd)*100:>+6.0f}%{cal:>+8.2f}{gp.sum():>+9.0f}")

# placebo for headline config
lam,floor=1.5,0.0
g,elig=gscale(lam,floor)
real_gp=g*base; real_cal,real_dd=cal_dd(real_gp); real_sh=ann(pd.Series(real_gp)/1e4)
print(f"\n=== G4 matched placebo: continuous LAMBDA={lam} FLOOR={floor} (permute g within side-eligible pool, 1000 seeds) ===")
pool=np.where(elig)[0]; gvals=g[pool].copy()
rng=np.random.default_rng(21); cals=[]; dds=[]; shs=[]
for s in range(1000):
    perm=rng.permutation(len(pool))
    gg=np.ones(n); gg[pool]=gvals[perm]
    gp=gg*base; c,dd=cal_dd(gp); cals.append(c); dds.append(dd); shs.append(ann(pd.Series(gp)/1e4))
cals=np.array(cals); dds=np.array(dds); shs=np.array(shs)
print(f"  real Calmar {real_cal:+.2f} | placebo median {np.median(cals):+.2f} p95 {np.percentile(cals,95):+.2f} max {cals.max():+.2f} -> p{(cals<real_cal).mean()*100:.0f}")
print(f"  real Sharpe {real_sh:+.2f} | placebo median {np.median(shs):+.2f} p95 {np.percentile(shs,95):+.2f} -> p{(shs<real_sh).mean()*100:.0f}")
print(f"  real maxDD {real_dd:+.0f} | placebo median {np.median(dds):+.0f} -> p{(dds<real_dd).mean()*100:.0f}")
