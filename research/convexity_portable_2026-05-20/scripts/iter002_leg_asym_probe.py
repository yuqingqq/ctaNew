"""iter-002 — leg-asymmetry probe. The anatomy shows the HL70 deep DD is ENTIRELY a
long-leg event (in-DD long -6142 bps, short +828). High trailing alt-correlation (corr7d)
PIT-precedes the bad cycles, and within SIDE regime hi-corr cycles lose while lo-corr win.

Mechanism hypothesis: when alts are highly co-moving (corr7d high), the cross-sectional
mean-reversion spread collapses -- the long leg becomes "long the alts that fell with the
market" = long beta into a grind, with no idiosyncratic bounce. The short leg is unharmed.

Test a STRUCTURAL leg-asymmetry: in SIDE regime when PIT corr-percentile is high, reduce
ONLY the long-leg weight (toward 0) while keeping the short leg. This directly attacks the
losing leg rather than throttling the whole book.

To do this from the context parquet we need per-cycle long/short pnl, which we have
(long_pnl, short_pnl). Reducing long leg by factor f scales long_pnl by f (the short leg
and its pnl are unchanged; cost approx unchanged since short turnover dominates -- we
approximate cost as unchanged, a slight conservatism). This is a quick structural preview;
the implementation agent rebuilds it inside the engine with exact turnover/cost.
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

corr=ctx["corr7d"].values; n=len(ctx)
side=(ctx["regime"]=="side").values
lp=ctx["long_pnl"].values; sp=ctx["short_pnl"].values; base=ctx["pnl"].values

pr=np.full(n,np.nan)
for i in range(n):
    if i<100 or not np.isfinite(corr[i]): continue
    hist=corr[:i]; hist=hist[np.isfinite(hist)]
    if len(hist)<50: continue
    pr[i]=np.mean(hist<=corr[i])

base_cal,base_dd=cal_dd(base); base_sh=ann(pd.Series(base)/1e4)
# note: long_pnl+short_pnl != pnl exactly (pnl has -cost). reconstruct cost residual:
cost_resid = base - (lp+sp)   # = -cost per cycle
print(f"BASE: Sharpe {base_sh:+.2f}  maxDD {base_dd:+.0f}  Calmar {base_cal:+.2f}  totPnL {base.sum():+.0f}")
print(f"(cost residual mean {cost_resid.mean():+.2f} bps/cyc)\n")

print(f"{'rule':<40}{'n_hot':>6}{'Sharpe':>8}{'maxDD':>8}{'DDcut':>7}{'Calmar':>8}{'totPnL':>9}")
def report(name, scale_long):
    # scale_long: per-cycle multiplier on the LONG leg only
    gp = scale_long*lp + sp + cost_resid
    cal,dd=cal_dd(gp); sh=ann(pd.Series(gp)/1e4)
    nh=int((scale_long<1.0).sum())
    print(f"{name:<40}{nh:>6}{sh:>+8.2f}{dd:>+8.0f}{(1-dd/base_dd)*100:>+6.0f}%{cal:>+8.2f}{gp.sum():>+9.0f}")
    return gp,cal,dd

for thr in [0.70,0.80]:
    for lf in [0.0,0.5]:
        hot=(pr>=thr)&np.isfinite(pr)&side
        sl=np.where(hot,lf,1.0)
        report(f"side hi-corr(thr={thr}) long*={lf}", sl)

# compare to throttling BOTH legs by same factor on same cycles (the "uniform de-lever" control)
for thr in [0.70]:
    for lf in [0.0,0.5]:
        hot=(pr>=thr)&np.isfinite(pr)&side
        gp=np.where(hot,lf,1.0)*base
        cal,dd=cal_dd(gp); sh=ann(pd.Series(gp)/1e4)
        print(f"{'  [ctrl] both-legs*'+str(lf)+' thr'+str(thr):<40}{int(hot.sum()):>6}{sh:>+8.2f}{dd:>+8.0f}{(1-dd/base_dd)*100:>+6.0f}%{cal:>+8.2f}{gp.sum():>+9.0f}")

# ---- G4 placebo: the long-leg-cut effect vs random-timing same-count long-leg cuts within side pool ----
print("\n=== G4 placebo: long-leg-cut in side hi-corr (thr=0.70 lf=0.0), within-side-pool, 500 seeds ===")
thr,lf=0.70,0.0
hot=(pr>=thr)&np.isfinite(pr)&side
real_gp=np.where(hot,lf,1.0)*lp + sp + cost_resid
real_cal,real_dd=cal_dd(real_gp)
n_hot=int(hot.sum()); pool=np.where(side)[0]; pool=pool[pool>=100]
rng=np.random.default_rng(13); cals=[]; dds=[]
for s in range(500):
    pick=rng.choice(pool,size=min(n_hot,len(pool)),replace=False)
    slv=np.ones(n); slv[pick]=lf
    gp=slv*lp+sp+cost_resid; c,dd=cal_dd(gp); cals.append(c); dds.append(dd)
cals=np.array(cals); dds=np.array(dds)
print(f"  real Calmar {real_cal:+.2f} | placebo median {np.median(cals):+.2f} p95 {np.percentile(cals,95):+.2f} max {cals.max():+.2f} -> real pctile p{(cals<real_cal).mean()*100:.0f}")
print(f"  real maxDD {real_dd:+.0f} | placebo median {np.median(dds):+.0f} -> real DD pctile p{(dds<real_dd).mean()*100:.0f}")
