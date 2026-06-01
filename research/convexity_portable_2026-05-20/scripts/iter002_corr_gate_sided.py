"""iter-002 — SIDE-regime-conditioned corr gate vs blanket corr gate, with the
matched-random-timing placebo (G4 preview) done PROPERLY:

The honest placebo for a SIDE-conditioned gate must draw the throttled cycles from the
SAME eligible pool (side-regime, post-warmup) — otherwise the placebo gets to throttle
cycles the real rule never could, biasing the comparison. We run BOTH:
  - naive placebo (shuffle across ALL cycles, like iter-001)
  - eligible-pool placebo (shuffle only within side-regime eligible cycles)

We also test the gate as a binary SKIP (FLOOR=0) and a partial de-lever, on a PIT
expanding-percentile reference (no in-sample threshold tuning beyond a small grid that
the implementation agent will run nested-OOS).
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

corr=ctx["corr7d"].values; pnl=ctx["pnl"].copy(); n=len(pnl)
side=(ctx["regime"]=="side").values

# PIT expanding percentile rank of corr7d (strictly trailing, warmup 100)
pr=np.full(n,np.nan)
for i in range(n):
    if i<100 or not np.isfinite(corr[i]): continue
    hist=corr[:i]; hist=hist[np.isfinite(hist)]
    if len(hist)<50: continue
    pr[i]=np.mean(hist<=corr[i])

base_cal,base_dd=cal_dd(pnl.values); base_sh=ann(pnl/1e4)
print(f"BASE: Sharpe {base_sh:+.2f}  maxDD {base_dd:+.0f}  Calmar {base_cal:+.2f}  totPnL {pnl.sum():+.0f}\n")

def run(thr, floor, side_only):
    elig = (pr>=thr) & np.isfinite(pr)
    if side_only: elig = elig & side
    sc=np.where(elig,floor,1.0)
    gp=pnl.values*sc
    cal,dd=cal_dd(gp); sh=ann(gp/1e4)
    return gp, elig, cal, dd, sh

print(f"{'mode':<14}{'thr':>5}{'flr':>5}{'n_hot':>7}{'Sharpe':>8}{'maxDD':>8}{'DDcut':>7}{'Calmar':>8}{'totPnL':>9}")
for side_only in [False,True]:
    tag="side-only" if side_only else "blanket"
    for thr in [0.70,0.80]:
        for floor in [0.0,0.3]:
            gp,elig,cal,dd,sh=run(thr,floor,side_only)
            print(f"{tag:<14}{thr:>5.2f}{floor:>5.1f}{int(elig.sum()):>7}{sh:>+8.2f}{dd:>+8.0f}{(1-dd/base_dd)*100:>+6.0f}%{cal:>+8.2f}{gp.sum():>+9.0f}")

# ---- G4 placebo for the headline side-only config ----
def placebo(thr,floor,side_only,pool_side_only,seeds=500,seed0=11):
    gp,elig,real_cal,real_dd,real_sh=run(thr,floor,side_only)
    n_hot=int(elig.sum())
    # eligible pool to draw from
    if pool_side_only:
        pool=np.where(side & (pr>=0)& np.isfinite(pr) | True, side, side)  # placeholder
        pool=np.where(side)[0]
        pool=pool[pool>=100]
    else:
        pool=np.where(np.isfinite(pr))[0]
    rng=np.random.default_rng(seed0); cals=[]; dds=[]
    pv=pnl.values
    for s in range(seeds):
        pick=rng.choice(pool,size=min(n_hot,len(pool)),replace=False)
        sc=np.ones(n); sc[pick]=floor
        c,dd=cal_dd(pv*sc); cals.append(c); dds.append(dd)
    cals=np.array(cals); dds=np.array(dds)
    return real_cal,real_dd,cals,dds

for thr,floor in [(0.70,0.0),(0.80,0.0),(0.70,0.3)]:
    print(f"\n=== G4 placebo: SIDE-ONLY gate thr={thr} floor={floor} ===")
    rc,rdd,cals,dds=placebo(thr,floor,True,pool_side_only=True,seeds=500)
    print(f"  real Calmar {rc:+.2f} | placebo(side-pool) median {np.median(cals):+.2f} p95 {np.percentile(cals,95):+.2f} max {cals.max():+.2f} -> real pctile p{(cals<rc).mean()*100:.0f}")
    print(f"  real maxDD  {rdd:+.0f} | placebo median {np.median(dds):+.0f} -> real DD pctile p{(dds<rdd).mean()*100:.0f}")
    # also naive all-cycle pool placebo
    rc2,rdd2,cals2,dds2=placebo(thr,floor,True,pool_side_only=False,seeds=500)
    print(f"  [naive all-pool] placebo median {np.median(cals2):+.2f} p95 {np.percentile(cals2,95):+.2f} -> real pctile p{(cals2<rc2).mean()*100:.0f}")
