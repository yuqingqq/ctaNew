"""LOOP2 iter-01 — WHY April (and Oct)? Decompose monthly edge into IC × cross-sectional dispersion.
Coarse regime doesn't explain it (April≈Jan/Mar). Hypothesis: money-months = high model IC AND/OR
high cross-sectional return dispersion (more spread to capture). If a PIT-observable (trailing
realized dispersion) predicts the good months, it's a sizing lever.
"""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
import warnings; warnings.filterwarnings("ignore")
REPO="/home/yuqing/ctaNew"
p=pd.read_parquet(f"{REPO}/live/state/convexity/xsz60_preds.parquet",columns=["symbol","open_time","return_pct","pred"])
p["open_time"]=pd.to_datetime(p["open_time"],utc=True)
H1S=pd.Timestamp("2025-10-04",tz="UTC");E=pd.Timestamp("2026-05-26",tz="UTC")
p=p[(p.open_time>=H1S)&(p.open_time<E)]; p["m"]=p.open_time.dt.strftime("%Y-%m")
K=3
# per-cycle metrics
rows=[]
for ot,g in p.groupby("open_time"):
    if len(g)<2*K: continue
    ic=spearmanr(g["pred"],g["return_pct"])[0]
    med=g["return_pct"].median()
    longe=g.nlargest(K,"pred")["return_pct"].mean()-med
    shorte=med-g.nsmallest(K,"pred")["return_pct"].mean()
    rows.append(dict(open_time=ot,m=ot.strftime("%Y-%m"),ic=ic,
                     xs_disp=g["return_pct"].std()*1e4, ls_edge=(longe+shorte)*1e4))
c=pd.DataFrame(rows)
print("=== per-month: IC, xs_dispersion, L/S selection edge ===")
print(f"  {'month':<9}{'meanIC':>9}{'xs_disp':>9}{'LS_edge':>9}{'nCyc':>6}")
mm=c.groupby("m").agg(ic=("ic","mean"),disp=("xs_disp","mean"),ls=("ls_edge","mean"),n=("ic","size"))
for m,r in mm.iterrows(): print(f"  {m:<9}{r.ic:>+9.4f}{r.disp:>9.0f}{r.ls:>+9.1f}{int(r.n):>6}")
# does monthly LS edge track IC, dispersion, or IC*disp?
print("\n=== drivers of monthly L/S edge (cross-month corr) ===")
mm["ic_x_disp"]=mm.ic*mm.disp
for col in ["ic","disp","ic_x_disp"]:
    r=pearsonr(mm[col],mm["ls"])[0]
    print(f"  corr(month {col:<10} , LS_edge) = {r:+.2f}")
# per-CYCLE: is LS edge driven by IC, dispersion, or interaction? (more data)
print("\n=== per-cycle drivers (n cycles) ===")
c["ic_x_disp"]=c.ic*c.xs_disp
for col in ["ic","xs_disp","ic_x_disp"]:
    r=spearmanr(c[col],c["ls_edge"])[0]
    print(f"  spearman(cycle {col:<10} , LS_edge) = {r:+.2f}")
# PIT test: does TRAILING dispersion (through t-1) predict current-cycle LS edge?
c=c.sort_values("open_time").reset_index(drop=True)
c["tr_disp"]=c["xs_disp"].rolling(42,min_periods=20).mean().shift(1)
c["tr_ic"]=c["ic"].rolling(42,min_periods=20).mean().shift(1)
sub=c.dropna(subset=["tr_disp","tr_ic"])
print("\n=== PIT predictors of current LS edge (trailing 7d, shifted) ===")
for col in ["tr_disp","tr_ic"]:
    r=spearmanr(sub[col],sub["ls_edge"])[0]
    q=pd.qcut(sub[col],4,labels=False,duplicates="drop")
    print(f"  {col}: spearman {r:+.3f} | Q1 LS {sub['ls_edge'][q==0].mean():+.1f} Q4 {sub['ls_edge'][q==q.max()].mean():+.1f}")
print("\n  => if trailing-disp Q4>>Q1 LS edge AND PIT, it's a sizing lever (size up high-disp regimes).")
EOF_MARKER_DONE=1
print("APRIL_MECH2 DONE")
