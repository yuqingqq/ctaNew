"""ITER1 (non-stop loop): ROOT CAUSE of the high-vol-long weakness — among the high-vol fallers the model LONGS
(top mean-rev pred = recent decliners), which PIT feature discriminates a BOUNCE from a CASCADE (keeps falling)?
Find the discriminator -> a conditional long filter. Read-only, PIT-strict."""
import sys, glob; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
REPO=Path('/home/yuqing/ctaNew'); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
V0=list(tt.V0)
PAN=pd.read_parquet(tt.PANEL,columns=['symbol','open_time','return_pct']+V0)
PAN['open_time']=pd.to_datetime(PAN['open_time'],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)]; PAN=PAN[PAN.open_time>=pd.Timestamp('2025-10-04',tz='UTC')]
g=PAN.groupby('open_time'); PAN['fwd']=PAN['return_pct']-g['return_pct'].transform('mean')
PAN['rvrank']=g['rvol_7d'].rank(ascending=False)
# preds (mean-rev): high pred = the model's LONG candidate
pr=pd.read_parquet(REPO/'live/state/convexity/hl_wfund175/fullflow_hl60.parquet')[['symbol','open_time','pred']]
pr['open_time']=pd.to_datetime(pr['open_time'],utc=True)
D=PAN.merge(pr,on=['symbol','open_time'],how='inner')
hi=D[D.rvrank<=80]                                  # high-vol book
longs=hi[hi.groupby('open_time')['pred'].rank(ascending=False,method='first')<=10]  # top-10 long candidates (broader for stats)
print(f'high-vol long candidates (top-10 pred/cyc): n={len(longs)}, win-rate {(longs.fwd>0).mean():.3f}, mean fwd {longs.fwd.mean():+.5f}')
print('\\n=== which feature discriminates BOUNCE (fwd>0) vs CASCADE among high-vol long candidates? (PIT) ===')
def ic(df,c): s=df[[c,'fwd']].dropna(); return s[c].corr(s['fwd'],method='spearman') if len(s)>200 else np.nan
rows=[]
for f in V0:
    rows.append((f, ic(longs,f)))
R=pd.DataFrame(rows,columns=['feat','IC_within_longs']).reindex(pd.DataFrame(rows,columns=['feat','IC']).IC.abs().sort_values(ascending=False).index)
print(R.head(12).round(4).to_string(index=False))
# tail check: worst-decile (cascades) vs rest — feature signature
print('\\n=== CASCADE signature: worst-decile-fwd long candidates vs rest (z-gap of feature means) ===')
tail=longs[longs.fwd<=longs.fwd.quantile(0.10)]; rest=longs[longs.fwd>longs.fwd.quantile(0.10)]
zr=[]
for f in V0:
    a,b=tail[f].dropna(),rest[f].dropna()
    if len(a)>50 and PAN[f].std()>0: zr.append((f,(a.mean()-b.mean())/PAN[f].std()))
ZR=pd.DataFrame(zr,columns=['feat','z_gap']); ZR=ZR.reindex(ZR.z_gap.abs().sort_values(ascending=False).index)
print(ZR.head(10).round(3).to_string(index=False))
