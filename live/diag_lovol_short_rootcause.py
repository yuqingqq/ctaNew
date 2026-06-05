"""ITER3 (non-stop loop): ROOT CAUSE of the low-vol-short weakness — among the low-vol names the model SHORTS
(bottom mean-rev pred = recent pumps), which PIT feature discriminates a GOOD short (fwd<0, drops) from a
SQUEEZE (fwd>0, rips up against us)? Mirror of iter1. Find the discriminator -> a conditional short filter.
Read-only, PIT-strict. Short PnL = -fwd, so a good short has fwd<0."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
REPO=Path('/home/yuqing/ctaNew'); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
V0=list(tt.V0)
PAN=pd.read_parquet(tt.PANEL,columns=['symbol','open_time','return_pct']+V0)
PAN['open_time']=pd.to_datetime(PAN['open_time'],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)]; PAN=PAN[PAN.open_time>=pd.Timestamp('2025-10-04',tz='UTC')]
g=PAN.groupby('open_time'); PAN['fwd']=PAN['return_pct']-g['return_pct'].transform('mean')
PAN['rvrank']=g['rvol_7d'].rank(ascending=False)
pr=pd.read_parquet(REPO/'live/state/convexity/hl_wfund175/v0full_hl60.parquet')[['symbol','open_time','pred']]
pr['open_time']=pd.to_datetime(pr['open_time'],utc=True)
D=PAN.merge(pr,on=['symbol','open_time'],how='inner')
lo=D[D.rvrank>80]                                   # low-vol (price) book
# short candidates = LOWEST pred (model says overbought / will mean-revert down)
shorts=lo[lo.groupby('open_time')['pred'].rank(ascending=True,method='first')<=10]
# short PnL is -fwd: GOOD short => fwd<0. win-rate of the SHORT:
print(f'low-vol short candidates (bottom-10 pred/cyc): n={len(shorts)}, short-win-rate {(shorts.fwd<0).mean():.3f}, mean short PnL {(-shorts.fwd).mean():+.5f}')
# IC of feature with SHORT PnL (-fwd): positive IC => higher feature -> better short
print('\\n=== which feature discriminates a GOOD short vs a SQUEEZE? (IC with short PnL=-fwd, PIT) ===')
def ic(df,c): s=df[[c]].assign(sp=-df['fwd']).dropna(); return s[c].corr(s['sp'],method='spearman') if len(s)>200 else np.nan
rows=[(f, ic(shorts,f)) for f in V0]
R=pd.DataFrame(rows,columns=['feat','IC_short_pnl']); R=R.reindex(R.IC_short_pnl.abs().sort_values(ascending=False).index)
print(R.head(12).round(4).to_string(index=False))
# SQUEEZE signature: worst-short-decile (biggest fwd>0 rip) vs rest
print('\\n=== SQUEEZE signature: worst-short-decile (largest fwd, rips up) vs rest (z-gap of feature means) ===')
sq=shorts[shorts.fwd>=shorts.fwd.quantile(0.90)]; rest=shorts[shorts.fwd<shorts.fwd.quantile(0.90)]
zr=[]
for f in V0:
    a,b=sq[f].dropna(),rest[f].dropna()
    if len(a)>50 and PAN[f].std()>0: zr.append((f,(a.mean()-b.mean())/PAN[f].std()))
ZR=pd.DataFrame(zr,columns=['feat','z_gap']); ZR=ZR.reindex(ZR.z_gap.abs().sort_values(ascending=False).index)
print(ZR.head(10).round(3).to_string(index=False))
