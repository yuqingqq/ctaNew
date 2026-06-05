"""User's A-long filter check (2026-06-03): allow A-long only if (1) prev12h residual alpha<0 AND
(2) danger_score=z(atr)+z(idio_vol_1h)-z(corr_btc) not in top quartile. Rigor: PIT, per-fold stability,
redundancy with model pred, and SHARPE (not just mean) of the filtered long basket. Read-only."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
REPO=Path('/home/yuqing/ctaNew'); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
V0=list(tt.V0); ann=np.sqrt(6*365); OOS=pd.Timestamp('2025-10-04',tz='UTC')
PAN=pd.read_parquet(tt.PANEL,columns=['symbol','open_time','return_pct','alpha_vs_btc_realized','atr_pct',
    'idio_vol_to_btc_1h','corr_to_btc_1d','rvol_7d'])
PAN['open_time']=pd.to_datetime(PAN['open_time'],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(['symbol','open_time'])
PAN['alpha_A']=PAN['alpha_vs_btc_realized']
# prev12h residual alpha (PIT): trailing 3-bar sum of realized per-bar alpha, strictly before t
PAN['prev12h_resid']=PAN.groupby('symbol')['alpha_A'].transform(lambda s: s.shift(1).rolling(3).sum())
P=PAN[PAN.open_time>=OOS].copy()
g=P.groupby('open_time')
# danger_score (cross-sectional z per cycle)
for c in ['atr_pct','idio_vol_to_btc_1h','corr_to_btc_1d']:
    P[f'z_{c}']=g[c].transform(lambda x:(x-x.mean())/x.std())
P['danger']=P['z_atr_pct']+P['z_idio_vol_to_btc_1h']-P['z_corr_to_btc_1d']
# high-vol book + long candidates = top-10 pred/cycle
pr=pd.read_parquet(REPO/'live/state/convexity/hl_wfund175/fullflow_hl60.parquet')[['symbol','open_time','pred']]
pr['open_time']=pd.to_datetime(pr['open_time'],utc=True)
rv=P.groupby('symbol')['rvol_7d'].mean().sort_values(ascending=False); hivol=set(rv.index[:80])
D=P[P.symbol.isin(hivol)].merge(pr,on=['symbol','open_time'],how='inner').dropna(subset=['pred','alpha_A','prev12h_resid','danger'])
D['cand_rank']=D.groupby('open_time')['pred'].rank(ascending=False,method='first')
C=D[D.cand_rank<=10].copy()   # A-long candidate pool
casc=C['alpha_A'].quantile(0.10)   # cascade = worst-decile fwd alpha
dq=C['danger'].quantile(0.75)
def row(name,mask):
    s=C[mask]
    print(f'  {name:28s} keep={len(s)/len(C)*100:4.0f}%  mean_alpha={s.alpha_A.mean()*1e4:+6.2f}bps  '
          f'cascade={ (s.alpha_A<=casc).mean()*100:4.1f}%  cand_Sh(per-cyc basket)={basket_sh(s):+.2f}')
def basket_sh(s):
    # per-cycle mean alpha of kept candidates (proxy for the long basket), annualized Sharpe
    pc=s.groupby('open_time')['alpha_A'].mean()
    return pc.mean()/pc.std()*ann if len(pc)>30 and pc.std()>0 else np.nan
print('='*92); print('(1) REPRODUCE candidate-level table (PIT prev12h, top-10 pred A-long pool)')
row('all A-long', C.index==C.index)
row('prev12h residual < 0', C['prev12h_resid']<0)
row('exclude top danger quartile', C['danger']<dq)
row('both', (C['prev12h_resid']<0)&(C['danger']<dq))
print('\n(2) REDUNDANCY — is prev12h_resid<0 just the model pred?  is danger just its parts?')
print(f"  corr(prev12h_resid, pred)        = {C[['prev12h_resid','pred']].corr().iloc[0,1]:+.3f}   (high => filter redundant w/ mean-rev pred)")
print(f"  mean pred | prev12h<0  vs  >=0    = {C[C.prev12h_resid<0].pred.mean():+.3f}  vs  {C[C.prev12h_resid>=0].pred.mean():+.3f}")
print('\n(3) PER-FOLD stability of "both" filter mean alpha (artifact check)')
C['month']=C.open_time.dt.to_period('M')
both=C[(C.prev12h_resid<0)&(C.danger<dq)]
mm=both.groupby('month')['alpha_A'].mean()*1e4; allm=C.groupby('month')['alpha_A'].mean()*1e4
for m in mm.index: print(f"  {str(m)}  both={mm[m]:+7.2f}bps  all={allm.get(m,np.nan):+7.2f}bps  lift={mm[m]-allm.get(m,np.nan):+7.2f}  {'+' if mm[m]>allm.get(m,0) else '-'}")
print(f"  months both>all: {(mm>allm.reindex(mm.index)).sum()}/{len(mm)}")
print('\n(4) CONSTRUCTION reality: how often do >=3 candidates pass the filter (else long starves below K=3)?')
passcnt=both.groupby('open_time').size().reindex(C.open_time.unique()).fillna(0)
print(f"  cycles with >=3 passing: {(passcnt>=3).mean()*100:.0f}%   >=1: {(passcnt>=1).mean()*100:.0f}%   mean passing/cyc: {passcnt.mean():.1f}")
