"""Explore genuinely DIFFERENT signal families (user: keep exploring). On long candidates (both books),
orthIC of each vs BOTH base pred AND resid_rev (so we only chase truly-new signal). Families:
 peer_rev   : cross-sectional reversion (return vs UNIVERSE mean, not BTC-beta-residual) — sector rotation
 funding_accel/funding_mom : funding term-structure (positioning dynamics, not price)
 vol_term   : short/long realized-vol ratio (vol expansion regime, per name)
 autocorr   : autocorr_pctile_7d (V0) — does the model already use it?
 retmom_3d  : longer-horizon raw momentum (continuation vs reversal)
Read-only, PIT."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
REPO=Path('/home/yuqing/ctaNew'); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
OOS=pd.Timestamp('2025-10-04',tz='UTC')
cols=['symbol','open_time','alpha_vs_btc_realized','return_pct','rvol_7d','funding_rate','funding_rate_1d_change',
      'autocorr_pctile_7d','ret_3d','return_1d']
PAN=pd.read_parquet(tt.PANEL,columns=cols); PAN['open_time']=pd.to_datetime(PAN['open_time'],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(['symbol','open_time'])
PAN['alpha_A']=PAN['alpha_vs_btc_realized']
a=PAN.groupby('symbol')['alpha_A']
PAN['resid_rev']=-a.transform(lambda s:s.shift(1).rolling(3).sum())     # the known signal (12h)
# peer reversion: cross-sectional demeaned return, trailing reversal (vs universe, not BTC)
g0=PAN.groupby('open_time'); PAN['xs_demean_ret']=PAN['return_pct']-g0['return_pct'].transform('mean')
PAN['peer_rev']=-PAN.groupby('symbol')['xs_demean_ret'].transform(lambda s:s.shift(1).rolling(3).sum())
# funding term structure
f=PAN.groupby('symbol')
PAN['funding_accel']=f['funding_rate_1d_change'].transform(lambda s:s.shift(1)-s.shift(2))
PAN['funding_mom']=f['funding_rate'].transform(lambda s:s.shift(1).rolling(6).mean())
# vol term structure (short 7d vs long 30d-ish via rolling of rvol)
PAN['vol_term']=(PAN['rvol_7d']/f['rvol_7d'].transform(lambda s:s.shift(1).rolling(42).mean())).clip(0,5)
PAN['autocorr']=f['autocorr_pctile_7d'].transform(lambda s:s.shift(1))
PAN['retmom_3d']=f['ret_3d'].transform(lambda s:s.shift(1))
P=PAN[PAN.open_time>=OOS].copy()
rv=P.groupby('symbol')['rvol_7d'].mean().sort_values(ascending=False); hivol=set(rv.index[:80]); lovol=set(rv.index[80:])
def analyze(book,label,predfile):
    pr=pd.read_parquet(REPO/predfile)[['symbol','open_time','pred']]; pr['open_time']=pd.to_datetime(pr['open_time'],utc=True)
    D=P[P.symbol.isin(book)].merge(pr,on=['symbol','open_time'],how='inner')
    D['cand']=D.groupby('open_time')['pred'].rank(ascending=False,method='first')
    C=D[D.cand<=10].dropna(subset=['pred','alpha_A','resid_rev']).copy()
    print(f'\n=== {label} long cands (n={len(C)}) — orthIC vs [pred, resid_rev] (truly NEW signal?) ===')
    print('  signal          rawIC     orthIC(vs pred+resid_rev)')
    for col in ['peer_rev','funding_accel','funding_mom','vol_term','autocorr','retmom_3d']:
        s=C.dropna(subset=[col])
        if len(s)<300: print(f'  {col:14s}  (insufficient)'); continue
        raw=s[col].corr(s['alpha_A'],method='spearman')
        X=np.column_stack([s['pred'].values,s['resid_rev'].values,np.ones(len(s))])
        b=np.linalg.lstsq(X,s[col].values,rcond=None)[0]; r=s[col].values-X@b
        o=pd.Series(r).corr(s['alpha_A'].reset_index(drop=True),method='spearman')
        print(f'  {col:14s}  {raw:+.4f}   {o:+.4f}')
analyze(hivol,'A(hivol)','live/state/convexity/hl/fullflow_hl60.parquet')
analyze(lovol,'B(lovol)','live/state/convexity/hl/v0full_hl60.parquet')
