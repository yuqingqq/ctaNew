"""Is there orthogonal alpha BEYOND resid_rev? Scan additional PIT candidate signals on book-B long candidates,
orthogonalized vs the resid_rev-MODEL pred (hl_residrev) — i.e. genuinely NEW beyond what iter13 already captures.
If a signal has orthIC>~0.04 vs the residrev pred, it extends the win; else iter13 is the ceiling. Read-only, PIT."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
REPO=Path('/home/yuqing/ctaNew'); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
OOS=pd.Timestamp('2025-10-04',tz='UTC')
cols=['symbol','open_time','alpha_vs_btc_realized','rvol_7d','funding_rate','funding_rate_z_7d',
      'funding_rate_1d_change','atr_pct','idio_vol_to_btc_1h','corr_to_btc_1d','return_1d','ret_3d','bars_since_high']
PAN=pd.read_parquet(tt.PANEL,columns=cols); PAN['open_time']=pd.to_datetime(PAN['open_time'],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(['symbol','open_time'])
PAN['alpha_A']=PAN['alpha_vs_btc_realized']
a=PAN.groupby('symbol')['alpha_A']
PAN['rr2']=-a.transform(lambda s:s.shift(1).rolling(2).sum())
PAN['rr6']=-a.transform(lambda s:s.shift(1).rolling(6).sum())
PAN['rr12']=-a.transform(lambda s:s.shift(1).rolling(12).sum())
# CANDIDATE NEW signals (PIT):
PAN['resid_accel']=PAN['rr2']-PAN['rr6']/3.0        # recent vs older residual (acceleration of reversal)
PAN['resid_vol']=a.transform(lambda s:s.shift(1).rolling(6).std())   # residual volatility (dispersion of recent alpha)
PAN['funding_rev']=-PAN.groupby('symbol')['funding_rate_z_7d'].transform(lambda s:s.shift(1))  # fade funding extreme
PAN['rvol_chg']=PAN.groupby('symbol')['rvol_7d'].transform(lambda s:s.pct_change(6).shift(1)).clip(-2,2)  # vol regime shift
PAN['ret_accel']=PAN.groupby('symbol')['return_1d'].transform(lambda s:s.shift(1))-PAN.groupby('symbol')['ret_3d'].transform(lambda s:s.shift(1))/3.0
P=PAN[PAN.open_time>=OOS].copy()
# book B (low-vol) long candidates ranked by resid_rev MODEL pred
rv=P.groupby('symbol')['rvol_7d'].mean().sort_values(ascending=False); lovol=set(rv.index[80:])
pr=pd.read_parquet(REPO/'live/state/convexity/hl_residrev/v0full_hl60.parquet')[['symbol','open_time','pred']].rename(columns={'pred':'rrpred'})
pr['open_time']=pd.to_datetime(pr['open_time'],utc=True)
D=P[P.symbol.isin(lovol)].merge(pr,on=['symbol','open_time'],how='inner')
D['cand']=D.groupby('open_time')['rrpred'].rank(ascending=False,method='first')
C=D[D.cand<=10].dropna(subset=['rrpred','alpha_A']).copy()
print(f'book-B long cands by resid_rev-model pred (n={len(C)}) — orthIC vs rrpred (NEW alpha beyond iter13?)')
print('  signal          rawIC     orthIC(vs rrpred)')
def oic(col):
    s=C.dropna(subset=[col]);
    if len(s)<300: return np.nan,np.nan
    raw=s[col].corr(s['alpha_A'],method='spearman')
    x=s[[col,'rrpred']].copy();x['c']=1;b=np.linalg.lstsq(x[['rrpred','c']].values,x[col].values,rcond=None)[0]
    r=x[col].values-x[['rrpred','c']].values@b
    return raw,pd.Series(r).corr(s['alpha_A'].reset_index(drop=True),method='spearman')
for col in ['resid_accel','resid_vol','funding_rev','rvol_chg','ret_accel','rr12','bars_since_high']:
    r,o=oic(col); print(f'  {col:14s}  {r:+.4f}   {o:+.4f}')
