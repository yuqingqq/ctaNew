"""SHORT-side orthogonal-signal hunt (for the four-leg #172 A-short/B-short metas).
For short candidates (bottom-pred per book), which signal predicts SHORT PnL (-fwd alpha), ORTHOGONAL to pred?
Scan resid_rev (recent residual gain = overbought = fade) at N bars + danger (squeeze risk). Read-only, PIT."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
REPO=Path('/home/yuqing/ctaNew'); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
OOS=pd.Timestamp('2025-10-04',tz='UTC')
PAN=pd.read_parquet(tt.PANEL,columns=['symbol','open_time','alpha_vs_btc_realized','rvol_7d','atr_pct','idio_vol_to_btc_1h','corr_to_btc_1d'])
PAN['open_time']=pd.to_datetime(PAN['open_time'],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(['symbol','open_time'])
PAN['alpha_A']=PAN['alpha_vs_btc_realized']
for N in [1,2,3,6,12]:
    PAN[f'prevR_{N}']=PAN.groupby('symbol')['alpha_A'].transform(lambda s:s.shift(1).rolling(N).sum())
P=PAN[PAN.open_time>=OOS].copy()
g=P.groupby('open_time')
for c in ['atr_pct','idio_vol_to_btc_1h','corr_to_btc_1d']: P[f'z_{c}']=g[c].transform(lambda x:(x-x.mean())/x.std())
P['danger']=P['z_atr_pct']+P['z_idio_vol_to_btc_1h']-P['z_corr_to_btc_1d']
rv=P.groupby('symbol')['rvol_7d'].mean().sort_values(ascending=False); hivol=set(rv.index[:80]); lovol=set(rv.index[80:])
def analyze(book,label,predfile):
    pr=pd.read_parquet(REPO/predfile)[['symbol','open_time','pred']]; pr['open_time']=pd.to_datetime(pr['open_time'],utc=True)
    D=P[P.symbol.isin(book)].merge(pr,on=['symbol','open_time'],how='inner')
    D['cand']=D.groupby('open_time')['pred'].rank(ascending=True,method='first')  # bottom pred = short cand
    C=D[D.cand<=10].dropna(subset=['pred','alpha_A']).copy(); C['spnl']=-C['alpha_A']  # short PnL
    print(f'\n=== {label} SHORT candidates (n={len(C)}, short-win {(C.spnl>0).mean():.3f}) — IC with SHORT PnL, orth vs pred ===')
    print('  signal            rawIC     orthIC(vs pred)')
    def orth_ic(col):
        s=C.dropna(subset=[col]); raw=s[col].corr(s['spnl'],method='spearman')
        x=s[[col,'pred']].copy(); x['c']=1; beta=np.linalg.lstsq(x[['pred','c']].values,x[col].values,rcond=None)[0]
        resid=x[col].values-x[['pred','c']].values@beta
        return raw, pd.Series(resid).corr(s['spnl'].reset_index(drop=True),method='spearman')
    for col in ['prevR_1','prevR_2','prevR_3','prevR_6','prevR_12','danger']:
        r,o=orth_ic(col); print(f'  {col:14s}   {r:+.4f}   {o:+.4f}')
analyze(hivol,'A(hivol)','live/state/convexity/hl/fullflow_hl60.parquet')
analyze(lovol,'B(lovol)','live/state/convexity/hl/v0full_hl60.parquet')
