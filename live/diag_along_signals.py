"""Find the orthogonal signal for the high-vol long (root cause hunt, 2026-06-03).
Hypothesis: model longs on 1d/3d mean-rev; high-vol names need a SHORT-horizon BTC-residual reversal.
Scan prev-Nh residual-alpha reversal at N in {1,2,3,6,12} bars; measure IC with forward alpha, RAW and
ORTHOGONALIZED vs the model pred (the marginal new info). Compare high-vol vs low-vol book. Read-only, PIT."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
REPO=Path('/home/yuqing/ctaNew'); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
ann=np.sqrt(6*365); OOS=pd.Timestamp('2025-10-04',tz='UTC')
PAN=pd.read_parquet(tt.PANEL,columns=['symbol','open_time','alpha_vs_btc_realized','rvol_7d'])
PAN['open_time']=pd.to_datetime(PAN['open_time'],utc=True)
PAN=PAN[(PAN.open_time.dt.hour%4==0)&(PAN.open_time.dt.minute==0)].sort_values(['symbol','open_time'])
PAN['alpha_A']=PAN['alpha_vs_btc_realized']
for N in [1,2,3,6,12]:
    PAN[f'prevR_{N}']=PAN.groupby('symbol')['alpha_A'].transform(lambda s:s.shift(1).rolling(N).sum())
P=PAN[PAN.open_time>=OOS].copy()
rv=P.groupby('symbol')['rvol_7d'].mean().sort_values(ascending=False); hivol=set(rv.index[:80]); lovol=set(rv.index[80:])
def analyze(book,label,predfile):
    pr=pd.read_parquet(REPO/predfile)[['symbol','open_time','pred']]; pr['open_time']=pd.to_datetime(pr['open_time'],utc=True)
    D=P[P.symbol.isin(book)].merge(pr,on=['symbol','open_time'],how='inner')
    D['cand']=D.groupby('open_time')['pred'].rank(ascending=False,method='first')
    C=D[D.cand<=10].dropna(subset=['pred','alpha_A']).copy()   # long candidate pool
    print(f'\n=== {label} long candidates (n={len(C)}) — IC of prev-Nh residual reversal with FORWARD alpha ===')
    print('  N(bars/hrs)  rawIC      orthIC(vs pred)   meanA(prevR<0)  meanA(prevR>=0)')
    for N in [1,2,3,6,12]:
        col=f'prevR_{N}'; s=C.dropna(subset=[col])
        raw=s[col].corr(s['alpha_A'],method='spearman')
        # orthogonalize prevR vs pred (residual), then IC with fwd alpha = marginal info
        x=s[[col,'pred']].copy(); x['c']=1
        beta=np.linalg.lstsq(x[['pred','c']].values, x[col].values, rcond=None)[0]
        resid=x[col].values - x[['pred','c']].values@beta
        orth=pd.Series(resid).corr(s['alpha_A'].reset_index(drop=True),method='spearman')
        m0=s[s[col]<0]['alpha_A'].mean()*1e4; m1=s[s[col]>=0]['alpha_A'].mean()*1e4
        print(f'  {N:2d} ({N*4:2d}h)     {raw:+.4f}    {orth:+.4f}          {m0:+6.2f}bps      {m1:+6.2f}bps')
analyze(hivol,'HIGH-VOL A','live/state/convexity/hl_wfund175/fullflow_hl60.parquet')
analyze(lovol,'LOW-VOL  B','live/state/convexity/hl_wfund175/v0full_hl60.parquet')
