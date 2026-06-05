"""Rigorous TOD test (user: Go). Is the book-B SHORT-leg-by-entry-hour pattern (strong near funding settles) STABLE
across months, or noise? Cohort attribution (each entry cohort's fwd alpha, independent of sleeve overlap), per-month
consistency, and an hour-shuffle placebo on the q(best)-q(worst) spread. Read-only, PIT."""
import sys; from pathlib import Path
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
REPO=Path('/home/yuqing/ctaNew'); sys.path.insert(0,str(REPO))
import live.train_twobook_models as tt
OOS=pd.Timestamp('2025-10-04',tz='UTC'); ann=np.sqrt(6*365)
P=pd.read_parquet(tt.PANEL,columns=['symbol','open_time','alpha_vs_btc_realized','rvol_7d'])
P['open_time']=pd.to_datetime(P['open_time'],utc=True); P=P[(P.open_time.dt.hour%4==0)&(P.open_time.dt.minute==0)]
P=P[P.open_time>=OOS]; P['alpha_A']=P['alpha_vs_btc_realized']
pr=pd.read_parquet('live/state/convexity/hl/v0full_hl60.parquet')[['symbol','open_time','pred']]
pr['open_time']=pd.to_datetime(pr['open_time'],utc=True)
rv=P.groupby('symbol')['rvol_7d'].mean().sort_values(ascending=False); lov=set(rv.index[80:])
D=P[P.symbol.isin(lov)].merge(pr,on=['symbol','open_time'],how='inner')
D['srank']=D.groupby('open_time')['pred'].rank(method='first')   # low = short
S=D[D.srank<=3].copy(); S['spnl']=-S['alpha_A']*1e4   # short PnL bps
S['hour']=S.open_time.dt.hour; S['month']=S.open_time.dt.to_period('M')
# per-cycle short-leg PnL (mean over the 3 names), then by hour
cyc=S.groupby('open_time').agg(spnl=('spnl','mean'),hour=('hour','first'),month=('month','first')).reset_index()
print('=== book-B SHORT leg: per-cycle PnL by entry hour, overall + per-month consistency ===')
print('hour  overallMean  Sharpe   months_positive(/8)')
for h in [0,4,8,12,16,20]:
    ch=cyc[cyc.hour==h]; mm=ch.groupby('month')['spnl'].mean()
    print(f'  {h:02d}h   {ch.spnl.mean():+7.2f}   {ch.spnl.mean()/ch.spnl.std()*ann:+.2f}    {(mm>0).sum()}/{mm.shape[0]}')
# placebo: is the best-worst hour spread unusual vs random hour labels?
rng=np.random.default_rng(17)
real_spread=cyc.groupby('hour')['spnl'].mean().max()-cyc.groupby('hour')['spnl'].mean().min()
ph=[]
for _ in range(2000):
    cyc['rh']=rng.permutation(cyc['hour'].values)
    g=cyc.groupby('rh')['spnl'].mean(); ph.append(g.max()-g.min())
ph=np.array(ph)
print(f'\nbest-worst hour spread: real {real_spread:.1f} bps  vs placebo mean {ph.mean():.1f}  p95 {np.percentile(ph,95):.1f}  rank={ (ph<real_spread).mean()*100:.0f}%')
