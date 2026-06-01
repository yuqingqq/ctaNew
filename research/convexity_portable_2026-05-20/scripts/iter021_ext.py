import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None
# EXT 2021-26 panel has funding + alpha target but NO cached pred. Test raw funding IC sign-stability across this different universe/era.
p=pd.read_parquet('outputs/vBTC_features/panel_ext2021_v0.parquet')
print('ext syms', p.symbol.nunique(), 'range', p.open_time.min(), p.open_time.max())
print('cols subset', [c for c in p.columns if 'fund' in c or 'alpha' in c or 'return' in c])
p=p.dropna(subset=['funding_rate','alpha_vs_btc_realized','return_pct'])
p['ts']=pd.to_datetime(p['open_time']); t0=p['ts'].min()
p['bar']=((p['ts']-t0).dt.total_seconds()//300).astype(int)
nov=p[p['bar']%48==0].copy()
def xs_ic(df,sig,tgt):
    ics=df.groupby('open_time').apply(lambda x: x[sig].corr(x[tgt],method='spearman') if x[sig].nunique()>3 else np.nan).dropna()
    n=len(ics); return ics.mean(), ics.mean()/(ics.std()/np.sqrt(n)), n, ics
for tgt in ['return_pct','alpha_vs_btc_realized']:
    mu,t,n,_=xs_ic(nov,'funding_rate',tgt); print(f'[EXT 4h] IC(funding_rate -> {tgt}): {mu:+.4f} t {t:+.2f} n {n}')
# yearly sign stability
nov['yr']=nov['ts'].dt.year
print('--- EXT yearly IC(funding->ret) ---')
for y,g in nov.groupby('yr'):
    mu,t,n,_=xs_ic(g,'funding_rate','return_pct'); print(f'{y}: IC {mu:+.4f} t {t:+.2f} n {n}')
