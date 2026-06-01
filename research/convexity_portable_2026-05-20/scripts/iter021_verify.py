import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None
pan = pd.read_parquet('outputs/vBTC_features/panel_hl70.parquet')
pred = pd.read_parquet('research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet')
m = pred.merge(pan[['symbol','open_time','funding_rate','funding_rate_z_7d']], on=['symbol','open_time'], how='left')
m=m.dropna(subset=['funding_rate','pred','return_pct','alpha_A'])

# NON-OVERLAPPING 4h grid: pick every 48th bar (4h = 48*5m). Align to a fixed phase.
m['ts']=pd.to_datetime(m['open_time'])
# bar index per the global 5m grid
t0=m['ts'].min()
m['bar']=((m['ts']-t0).dt.total_seconds()//300).astype(int)
nov = m[m['bar']%48==0].copy()
print('non-overlap rows', len(nov), 'cycles', nov.open_time.nunique())

def xs_ic(df,sig,tgt):
    ics=df.groupby('open_time').apply(lambda x: x[sig].corr(x[tgt],method='spearman') if x[sig].nunique()>3 else np.nan).dropna()
    n=len(ics); t=ics.mean()/(ics.std()/np.sqrt(n)) if n>1 else np.nan
    return ics.mean(), t, n

for sig in ['funding_rate','funding_rate_z_7d']:
    mu,t,n=xs_ic(nov,sig,'return_pct')
    print(f'[4h-nonoverlap] IC({sig}->ret): {mu:+.4f} t {t:+.2f} n {n}')
mu,t,n=xs_ic(nov,'pred','alpha_A'); print(f'[4h-nonoverlap] IC(pred->alpha_A): {mu:+.4f} t {t:+.2f} n {n}')

# Per-fold sign stability of funding_rate IC (use fold column on full overlap grid is fine for sign)
print('--- per-fold IC(funding_rate->return_pct), non-overlap ---')
for f,g in nov.groupby('fold'):
    mu,t,n=xs_ic(g,'funding_rate','return_pct')
    print(f'fold {f}: IC {mu:+.4f} t {t:+.2f} n {n}')
