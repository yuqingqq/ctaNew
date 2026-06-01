import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None
pan = pd.read_parquet('outputs/vBTC_features/panel_hl70.parquet')
pred = pd.read_parquet('research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet')
m = pred.merge(pan[['symbol','open_time','funding_rate','alpha_vs_btc_realized']], on=['symbol','open_time'], how='left')
m=m.dropna(subset=['funding_rate','pred','return_pct','alpha_A','alpha_vs_btc_realized'])
m['ts']=pd.to_datetime(m['open_time']); t0=m['ts'].min()
m['bar']=((m['ts']-t0).dt.total_seconds()//300).astype(int)
nov=m[m['bar']%48==0].copy()

def xs_ic(df,sig,tgt):
    ics=df.groupby('open_time').apply(lambda x: x[sig].corr(x[tgt],method='spearman') if x[sig].nunique()>3 else np.nan).dropna()
    n=len(ics); return ics.mean(), ics.mean()/(ics.std()/np.sqrt(n)), n

# vs alpha-residual target (what the book actually trades, beta-neutral)
for tgt in ['alpha_vs_btc_realized']:
    mu,t,n=xs_ic(nov,'funding_rate',tgt); print(f'IC(funding_rate -> {tgt}): {mu:+.4f} t {t:+.2f} n {n}')

# Combined signal orthogonality benefit: does funding add to pred? rank-combine
nov['z_pred']=nov.groupby('open_time')['pred'].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
nov['z_fund']=nov.groupby('open_time')['funding_rate'].transform(lambda x:(x-x.mean())/(x.std()+1e-9))
for w in [0.0,0.25,0.5,0.75,1.0]:
    nov['combo']=(1-w)*nov['z_pred']+w*nov['z_fund']
    mu,t,n=xs_ic(nov,'combo','alpha_vs_btc_realized')
    print(f'combo w_fund={w}: IC->alpha_resid {mu:+.4f} t {t:+.2f}')

# GROSS PnL of a pure funding-tilt L/S book (long high funding, K=5) on alpha-residual return, no cost
K=5
def cyc(df):
    df=df.sort_values('funding_rate')
    lo=df.head(K); hi=df.tail(K)
    # POSITIVE IC: long HIGH funding, short LOW funding
    px=(hi['alpha_vs_btc_realized'].mean()-lo['alpha_vs_btc_realized'].mean())*1e4
    return px
res=nov.groupby('open_time').apply(cyc).dropna()
ann=np.sqrt(6*365)
print(f'GROSS funding-tilt book (alpha-resid, K=5): mean/cyc {res.mean():+.2f}bps total {res.sum():+.0f} Sharpe {res.mean()/res.std()*ann:+.2f}')
# net of 4.5bps/leg *2 legs *2 sides = full turnover assume 100%: cost ~ 4.5*2 bps/cyc on this standalone book
for c in [1,3,4.5]:
    net=res-c*2  # 2 legs each side roundtrip approx; conservative
    print(f'  net @ {c}bps/leg: mean/cyc {net.mean():+.2f} Sharpe {net.mean()/net.std()*ann:+.2f}')
