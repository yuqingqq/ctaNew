"""
iter-042 — broaden the cross-sectional-within-new-listing scan beyond runup/mom3.
Same dollar-neutral within-cohort book as iter-041. Test more PIT early-life features
as the ranking signal: realized vol (rv_7d), volume-decay (recent vs early volume),
short-term reversal (ret_1d), and run-up*vol interaction. GATE: a feature is only
worth the expensive placebo if its book Sharpe is sign-CONSISTENT across >=2 cohort
years AND positive overall (iter-041 killer was sign-flip + placebo p50).
"""
import pandas as pd, numpy as np, glob, os
np.seterr(all='ignore')
CACHE='/home/yuqing/ctaNew/data/ml/cache'
COST=4.5e-4; W_DAYS=45; MIN_CONC=2

ev=pd.read_parquet('agents_system/research/scripts/iter037_events.parquet')
ev['list_date']=pd.to_datetime(ev['list_date'],utc=True)
ev=ev.sort_values('list_date').reset_index(drop=True)

def load(sym):
    df=pd.read_parquet(f'{CACHE}/xs_feats_{sym}.parquet',columns=['close','volume']).dropna(subset=['close'])
    c=df['close'].resample('1D').last().dropna()
    v=df['volume'].resample('1D').sum().reindex(c.index).fillna(0.0)
    return c,v

P={}; V={}
for s in ev['sym']:
    try:
        c,v=load(s)
        if len(c)>=8: P[s]=c; V[s]=v
    except Exception: pass
ev=ev[ev['sym'].isin(P)].reset_index(drop=True)
list_date=ev.set_index('sym')['list_date']

def _upto(p,t): return p[p.index<=t]
def runup(s,t):
    pp=_upto(P[s],t);  return np.nan if len(pp)<2 else pp.iloc[-1]/pp.iloc[0]-1
def rv7(s,t):
    pp=_upto(P[s],t);  r=pp.pct_change().dropna(); return np.nan if len(r)<4 else r.tail(7).std()
def vol_decay(s,t):   # recent-3d avg vol / first-3d avg vol  (high = still hot)
    vv=V[s][V[s].index<=t]
    if len(vv)<6: return np.nan
    e=vv.head(3).mean();  return np.nan if e<=0 else vv.tail(3).mean()/e
def rev1(s,t):
    pp=_upto(P[s],t);  return np.nan if len(pp)<2 else pp.iloc[-1]/pp.iloc[-2]-1
def runup_x_rv(s,t):
    a,b=runup(s,t),rv7(s,t); return np.nan if (np.isnan(a)or np.isnan(b)) else a*b

FEATS={'rv7_fade':rv7,'voldecay_fade':vol_decay,'rev1_fade':rev1,'runupXrv_fade':runup_x_rv}

def fwd1(s,t):
    p=P[s]; cur=p[p.index<=t]; nxt=p[p.index>t]
    return np.nan if (len(cur)<1 or len(nxt)<1) else nxt.iloc[0]/cur.iloc[-1]-1
def cohort_at(t):
    m=(list_date<=t)&(t<list_date+pd.Timedelta(days=W_DAYS)); return list_date.index[m.values].tolist()

g0=ev['list_date'].min().normalize(); g1=ev['list_date'].max().normalize()+pd.Timedelta(days=W_DAYS)
grid=pd.date_range(g0,g1,freq='1D',tz='UTC')

def run_book(fn,seed=None):
    rng=np.random.default_rng(seed) if seed is not None else None
    rets=[]; dates=[]; prev={}
    for t in grid:
        cs=cohort_at(t)
        if len(cs)<MIN_CONC: prev={}; continue
        sig=np.array([fn(s,t) for s in cs]); fwd=np.array([fwd1(s,t) for s in cs])
        ok=~(np.isnan(sig)|np.isnan(fwd)); cs=[c for c,o in zip(cs,ok) if o]; sig=sig[ok]; fwd=fwd[ok]
        if len(cs)<MIN_CONC: prev={}; continue
        w=rng.standard_normal(len(cs)) if rng is not None else -(sig-sig.mean())
        w=w-w.mean(); sa=np.abs(w).sum()
        if sa<1e-12: prev={}; continue
        w=w/sa; wd=dict(zip(cs,w))
        gr=float(np.sum(w*fwd)); allk=set(wd)|set(prev)
        turn=sum(abs(wd.get(k,0)-prev.get(k,0)) for k in allk)
        rets.append(gr-COST*turn); dates.append(t); prev=wd
    return pd.Series(rets,index=pd.DatetimeIndex(dates))

def sh(r): return r.mean()/r.std()*np.sqrt(365) if (len(r)>2 and r.std()>0) else np.nan

print(f'=== iter-042 XS feature scan (neutral within-cohort book, W={W_DAYS}d) ===')
promising=[]
for name,fn in FEATS.items():
    r=run_book(fn); S=sh(r)
    ys={yr:sh(r[r.index.year==yr]) for yr in [2023,2024,2025]}
    yv=[v for v in ys.values() if not np.isnan(v)]
    consistent=len(yv)>=2 and (all(v>0 for v in yv) or all(v<0 for v in yv))
    print(f'[{name}] overall Sharpe {S:+.3f} | per-yr '+' '.join(f'{y}:{ys[y]:+.2f}' for y in ys)+f' | sign-consistent={consistent}')
    if consistent and S>0: promising.append((name,fn,S))

if not promising:
    print('\nNO feature is sign-consistent-positive across cohorts → cross-sectional angle CLOSED (no placebo needed).')
else:
    print('\nPromising (sign-consistent + positive) → running placebo:')
    for name,fn,S in promising:
        ph=np.array([x for x in (sh(run_book(fn,seed=sd)) for sd in range(150)) if not np.isnan(x)])
        pct=100*(ph<S).mean()
        print(f'  [{name}] real {S:+.3f} ranks p{pct:.0f} (placebo p95 {np.percentile(ph,95):+.2f})')
