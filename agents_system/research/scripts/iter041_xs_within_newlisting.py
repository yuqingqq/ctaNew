"""
iter-041 — CROSS-SECTIONAL rank WITHIN the new-listing cohort (market-neutral).

Motivation: i037/039/040 all hit the regime/thin-data wall because they are
DIRECTIONAL (net-short the cohort) → exposed to the alt-bull moonshot tail and to
"is it a bear regime" (which only ~1 episode populates). A book that is
DOLLAR-NEUTRAL *within* concurrent new listings nets out the common new-listing
beta AND the regime, by construction. If the fade is a CROSS-SECTIONAL phenomenon
(big run-ups fade harder than small ones, relative to each other), this should
survive where the directional bet did not.

Design (PIT, daily grid):
  - early-life window W days from listing; at each day t the "cohort" = names with
    list_date <= t < list_date+W AND >=2 concurrent.
  - SIGNAL(t) = run-up since listing through t (PIT cumulative return). Hypothesis:
    rank within cohort; SHORT high-runup, LONG low-runup; dollar-neutral, demeaned
    weights so sum(w)=0, sum|w|=1.  (also test: signal = momentum_3d, rv_7d.)
  - hold 1 day, forward return t->t+1d per name; book ret = sum w_i * r_i  - cost.
  - cost = 4.5 bps/leg on turnover (weights change daily as cohort/runup move).

Honest gates:
  - bootstrap CI on daily book returns: P(mean>0) >= 95%.
  - TRANSPORT: per-year-cohort Sharpe must be sign-consistent across >=2 years.
  - PLACEBO: random sign assignment within the SAME concurrent set, matched leg
    count, 200 seeds -> real must rank >= p95.
"""
import pandas as pd, numpy as np, glob, os
np.seterr(all='ignore')
CACHE='/home/yuqing/ctaNew/data/ml/cache'
COST=4.5e-4
W_DAYS=45          # early-life window (concurrency probe: 84% days >=2 concurrent)
HOLD_D=1           # rebalance daily
MIN_CONC=2

ev=pd.read_parquet('agents_system/research/scripts/iter037_events.parquet')
ev['list_date']=pd.to_datetime(ev['list_date'],utc=True)
ev=ev.sort_values('list_date').reset_index(drop=True)

# ---- load daily close per event symbol (from listing) ----
def load_daily(sym):
    df=pd.read_parquet(f'{CACHE}/xs_feats_{sym}.parquet',columns=['close']).dropna()
    return df['close'].resample('1D').last().dropna()

prices={}
for s in ev['sym']:
    try:
        p=load_daily(s)
        if len(p)>=5: prices[s]=p
    except Exception: pass
ev=ev[ev['sym'].isin(prices)].reset_index(drop=True)
print(f'events with daily price: {len(ev)}  cohort/yr {ev["year"].value_counts().sort_index().to_dict()}')

# ---- daily decision grid ----
g0=ev['list_date'].min().normalize()
g1=ev['list_date'].max().normalize()+pd.Timedelta(days=W_DAYS)
grid=pd.date_range(g0,g1,freq='1D',tz='UTC')

list_date=ev.set_index('sym')['list_date']

def runup(sym,t):  # PIT cumulative return listing->t
    p=prices[sym]; pp=p[p.index<=t]
    if len(pp)<2: return np.nan
    return pp.iloc[-1]/pp.iloc[0]-1.0

def mom3(sym,t):
    p=prices[sym]; pp=p[p.index<=t]
    if len(pp)<4: return np.nan
    return pp.iloc[-1]/pp.iloc[-4]-1.0

def fwd1(sym,t):   # forward 1d return t->t+1d
    p=prices[sym]; pp=p[p.index>=t]
    if len(pp)<2: return np.nan
    # next-day close vs current (current = last <= t)
    cur=p[p.index<=t]
    if len(cur)<1: return np.nan
    nxt=p[p.index>t]
    if len(nxt)<1: return np.nan
    return nxt.iloc[0]/cur.iloc[-1]-1.0

def cohort_at(t):
    m=(list_date<=t)&(t<list_date+pd.Timedelta(days=W_DAYS))
    return list_date.index[m.values].tolist()

def run_book(signal_fn, sign=+1, seed=None):
    """sign=+1 -> short high-signal/long low-signal (fade). weights demeaned, sum|w|=1.
       seed not None -> RANDOM sign permutation within cohort (placebo)."""
    rng=np.random.default_rng(seed) if seed is not None else None
    rets=[]; dates=[]; prev_w={}
    for t in grid:
        cs=cohort_at(t)
        if len(cs)<MIN_CONC: prev_w={}; continue
        sig=np.array([signal_fn(s,t) for s in cs]); fwd=np.array([fwd1(s,t) for s in cs])
        ok=~(np.isnan(sig)|np.isnan(fwd))
        cs=[c for c,o in zip(cs,ok) if o]; sig=sig[ok]; fwd=fwd[ok]
        if len(cs)<MIN_CONC: prev_w={}; continue
        if rng is not None:
            w=rng.standard_normal(len(cs))      # random cross-section
        else:
            w=-sign*(sig-sig.mean())            # short high signal (fade), demeaned
        if np.all(w==0): prev_w={}; continue
        w=w-w.mean()
        s_abs=np.abs(w).sum()
        if s_abs<1e-12: prev_w={}; continue
        w=w/s_abs
        wd=dict(zip(cs,w))
        gross_ret=float(np.sum(w*fwd))
        # turnover cost vs prev weights
        allk=set(wd)|set(prev_w)
        turn=sum(abs(wd.get(k,0)-prev_w.get(k,0)) for k in allk)
        rets.append(gross_ret-COST*turn); dates.append(t); prev_w=wd
    return pd.Series(rets,index=pd.DatetimeIndex(dates))

def stats(r):
    if len(r)<10: return dict(n=len(r),mean=np.nan,sh=np.nan)
    sh=r.mean()/r.std()*np.sqrt(365/HOLD_D) if r.std()>0 else np.nan
    return dict(n=len(r),mean=r.mean(),sh=sh,tot=r.sum())

def boot_pgt0(r,nb=2000,seed=1):
    rng=np.random.default_rng(seed); v=r.values
    means=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(nb)])
    return (means>0).mean()

print(f'\n=== iter-041 XS-within-newlisting book (W={W_DAYS}d, hold {HOLD_D}d, cost {COST*1e4:.1f}bps/leg) ===')
for name,fn in [('runup_fade',runup),('mom3_fade',mom3)]:
    r=run_book(fn,sign=+1)
    st=stats(r); p=boot_pgt0(r) if len(r)>=10 else np.nan
    print(f'\n[{name}] n={st["n"]} Sharpe {st["sh"]:+.3f} mean/day {st["mean"]*1e4:+.1f}bps tot {st["tot"]:+.3f}  P(mean>0)={p:.3f}')
    # transport by year
    for yr in [2023,2024,2025]:
        ry=r[r.index.year==yr]
        if len(ry)>=10:
            sy=stats(ry); print(f'    {yr}: n={sy["n"]:>3} Sharpe {sy["sh"]:+.3f} mean {sy["mean"]*1e4:+.1f}bps')
    # placebo: random sign within same concurrent set
    if len(r)>=10:
        ph=[]
        for sd in range(200):
            rp=run_book(fn,seed=sd); ph.append(stats(rp)['sh'])
        ph=np.array([x for x in ph if not np.isnan(x)])
        pct=100*(ph<st['sh']).mean()
        print(f'    PLACEBO (random sign, {len(ph)} seeds): real Sharpe {st["sh"]:+.3f} ranks p{pct:.0f}  (placebo mean {ph.mean():+.2f} p95 {np.percentile(ph,95):+.2f})')
