"""
iter-044 — PRICE-CONFIRMATION short (genuinely-different mechanism; no regime data).
Instead of fading the run-up (killed by continued moonshot pumps) or detecting the
macro regime (thin data), SHORT ONLY names that have ALREADY confirmed a breakdown:
first day d in a window where price falls >=THRESH below its running peak-since-listing.
A name that keeps pumping (a moonshot) NEVER triggers → it is auto-excluded WITHOUT
knowing the regime. Hypothesis: this self-filters the bull moonshot tail via the
name's own price action and transports across cohorts.

Entry: scan days d in [DMIN, DMAX]; trigger at first d with close <= (1-THRESH)*max(close[0:d]).
Short from trigger close, hold HOLD days, realized = -(P[d+HOLD]/P[d]-1) - cost.
Names that never trigger in the window = NOT traded.

Look-ahead: trigger uses close[0:d] only; return is forward from d. PIT clean.
Gates: pooled P(mean>0)>=95%, sign-consistent transport across >=2 cohorts,
random-entry-day placebo (matched #traded, entry on a random day in window) >=p95.
"""
import pandas as pd, numpy as np, glob, os
np.seterr(all='ignore')
CACHE='/home/yuqing/ctaNew/data/ml/cache'; COST=4.5e-4

ev=pd.read_parquet('agents_system/research/scripts/iter037_events.parquet')
ev['list_date']=pd.to_datetime(ev['list_date'],utc=True); ev=ev.sort_values('list_date').reset_index(drop=True)
def load(s):
    c=pd.read_parquet(f'{CACHE}/xs_feats_{s}.parquet',columns=['close']).dropna()['close']
    return c.resample('1D').last().dropna()
P={}
for s in ev['sym']:
    try:
        c=load(s)
        if len(c)>=10: P[s]=c.values  # day-indexed array from listing
    except Exception: pass
ev=ev[ev['sym'].isin(P)].reset_index(drop=True)
yr=ev.set_index('sym')['year'].to_dict()

def trigger_day(p,thresh,dmin,dmax):
    pk=p[0]
    for d in range(1,min(dmax+1,len(p))):
        pk=max(pk,p[d])
        if d>=dmin and p[d]<=(1-thresh)*pk:
            return d
    return None

def run(thresh,dmin,dmax,hold,rand=None):
    rng=np.random.default_rng(rand) if rand is not None else None
    rows=[]
    for s,p in P.items():
        if rand is None:
            d=trigger_day(p,thresh,dmin,dmax)
        else:
            # placebo: trade the SAME names that really trigger, but on a random day in window
            dd=trigger_day(p,thresh,dmin,dmax)
            if dd is None: continue
            d=int(rng.integers(dmin,dmax+1))
        if d is None or d+hold>=len(p): continue
        r=-(p[d+hold]/p[d]-1.0)-2*COST
        rows.append((s,yr[s],r))
    return pd.DataFrame(rows,columns=['sym','year','pnl'])

def pgt0(v,nb=2000,sd=0):
    rng=np.random.default_rng(sd); return (np.array([rng.choice(v,len(v),True).mean() for _ in range(nb)])>0).mean()

print('=== iter-044 price-confirmation short ===')
best=None
for thresh in [0.15,0.25,0.35]:
    for hold in [14,21]:
        d=run(thresh,3,21,hold)
        if len(d)<15: print(f'thresh{thresh} hold{hold}: n={len(d)} too few'); continue
        v=d['pnl'].values; mn=v.mean(); sh=mn/v.std()*np.sqrt(365/hold) if v.std()>0 else np.nan
        ys={y:d[d.year==y]['pnl'].mean() for y in [2023,2024,2025] if (d.year==y).sum()>=5}
        cons=len([x for x in ys.values() if x>0])>=2 and len([x for x in ys.values() if x<0])>=2  # mixed?
        signc=(all(x>0 for x in ys.values()) and len(ys)>=2)
        print(f'thresh{thresh} hold{hold}: n={len(d):>3} mean {mn:+.3f} Sh {sh:+.2f} P>0={pgt0(v):.2f} | '+' '.join(f'{y}:{ys[y]:+.2f}' for y in ys)+f' signpos={signc}')
        if signc and (best is None or mn>best[0]): best=(mn,thresh,hold,d)

if best is None:
    print('\nNo sign-consistent-positive config across cohorts → price-confirmation short does NOT transport.')
else:
    mn,thresh,hold,d=best; v=d['pnl'].values
    print(f'\nBEST sign-consistent: thresh{thresh} hold{hold} mean {mn:+.3f} → random-entry-day placebo:')
    ph=[]
    for sd in range(300):
        dp=run(thresh,3,21,hold,rand=sd)
        if len(dp)>=10: ph.append(dp['pnl'].mean())
    ph=np.array(ph); pct=100*(ph<mn).mean()
    print(f'  real mean {mn:+.4f} ranks p{pct:.0f} of random-entry-day (p95 {np.percentile(ph,95):+.4f}, mean {ph.mean():+.4f})')
