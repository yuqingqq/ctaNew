"""
iter-039 (A) — RISK-MANAGED fade-short with HARD STOP + realistic GAP SLIPPAGE.

Mechanism under test: iter-037 found new perps FADE (median -19%/30d, 64% down) but a NAKED short
has mean ~0 because ~8% moonshot 2-5x; one 5x = -500% on a short eats ~100 winning shorts.
Idea: cap the loss with a hard stop (cut short if price rallies +X% from entry).
HONESTY: new listings GAP/pump violently -> a stop does NOT fill at the trigger. We model the fill
at a WORSE level than the trigger using the actual bar path:
  - We walk the hourly path. The first bar whose HIGH >= entry*(1+X) triggers the stop.
  - Fill = max(trigger_level, that_bar_CLOSE) i.e. assume we fill no better than the bar close once
    triggered (the bar that breached can close far above the trigger -> realistic gap-through).
  - Also test an even-worse assumption: fill at the NEXT bar's high (pump continues).
Compare stopped-short vs naked-short across X in {30,50,100%}, two gap models, realistic cost.
"""
import pandas as pd, numpy as np, os
np.seterr(all='ignore')
CACHE = '/home/yuqing/ctaNew/data/ml/cache'
events = pd.read_parquet('/home/yuqing/ctaNew/agents_system/research/scripts/iter039_events.parquet') \
    if os.path.exists('/home/yuqing/ctaNew/agents_system/research/scripts/iter039_events.parquet') \
    else pd.read_parquet('/home/yuqing/ctaNew/agents_system/research/scripts/iter037_events.parquet')

def load_hl(sym):
    df = pd.read_parquet(f'{CACHE}/xs_feats_{sym}.parquet', columns=['close','high','low']).dropna(subset=['close'])
    c = df['close'].resample('1h').last()
    hi = df['high'].resample('1h').max()
    lo = df['low'].resample('1h').min()
    out = pd.concat([c.rename('close'), hi.rename('high'), lo.rename('low')], axis=1).dropna()
    return out

H = {}
for s in events['sym']:
    d = load_hl(s)
    if len(d) >= 24*32:
        H[s] = d
ev = events[events['sym'].isin(H)].copy().set_index('sym')
print(f'Usable events: {len(H)}  cohorts={ev["year"].value_counts().sort_index().to_dict()}')

COST_BPS = 15  # per leg, thin new perps (RT = 2x)
def rt_cost(cb=COST_BPS): return 2*cb/1e4

def stopped_short_pnl(d, entry_h, exit_h, stopX, gap='close'):
    """Short at entry_h. Hold to exit_h. If at any bar in (entry,exit] high>=entry*(1+stopX), stop out.
    Fill modeling (worse-than-trigger):
      gap='close': fill = max(trigger_level, breach_bar_close)
      gap='next':  fill = next bar's HIGH (assume pump continues 1 more bar)  -- harsher
    Returns short PnL (price fall = profit), net of RT cost. None if not enough data."""
    if len(d) <= exit_h: return None
    p0 = d['close'].iloc[entry_h]
    trig = p0 * (1+stopX)
    seg_hi = d['high'].iloc[entry_h+1:exit_h+1]
    breach = seg_hi[seg_hi >= trig]
    if len(breach) == 0:
        # no stop: exit at horizon close
        px_exit = d['close'].iloc[exit_h]
    else:
        bidx = d.index.get_loc(breach.index[0])
        if gap == 'close':
            fill = max(trig, d['close'].iloc[bidx])
        else:  # 'next'
            nxt = min(bidx+1, len(d)-1)
            fill = max(trig, d['high'].iloc[nxt])
        px_exit = fill
    raw = px_exit / p0 - 1.0   # price change
    short_raw = -raw           # short profits when price falls
    return short_raw - rt_cost()

def naked_short_pnl(d, entry_h, exit_h):
    if len(d) <= exit_h: return None
    raw = d['close'].iloc[exit_h]/d['close'].iloc[entry_h] - 1.0
    return -raw - rt_cost()

def stats(pnls, label, seed=0):
    p = np.array([x for x in pnls if x is not None and np.isfinite(x)])
    n=len(p); m=p.mean(); md=np.median(p); hit=(p>0).mean()
    t = m/(p.std()/np.sqrt(n)) if p.std()>0 else 0
    rng=np.random.default_rng(seed)
    boots=np.array([rng.choice(p,n,replace=True).mean() for _ in range(3000)])
    lo,hi=np.percentile(boots,[2.5,97.5]); pgt=(boots>0).mean()
    print(f'  {label:<42} n={n:>3} mean {m:+.4f} med {md:+.4f} hit {hit:4.0%} t {t:+.2f} '
          f'CI[{lo:+.3f},{hi:+.3f}] P(>0)={pgt:.0%} worst {p.min():+.2f}')
    return dict(label=label,n=n,mean=m,med=md,hit=hit,t=t,ci_lo=lo,ci_hi=hi,pgt=pgt,pnls=p)

# entry day3, hold to day14 and day30 (matching iter-037 best windows)
print('\n=== NAKED SHORT (baseline, no stop) ===')
for W,Hh in [(3,14),(3,30),(7,30)]:
    stats([naked_short_pnl(H[s],W*24,Hh*24) for s in H], f'naked short@{W}d->{Hh}d')

for gap in ['close','next']:
    print(f'\n=== STOPPED SHORT (gap model = {gap}) ===')
    for W,Hh in [(3,14),(3,30),(7,30)]:
        for X in [0.30,0.50,1.00]:
            stats([stopped_short_pnl(H[s],W*24,Hh*24,X,gap) for s in H],
                  f'short@{W}d->{Hh}d stop+{X:.0%} [{gap}]')

# ---- COHORT TRANSPORT for the best-looking stopped config ----
print('\n=== COHORT TRANSPORT — stopped short@3d->30d stop+50% [close] ===')
W,Hh,X,gap = 3,30,0.50,'close'
for yr in [2023,2024,2025]:
    syms=[s for s in ev[ev['year']==yr].index if s in H]
    stats([stopped_short_pnl(H[s],W*24,Hh*24,X,gap) for s in syms], f'  {yr} (n={len(syms)})', seed=yr)

print('\n=== COHORT TRANSPORT — stopped short@3d->30d stop+30% [close] ===')
X=0.30
for yr in [2023,2024,2025]:
    syms=[s for s in ev[ev['year']==yr].index if s in H]
    stats([stopped_short_pnl(H[s],W*24,Hh*24,X,gap) for s in syms], f'  {yr} (n={len(syms)})', seed=yr)

# ---- COST SWEEP for best stopped config ----
print('\n=== COST SWEEP — stopped short@3d->30d stop+50% [close] ===')
X=0.50
for cb in [5,10,15,20,30]:
    p=np.array([x for x in [stopped_short_pnl(H[s],W*24,Hh*24,X,gap) for s in H] if x is not None])
    # re-add the cost difference (function used 15): adjust
    adj = p + rt_cost(15) - rt_cost(cb)
    print(f'  cost {cb:>2}bps/leg: mean {adj.mean():+.4f} hit {(adj>0).mean():4.0%} t {adj.mean()/(adj.std()/np.sqrt(len(adj))):+.2f}')
