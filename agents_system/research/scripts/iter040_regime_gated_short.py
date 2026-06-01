"""
iter-040 — REGIME-GATED new-listing short. Human lead: short the new-listing fade ONLY in alt-BEAR
(where moonshots are rare and the fade dominates), stand FLAT in alt-BULL (where moonshots cluster
and eat the short).

Mechanism (from iter-037/039): new perps FADE (median -18.7%/30d, 64% down) but a NAKED short's
MEAN ~ 0 because ~8% MOONSHOT 2-5x, and the moonshot FREQUENCY is regime-driven:
  cohort   moonshot-rate  mean-ret30
  2023     21%            +0.24   (alt-bull)
  2024     23%            +0.16   (alt-bull)
  2025      9%            -0.13   (alt-bear)
The short only profits when moonshots are rare. Fix: GATE BY REGIME — at the entry decision read the
forward-knowable alt-index trailing-30d return; short the fade only when alt-bear, else FLAT.

ALT-INDEX (iter-006/007 definition, PIT): equal-weight mean of per-symbol trailing-30d cumulative
return over the SEASONED universe (>=30d history), .shift(1)-lagged so it's known at entry. We
exclude the just-listed name itself from the index (it's new). This is the regime axis.

Stop+gap modeling reused from iter-039 (realistic: fill = max(trigger, breach-bar close)).
"""
import pandas as pd, numpy as np, os, glob
np.seterr(all='ignore')
CACHE = '/home/yuqing/ctaNew/data/ml/cache'
SDIR  = '/home/yuqing/ctaNew/agents_system/research/scripts'
events = pd.read_parquet(f'{SDIR}/iter037_events.parquet')

# ---------- load hourly OHLC for events ----------
def load_hl(sym):
    df = pd.read_parquet(f'{CACHE}/xs_feats_{sym}.parquet', columns=['close','high','low']).dropna(subset=['close'])
    c  = df['close'].resample('1h').last()
    hi = df['high'].resample('1h').max()
    lo = df['low'].resample('1h').min()
    return pd.concat([c.rename('close'), hi.rename('high'), lo.rename('low')], axis=1).dropna()

H = {}
for s in events['sym']:
    d = load_hl(s)
    if len(d) >= 24*32:
        H[s] = d
ev = events[events['sym'].isin(H)].copy().set_index('sym')
print(f'Usable events: {len(H)}  cohorts={ev["year"].value_counts().sort_index().to_dict()}')

# ---------- build PIT alt-index (daily) over the WHOLE universe ----------
# Per-symbol daily close, then trailing-30d cum return, equal-weight across symbols that have >=30d
# of history at that date. Shift(1) so the value at date D uses data through D-1 close.
files = sorted(glob.glob(f'{CACHE}/xs_feats_*.parquet'))
closes = {}
for f in files:
    sym = os.path.basename(f).replace('xs_feats_','').replace('.parquet','')
    c = pd.read_parquet(f, columns=['close'])['close'].dropna()
    if len(c)==0: continue
    closes[sym] = c.resample('1D').last()
px = pd.DataFrame(closes).sort_index()
# trailing-30d return per symbol (only valid where 30d of history exists)
ret30 = px / px.shift(30) - 1.0   # NaN until 30d of history
alt_index_30d = ret30.mean(axis=1, skipna=True).shift(1)   # equal-weight across seasoned syms, PIT-lagged
alt_index_30d = alt_index_30d.dropna()
print(f'alt_index_30d span: {alt_index_30d.index[0].date()} -> {alt_index_30d.index[-1].date()}  '
      f'n={len(alt_index_30d)}  mean={alt_index_30d.mean():+.3f}  '
      f'pct<0={100*(alt_index_30d<0).mean():.0f}%  pct<-0.10={100*(alt_index_30d<-0.10).mean():.0f}%')

# ---------- stop+gap short PnL (iter-039 model) ----------
COST_BPS = 15
def rt_cost(cb=COST_BPS): return 2*cb/1e4
def stopped_short_pnl(d, entry_h, exit_h, stopX=0.30, gap='close'):
    if len(d) <= exit_h: return None
    p0 = d['close'].iloc[entry_h]; trig = p0*(1+stopX)
    seg_hi = d['high'].iloc[entry_h+1:exit_h+1]
    breach = seg_hi[seg_hi >= trig]
    if len(breach)==0:
        px_exit = d['close'].iloc[exit_h]
    else:
        bidx = d.index.get_loc(breach.index[0])
        if gap=='close': fill = max(trig, d['close'].iloc[bidx])
        else:
            nxt = min(bidx+1, len(d)-1); fill = max(trig, d['high'].iloc[nxt])
        px_exit = fill
    return -(px_exit/p0 - 1.0) - rt_cost()
def naked_short_pnl(d, entry_h, exit_h):
    if len(d) <= exit_h: return None
    return -(d['close'].iloc[exit_h]/d['close'].iloc[entry_h]-1.0) - rt_cost()

# ---------- per-event table: entry date, alt-regime at entry, PnL ----------
ENTRY_D, EXIT_D = 3, 30
def alt_at_entry(sym):
    list_date = ev.loc[sym, 'list_date']
    entry_date = (pd.Timestamp(list_date) + pd.Timedelta(days=ENTRY_D)).normalize()
    # most recent alt_index value at or before entry_date (PIT, already shift(1))
    s = alt_index_30d[alt_index_30d.index <= entry_date]
    return (s.iloc[-1], entry_date) if len(s) else (np.nan, entry_date)

rows = []
for s in H:
    a, ed = alt_at_entry(s)
    pnl_naked   = naked_short_pnl(H[s], ENTRY_D*24, EXIT_D*24)
    pnl_stop30  = stopped_short_pnl(H[s], ENTRY_D*24, EXIT_D*24, 0.30, 'close')
    rows.append(dict(sym=s, year=ev.loc[s,'year'], entry_date=ed, alt30=a,
                     pnl_naked=pnl_naked, pnl_stop30=pnl_stop30))
T = pd.DataFrame(rows).dropna(subset=['alt30','pnl_naked'])
print(f'\nEvents with alt-regime + PnL: {len(T)}')

def summ(p, label, seed=0):
    p = np.asarray([x for x in p if x is not None and np.isfinite(x)])
    if len(p)==0:
        print(f'  {label:<46} n=0'); return None
    n=len(p); m=p.mean(); md=np.median(p); hit=(p>0).mean()
    t = m/(p.std()/np.sqrt(n)) if p.std()>0 else 0
    rng=np.random.default_rng(seed)
    boots=np.array([rng.choice(p,n,replace=True).mean() for _ in range(5000)])
    lo,hi=np.percentile(boots,[2.5,97.5]); pgt=(boots>0).mean()
    print(f'  {label:<46} n={n:>3} mean {m:+.4f} med {md:+.4f} hit {hit:4.0%} t {t:+.2f} '
          f'CI[{lo:+.3f},{hi:+.3f}] P(>0)={pgt:.0%} worst {p.min():+.2f}')
    return dict(label=label,n=n,mean=m,med=md,hit=hit,t=t,ci_lo=lo,ci_hi=hi,pgt=pgt)

# ---------- STEP 2: gated vs ungated vs inverse, sweep threshold X ----------
print('\n========== UNGATED (short ALL events) ==========')
summ(T['pnl_naked'].values,  'ungated naked short@3d->30d')
summ(T['pnl_stop30'].values, 'ungated stop+30% short@3d->30d')

for pnlcol, tag in [('pnl_naked','naked'), ('pnl_stop30','stop+30%')]:
    print(f'\n========== REGIME-GATED [{tag}] — short ONLY if alt-bear ==========')
    for X in [0.0, -0.10, -0.20]:
        bear = T[T['alt30'] < X]
        summ(bear[pnlcol].values, f'GATED short if alt30<{X:+.2f}', seed=int(abs(X*100)))
    print(f'---------- INVERSE [{tag}] — short ONLY if alt-BULL (sanity, should be bad) ----------')
    for X in [0.0, -0.10, -0.20]:
        bull = T[T['alt30'] >= X]
        summ(bull[pnlcol].values, f'INVERSE short if alt30>={X:+.2f}', seed=int(abs(X*100))+1)

# ---------- regime composition: how many events bear vs bull at each X ----------
print('\n========== REGIME COMPOSITION (event counts) ==========')
for X in [0.0,-0.10,-0.20]:
    nb=(T['alt30']<X).sum(); print(f'  alt30<{X:+.2f}: bear={nb}  bull={len(T)-nb}')

T.to_parquet(f'{SDIR}/iter040_events_gated.parquet')
print('\nSaved iter040_events_gated.parquet')
