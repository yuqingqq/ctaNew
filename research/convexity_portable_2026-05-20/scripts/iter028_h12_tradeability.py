"""
iter-028 STEP-3 tradeability pre-check for the best NON-4h transport-stable cell.

From the heatmap, the only transport-stable predictability at ANY horizon is
short-horizon REVERSAL, and its IC DECREASES with horizon (4h 0.032 -> 12h 0.030
-> 1d 0.028 -> 3d 0.019 -> 1w 0.007). The 4h-residual cell is the PEAK.

The one angle the orchestrator asked to check: is a LONGER horizon TRADEABLE with a
better cost profile? The best longer cell is 12h reversal (min|IC| 0.0295, close to 4h).
A 12h hold has ~3x lower turnover than 4h -> if its gross edge survives, cost helps.

So: build a simple rank-K reversal long-short at 12h and 1d horizons (rank by -trail1d),
non-overlapping, and report GROSS per-cycle long-short spread + transport (HL70 & EXT)
+ a matched-random-timing/count G4 placebo. Compare to the 4h-reversal which iter-022
ALREADY rejected (gross collapses pre-cost). If 12h gross is no better, NO-CANDIDATE.
"""
import pandas as pd, numpy as np, pyarrow.parquet as pq
pd.options.mode.chained_assignment=None
rng=np.random.default_rng(11)

GRID=48; K=5; NSEED=200
PANELS={'HL70':'outputs/vBTC_features/panel_hl70.parquet',
        'EXT':'outputs/vBTC_features/panel_ext2021_v0.parquet'}
H_BLOCKS={'4h':1,'12h':3,'1d':6}

def build(path):
    p=pd.read_parquet(path,columns=['symbol','open_time','return_pct'])
    p['ot']=pd.to_datetime(p['open_time']); p=p.sort_values(['symbol','open_time'])
    uts=np.sort(p['ot'].unique()); grid=pd.Index(uts[::GRID])
    g=p[p['ot'].isin(grid)].copy().sort_values(['symbol','ot'])
    grp=g.groupby('symbol')
    g['trail1d']=grp['return_pct'].transform(lambda s:s.shift(1).rolling(6,min_periods=4).sum())
    g['rev']=-g['trail1d']
    for hn,hb in H_BLOCKS.items():
        if hb==1: g[f'fwd_{hn}']=g['return_pct']
        else:
            g[f'fwd_{hn}']=grp['return_pct'].transform(
                lambda s:s.rolling(hb,min_periods=hb).sum().shift(-(hb-1)))
    return g

def rankK_ls(g, hname):
    """Rank by rev, long top-K short bottom-K, GROSS fwd spread per ENTRY cycle.
    For horizon H, only enter every H blocks (non-overlapping) to mimic an H-hold book."""
    hb=H_BLOCKS[hname]
    fwd=f'fwd_{hname}'
    d=g.dropna(subset=['rev',fwd]).copy()
    cyc=np.sort(d['ot'].unique())[::hb]      # non-overlapping entries at horizon spacing
    real=[]; turn=[]
    rand=[[] for _ in range(NSEED)]
    prev_long=set(); prev_short=set()
    for ot in cyc:
        c=d[d['ot']==ot]
        if len(c)<2*K: continue
        c=c.sort_values('rev')
        short=c.iloc[:K]; long=c.iloc[-K:]
        real.append(long[fwd].mean()-short[fwd].mean())
        lset=set(long.symbol); sset=set(short.symbol)
        turn.append((len(lset-prev_long)+len(sset-prev_short))/(2*K))
        prev_long,prev_short=lset,sset
        syms=c.symbol.values; fv=c[fwd].values
        for s in range(NSEED):
            idx=rng.permutation(len(c))
            li=idx[-K:]; si=idx[:K]
            rand[s].append(fv[li].mean()-fv[si].mean())
    real=np.array(real); randm=np.array([np.mean(x) for x in rand])
    pct=(randm<real.mean()).mean()*100
    ann=real.mean()/real.std()*np.sqrt(252* (24/(4*hb)) ) if real.std()>0 else np.nan
    return dict(gross=real.mean()*1e4, sharpe=ann, turn=np.mean(turn), pct=pct, n=len(real))

print("="*92)
print("12h/1d REVERSAL rank-K long-short: GROSS spread, turnover, G4 placebo, transport")
print(f"  K={K} (rank by -trailing-1d), non-overlapping entries at horizon spacing")
print("="*92)
print(f"{'panel':<6}{'hor':<5}{'gross_bps':>10}{'sharpe':>8}{'turnover':>9}{'G4_rank':>8}{'n_cyc':>7}")
for nm,path in PANELS.items():
    g=build(path)
    for hn in ['4h','12h','1d']:
        r=rankK_ls(g,hn)
        print(f"{nm:<6}{hn:<5}{r['gross']:>+10.2f}{r['sharpe']:>+8.2f}{r['turn']:>9.2f}"
              f"{('p%d'%r['pct']):>8}{r['n']:>7}")
print()
print("READING: longer horizon helps cost only if GROSS spread stays positive AND transports.")
print("If 12h gross <= 4h gross and/or G4 < p95, the longer-horizon reversal is NOT a tradeable")
print("new product -- it is iter-022's reversal that already collapses pre-cost. NO-CANDIDATE.")
print("DONE")
