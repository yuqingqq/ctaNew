"""
iter-027 decisive construction-layer marginal pre-check (the iter-022/023 killer).

For each candidate signal, build the production-style pred-conditioned pool
(top-K_pool / bottom-K_pool by pred), then ask: does tilting the FINAL K=5 pick
by the candidate beat a matched-RANDOM K pick from the SAME pool (>=p95)?
This is the layer where rel_ret_1d (i22), funding (i21), MAX (i23) all died.

We test on HL70 (has pred). Candidate signals:
  - rev_6  : iter-022 reversal in disguise (calibration: EXPECT FAIL ~p<95)
  - vov    : NEW vol-of-vol signal (the only new transport-stable family)
  - vov_resid : vov residualized on rev_6 + |mom_6| (strip vol/reversal it proxies)

Metric: mean forward alpha_A of the signal-tilted long-short pick vs the
distribution of 200 random picks from the same pool. Percentile rank reported.
"""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment = None
rng = np.random.default_rng(7)

GRID=48; K_POOL=15; K=5; NSEED=200
PRED=('research/convexity_portable_2026-05-20/results/_cache/'
      'x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet')

p = pd.read_parquet('outputs/vBTC_features/panel_hl70.parquet',
                    columns=['symbol','open_time','return_pct','alpha_vs_btc_realized'])
p['ot']=pd.to_datetime(p['open_time']).astype('datetime64[ns, UTC]')
p=p.sort_values(['symbol','open_time'])
uts=np.sort(p['ot'].unique()); grid=set(uts[::GRID])
p['trail4h']=p.groupby('symbol')['return_pct'].shift(GRID)
g=p[p['ot'].isin(grid)].copy().sort_values(['symbol','ot'])
grp=g.groupby('symbol')
g['mom_6']=grp['trail4h'].transform(lambda s:s.rolling(6,min_periods=4).sum())
g['rev_6']=-g['mom_6']
g['rv_6']=grp['trail4h'].transform(lambda s:s.rolling(6,min_periods=4).std())
g['vov']=grp['rv_6'].transform(lambda s:s.rolling(18,min_periods=12).std())

pred=pd.read_parquet(PRED, columns=['symbol','open_time','pred','alpha_A'])
pred['ot']=pd.to_datetime(pred['open_time']).astype('datetime64[ns, UTC]')
mm=g.merge(pred[['symbol','ot','pred','alpha_A']],on=['symbol','ot'],how='inner')

# vov residualized on rev_6 and |mom_6| per cycle (strip reversal/vol proxy)
def resid_on(df, y, xs):
    d=df.dropna(subset=[y]+xs).copy()
    out=[]
    for ot,c in d.groupby('ot'):
        X=c[xs].values; X=np.column_stack([np.ones(len(c)),X])
        yv=c[y].values
        try:
            beta,_,_,_=np.linalg.lstsq(X,yv,rcond=None)
            c=c.copy(); c[y+'_resid']=yv-X@beta
        except Exception:
            c=c.copy(); c[y+'_resid']=np.nan
        out.append(c)
    return pd.concat(out)
mm=resid_on(mm,'vov',['rev_6','mom_6'])

def construction_marginal(df, sig, pred_col='pred', tgt='alpha_A', short_side=True):
    """Within bottom-K_POOL by pred (the SHORT pool) and top-K_POOL (LONG pool),
    tilt by sig; compare to random-K from same pool. Long-short spread metric."""
    d=df.dropna(subset=[sig,pred_col,tgt]).copy()
    real_ls=[]; rand_ls=[[] for _ in range(NSEED)]
    for ot,c in d.groupby('ot'):
        if len(c)<2*K_POOL: continue
        c=c.sort_values(pred_col)
        short_pool=c.iloc[:K_POOL]; long_pool=c.iloc[-K_POOL:]
        # production short = lowest pred; long = highest pred. tilt within pool by sig.
        # signal: more-negative sig -> better long (sign per transport: vov NEG IC -> low vov = long)
        long_pick = long_pool.nsmallest(K, sig)   # low-sig within long pool
        short_pick= short_pool.nlargest(K, sig)   # high-sig within short pool
        real_ls.append(long_pick[tgt].mean() - short_pick[tgt].mean())
        for s in range(NSEED):
            li=rng.choice(len(long_pool),K,replace=False)
            si=rng.choice(len(short_pool),K,replace=False)
            rand_ls[s].append(long_pool[tgt].values[li].mean()-short_pool[tgt].values[si].mean())
    real=np.mean(real_ls)
    rand=np.array([np.mean(x) for x in rand_ls])
    pct=(rand<real).mean()*100
    return real, rand.mean(), pct, len(real_ls)

print("="*78)
print("CONSTRUCTION-LAYER MARGINAL (HL70): tilt within pred-pool vs random-from-pool")
print(f"  K_POOL={K_POOL} K={K} seeds={NSEED}")
print("="*78)
for sig in ['rev_6','vov','vov_resid']:
    real,randm,pct,n=construction_marginal(mm,sig)
    flag='PASS' if pct>=95 else 'fail'
    print(f"  {sig:<12} real_LS={real:+.5f}  rand_mean={randm:+.5f}  rank=p{pct:.0f}  ({flag})  n={n}")
print("\n  (rev_6 EXPECTED to fail ~iter-022; vov is the new test. Need >=p95 to survive.)")
