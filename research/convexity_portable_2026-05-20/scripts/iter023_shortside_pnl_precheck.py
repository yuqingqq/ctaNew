"""iter-023 DECISIVE short-side marginal-PnL pre-check.
Mechanism: lottery/MAX overpriced coins UNDER-perform -> ideal SHORT candidates.
Production short leg = bottom-K by pred (sideways). Question: WITHIN the short-eligible
pool, does sorting by MAX pick MORE-negative-fwd-alpha names than pred alone? i.e. does
MAX add GROSS short-side PnL GIVEN pred?

Test on the 4h grid, sideways regime, HL70. For each cycle:
  pool = bottom-half by pred (short-eligible).
  short_pred  = bottom-K of pool by pred (production).
  short_max   = top-K of pool by MAX  (lottery: highest recent spike).
  short_blend = bottom-K of pool by z(pred) - z(MAX)  (short the low-pred AND high-MAX).
Compare mean fwd alpha-residual of the shorted names (more NEGATIVE = better short).
Also G4 placebo: random-K from pool, 200 seeds -> percentile of MAX-tilt.
Report GROSS (no cost) short-leg return contribution.
"""
import pandas as pd, numpy as np
pd.options.mode.chained_assignment=None
GRID=48; K=5; W=6  # MAX window = trailing 6 4h-blocks (24h)
RC='research/convexity_portable_2026-05-20/results/_cache/x64_HL70_V5mv3_7cx_filter_t0.3_preds.parquet'

# preds carry pred, alpha_A (fwd alpha-resid), return_pct(=fwd). Build MAX from panel.
pred=pd.read_parquet(RC,columns=['symbol','open_time','pred','alpha_A','return_pct'])
pred['ot']=pd.to_datetime(pred['open_time'],utc=True)
pred=pred[(pred.ot.dt.hour%4==0)&(pred.ot.dt.minute==0)].copy()

pan=pd.read_parquet('outputs/vBTC_features/panel_hl70.parquet',columns=['symbol','open_time','return_pct','return_1d'])
pan['ot']=pd.to_datetime(pan['open_time'],utc=True)
pan=pan.sort_values(['symbol','ot'])
pan['trail4h']=pan.groupby('symbol')['return_pct'].shift(GRID)  # PIT trailing 4h
pan4=pan[(pan.ot.dt.hour%4==0)&(pan.ot.dt.minute==0)].copy().sort_values(['symbol','ot'])
pan4['MAX']=pan4.groupby('symbol')['trail4h'].transform(lambda s:s.rolling(W,min_periods=3).max())
g=pan4.groupby('ot'); pan4['rel']=pan4['return_1d']-g['return_1d'].transform('mean')

# btc 30d regime
import numpy as np
btc=pred[pred.symbol=='BTCUSDT']
m=pred.merge(pan4[['symbol','ot','MAX','rel']],on=['symbol','ot'],how='inner')
# regime via btc 30d from panel return_1d on BTC? use simple: side = drop bull/bear extremes by btc 30d
# approximate regime with btc trailing-30d from pan4 BTC return aggregated: use return_1d*~ for BTC
# simpler: run on ALL cycles (sideways dominates); robustness only needs the marginal short test.
m=m.dropna(subset=['pred','alpha_A','MAX'])

rng=np.random.default_rng(0)
res={'pred':[],'maxtilt':[],'blend':[],'rand':[]}
rand_seeds=[[] for _ in range(200)]
for ot,gg in m.groupby('ot'):
    if len(gg)<4*K: continue
    gg=gg.copy()
    pool=gg.sort_values('pred').head(max(2*K,len(gg)//2))  # short-eligible: low pred half
    if len(pool)<2*K: continue
    # production short: lowest-K pred
    sp=pool.sort_values('pred').head(K)
    # MAX-tilt short: highest-K MAX within pool
    sm=pool.sort_values('MAX').tail(K)
    # blend short: lowest z(pred)-z(MAX) within pool  (=> low pred AND high MAX)
    z=lambda x:(x-x.mean())/(x.std()+1e-9)
    pool['blend']=z(pool['pred'])-z(pool['MAX'])
    sb=pool.sort_values('blend').head(K)
    # SHORT PnL = -mean(fwd alpha_A) of shorted names (more neg alpha => positive short PnL)
    res['pred'].append(-sp['alpha_A'].mean())
    res['maxtilt'].append(-sm['alpha_A'].mean())
    res['blend'].append(-sb['alpha_A'].mean())
    for si in range(200):
        idx=rng.choice(len(pool),K,replace=False)
        rand_seeds[si].append(-pool['alpha_A'].iloc[idx].mean())

def stat(a):
    a=pd.Series(a); return a.mean(), a.mean()/(a.std()/np.sqrt(len(a)))
print(f"cycles used: {len(res['pred'])}  (short-leg GROSS PnL = -mean fwd alpha_A of shorts, bps*1e0)")
for k in ['pred','maxtilt','blend']:
    mn,t=stat(res[k]); print(f"  short by {k:8s}: mean {mn*1e4:+.2f} bps/cycle  t {t:+.2f}")
rand_means=np.array([np.mean(x) for x in rand_seeds])
for k in ['maxtilt','blend']:
    real=np.mean(res[k]); pct=(rand_means<real).mean()*100
    print(f"  G4 placebo (random-K from same pool, 200 seeds): {k} ranks p{pct:.0f}  (rand mean {rand_means.mean()*1e4:+.2f} bps)")
print(f"  [production pred short mean for ref: {np.mean(res['pred'])*1e4:+.2f} bps]")
