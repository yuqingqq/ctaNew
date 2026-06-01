"""
iter-039 placebo / final validation:
1. MODEL-GATE PLACEBO (G4): does skipping the model's top-decile predicted-moonshots beat skipping a
   RANDOM same-count subset? If random skip does as well, the 'model' adds nothing.
2. Moonshot base-rate by cohort (is the moonshot phenomenon even stationary?).
3. Combined verdict: best honest config = stopped short, gated. Pooled event-bootstrap with cohort
   sign-flip flagged.
"""
import pandas as pd, numpy as np, os
np.seterr(all='ignore')
CACHE='/home/yuqing/ctaNew/data/ml/cache'
df=pd.read_parquet('/home/yuqing/ctaNew/agents_system/research/scripts/iter039_model_feats.parquet')

def load_hl(sym):
    d=pd.read_parquet(f'{CACHE}/xs_feats_{sym}.parquet',columns=['close','high','low']).dropna(subset=['close'])
    c=d['close'].resample('1h').last(); hi=d['high'].resample('1h').max(); lo=d['low'].resample('1h').min()
    return pd.concat([c.rename('close'),hi.rename('high'),lo.rename('low')],axis=1).dropna()
COST=2*15/1e4
def short_stop_pnl(d, X=0.50, W=72, Hh=720):
    if len(d)<=Hh: return None
    p0=d['close'].iloc[W]; trig=p0*(1+X); seg=d['high'].iloc[W+1:Hh+1]; br=seg[seg>=trig]
    if len(br)==0: px=d['close'].iloc[Hh]
    else:
        bi=d.index.get_loc(br.index[0]); px=max(trig,d['close'].iloc[bi])
    return -(px/p0-1.0)-COST

# precompute pnl per sym
PNL={s:short_stop_pnl(load_hl(s)) for s in df['sym']}
df['pnl']=df['sym'].map(PNL)
dfp=df.dropna(subset=['pnl']).copy()

# Refit OOS model in-memory (logit) to get pmoon for the 2025 test (the favorable cohort)
from sklearn.linear_model import LogisticRegression
FEATS=['ret_1d','ret_3d','rv_3d','maxrunup_3d','maxdd_3d','fund_mean','fund_last']
def gate_pmoon(train_yrs, test_yr):
    tr=dfp[dfp.year.isin(train_yrs)]; te=dfp[dfp.year==test_yr].copy()
    Xtr=tr[FEATS].fillna(tr[FEATS].median()); Xte=te[FEATS].fillna(tr[FEATS].median())
    mu,sd=Xtr.mean(),Xtr.std().replace(0,1)
    lr=LogisticRegression(max_iter=1000,class_weight='balanced'); lr.fit((Xtr-mu)/sd,tr['moonshot'])
    te['pmoon']=lr.predict_proba((Xte-mu)/sd)[:,1]; return te

print('=== MODEL-GATE PLACEBO (G4): real skip-top-decile vs random same-count skip ===')
for tyrs,test in [([2023],2024),([2023,2024],2025)]:
    te=gate_pmoon(tyrs,test)
    n=len(te); kskip=max(1,int(round(0.10*n)))
    # real: skip the kskip highest pmoon
    real_keep=te.sort_values('pmoon').iloc[:n-kskip]
    real_mean=real_keep['pnl'].mean()
    # placebo: 5000 random skips of kskip names
    rng=np.random.default_rng(0); pv=te['pnl'].values
    placebo=[]
    for _ in range(5000):
        idx=rng.choice(n,n-kskip,replace=False); placebo.append(pv[idx].mean())
    placebo=np.array(placebo); pct=(placebo<real_mean).mean()
    print(f'  train{tyrs}->{test}: real_gated_mean {real_mean:+.4f} | placebo mean {placebo.mean():+.4f} '
          f'p{placebo.mean():+.0f} -> real ranks p{100*pct:.0f} (need >=p95)  ALLmean {te["pnl"].mean():+.4f}')

print('\n=== MOONSHOT base-rate by cohort (stationarity of the tail) ===')
print('  moonshot rate (>=+50% d3->30d):', {int(y):round(g["moonshot"].mean(),3) for y,g in df.groupby("year") if y<2026})
print('  fade rate (<0 d3->30d):       ', {int(y):round(g["fade"].mean(),3) for y,g in df.groupby("year") if y<2026})
print('  mean ret30_d3 by cohort:      ', {int(y):round(g["ret30_d3"].mean(),3) for y,g in df.groupby("year") if y<2026})
print('  median ret30_d3 by cohort:    ', {int(y):round(g["ret30_d3"].median(),3) for y,g in df.groupby("year") if y<2026})

print('\n=== POOLED stopped-short@3d->30d stop+50% by cohort (final honesty) ===')
for y,g in dfp.groupby('year'):
    if y>=2026: continue
    p=g['pnl'].values; rng=np.random.default_rng(int(y))
    b=np.array([rng.choice(p,len(p),replace=True).mean() for _ in range(3000)])
    print(f'  {int(y)} (n={len(p)}): mean {p.mean():+.4f} med {np.median(p):+.4f} hit {(p>0).mean():3.0%} '
          f'CI[{np.percentile(b,2.5):+.3f},{np.percentile(b,97.5):+.3f}] P(>0)={ (b>0).mean():.0%}')
