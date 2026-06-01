"""
iter-039 (B) — MODEL to predict fade-vs-moonshot from EARLY-WINDOW features (cohort-OOS).

Build PIT early-window features per listing (measured only within first 3 days of life):
  - ret_1d, ret_3d (early trend)
  - rv_3d (realized vol, hourly, ann)
  - maxrunup_3d, maxdd_3d (path extremes within window)
  - early funding mean/last (where available, else NaN -> impute)
Target: 30d outcome.
  - classification: P(moonshot) where moonshot = ret_30d (from day3 entry) >= +0.50 (the thing that
    kills the short). Also report a fade-label classifier (ret_30d_from_d3 < 0).
  - regression: ret_30d_from_d3 (the short's adverse/favorable move).
Cohort-OOS: train 2023 -> test 2024 ; train 2023-24 -> test 2025.
Report OOS AUC (moonshot) + Spearman IC (regression), then a model-gated short:
  short only names predicted NOT moonshot (or predicted fade), event-bootstrap CI + cohort split.
Skeptical: ~163 events, ~13-20 moonshots -> THIN. Report counts.
"""
import pandas as pd, numpy as np, os
np.seterr(all='ignore')
CACHE='/home/yuqing/ctaNew/data/ml/cache'
events=pd.read_parquet('/home/yuqing/ctaNew/agents_system/research/scripts/iter037_events.parquet')

def load_hl(sym):
    df=pd.read_parquet(f'{CACHE}/xs_feats_{sym}.parquet',columns=['close','high','low']).dropna(subset=['close'])
    c=df['close'].resample('1h').last(); hi=df['high'].resample('1h').max(); lo=df['low'].resample('1h').min()
    return pd.concat([c.rename('close'),hi.rename('high'),lo.rename('low')],axis=1).dropna()

def early_funding(sym, list_dt):
    fp=f'{CACHE}/funding_{sym}.parquet'
    if not os.path.exists(fp): return np.nan, np.nan
    f=pd.read_parquet(fp); f['calc_time']=pd.to_datetime(f['calc_time'],utc=True)
    f=f.set_index('calc_time').sort_index()
    if len(f)==0 or f.index[0] > list_dt + pd.Timedelta(days=2): return np.nan, np.nan
    e=f.loc[:list_dt+pd.Timedelta(days=3),'funding_rate']
    if len(e)<2: return np.nan, np.nan
    return e.mean(), e.iloc[-1]

rows=[]
for s in events['sym']:
    d=load_hl(s)
    if len(d) < 24*31: continue
    p0=d['close'].iloc[0]; W=3*24
    if len(d)<=W: continue
    pW=d['close'].iloc[W]
    seg=d.iloc[:W+1]
    fmean,flast=early_funding(s, d.index[0])
    # forward outcome from day3 entry to day30
    if len(d) <= 30*24: continue
    ret30_d3 = d['close'].iloc[30*24]/pW - 1.0
    rows.append(dict(
        sym=s, year=events.set_index('sym').loc[s,'year'],
        ret_1d=d['close'].iloc[24]/p0-1.0, ret_3d=pW/p0-1.0,
        rv_3d=seg['close'].pct_change().std()*np.sqrt(24*365),
        maxrunup_3d=seg['high'].max()/p0-1.0, maxdd_3d=seg['low'].min()/p0-1.0,
        fund_mean=fmean, fund_last=flast,
        ret30_d3=ret30_d3,
        moonshot=int(ret30_d3 >= 0.50),
        fade=int(ret30_d3 < 0.0),
    ))
df=pd.DataFrame(rows)
print(f'n={len(df)}  moonshots(>=+50% d3->30d)={df["moonshot"].sum()}  faders(<0)={df["fade"].sum()}')
print('moonshots by cohort:', df.groupby('year')['moonshot'].sum().to_dict())
print('faders by cohort:   ', df.groupby('year')['fade'].sum().to_dict())

FEATS=['ret_1d','ret_3d','rv_3d','maxrunup_3d','maxdd_3d','fund_mean','fund_last']
# impute funding NaN with median of TRAINING set only (done inside fit)

# ---- univariate IC of each early feature vs ret30_d3 (sanity, pooled) ----
from scipy.stats import spearmanr
print('\n=== univariate Spearman IC: early feature vs ret30_d3 (pooled, descriptive) ===')
for f in FEATS:
    v=df[[f,'ret30_d3']].dropna()
    if len(v)>10:
        ic,_=spearmanr(v[f],v['ret30_d3'])
        icm,_=spearmanr(v[f],(v['ret30_d3']>=0.50).astype(int))
        print(f'  {f:<12} IC(ret)={ic:+.3f}  IC(moonshot)={icm:+.3f}  n={len(v)}')

try:
    import lightgbm as lgb
    HAVE_LGB=True
except Exception:
    HAVE_LGB=False
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

def fit_predict_cls(tr, te, target='moonshot'):
    """Logistic (robust for thin/imbalanced) + LGBM if available. Returns dict of oos prob arrays."""
    Xtr=tr[FEATS].copy(); Xte=te[FEATS].copy()
    med=Xtr.median()
    Xtr=Xtr.fillna(med); Xte=Xte.fillna(med)
    # standardize for logistic
    mu,sd=Xtr.mean(),Xtr.std().replace(0,1)
    out={}
    lr=LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit((Xtr-mu)/sd, tr[target])
    out['logit']=lr.predict_proba((Xte-mu)/sd)[:,1]
    if HAVE_LGB and tr[target].sum()>=3:
        m=lgb.LGBMClassifier(n_estimators=100,num_leaves=7,learning_rate=0.05,
                             min_child_samples=8,subsample=0.8,colsample_bytree=0.8,
                             class_weight='balanced',verbose=-1)
        m.fit(Xtr,tr[target]); out['lgbm']=m.predict_proba(Xte)[:,1]
    return out

def fit_predict_reg(tr, te):
    Xtr=tr[FEATS].fillna(tr[FEATS].median()); Xte=te[FEATS].fillna(tr[FEATS].median())
    mu,sd=Xtr.mean(),Xtr.std().replace(0,1)
    r=Ridge(alpha=5.0); r.fit((Xtr-mu)/sd, tr['ret30_d3'])
    return r.predict((Xte-mu)/sd)

print('\n=== COHORT-OOS: predict MOONSHOT (classification, AUC) ===')
splits=[('2023->2024', df[df.year==2023], df[df.year==2024]),
        ('2023-24->2025', df[df.year.isin([2023,2024])], df[df.year==2025])]
oos_store={}
for name,tr,te in splits:
    nm_tr=tr['moonshot'].sum(); nm_te=te['moonshot'].sum()
    preds=fit_predict_cls(tr,te,'moonshot')
    line=f'  {name:<16} train_moon={nm_tr} test_moon={nm_te}/{len(te)}  '
    for mdl,pr in preds.items():
        try: auc=roc_auc_score(te['moonshot'],pr)
        except Exception: auc=np.nan
        line+=f'AUC[{mdl}]={auc:.3f}  '
    print(line)
    oos_store[name]=(te,preds)

print('\n=== COHORT-OOS: predict ret30_d3 (regression, Spearman IC) ===')
for name,tr,te in splits:
    pr=fit_predict_reg(tr,te)
    ic,_=spearmanr(pr, te['ret30_d3'])
    # IC vs moonshot label too
    icm,_=spearmanr(pr, te['moonshot'])
    print(f'  {name:<16} IC(ret)={ic:+.3f}  IC(moonshot)={icm:+.3f}  n={len(te)}')

# ---- MODEL-GATED SHORT: short names NOT predicted moonshot, OOS ----
# Use moonshot prob: skip top-decile predicted-moonshot; short the rest. day3->day30, stop+50% close gap.
def load_for_pnl(s):
    return load_hl(s)
COST=2*15/1e4
def short_stop_pnl(d, X=0.50, W=3*24, Hh=30*24, gap='close'):
    if len(d)<=Hh: return None
    p0=d['close'].iloc[W]; trig=p0*(1+X)
    seg=d['high'].iloc[W+1:Hh+1]; br=seg[seg>=trig]
    if len(br)==0: px=d['close'].iloc[Hh]
    else:
        bi=d.index.get_loc(br.index[0])
        px=max(trig, d['close'].iloc[bi]) if gap=='close' else max(trig, d['high'].iloc[min(bi+1,len(d)-1)])
    return -(px/p0-1.0)-COST

print('\n=== MODEL-GATED SHORT (OOS): short names with low predicted P(moonshot) ===')
def bootci(p,seed=0):
    p=np.array(p); rng=np.random.default_rng(seed)
    b=np.array([rng.choice(p,len(p),replace=True).mean() for _ in range(3000)])
    return np.percentile(b,[2.5,97.5]),(b>0).mean()

for name,(te,preds) in oos_store.items():
    for mdl,pr in preds.items():
        te2=te.copy(); te2['pmoon']=pr
        for q in [0.70,0.80,0.90]:
            thr=np.quantile(pr,q)
            keep=te2[te2['pmoon']<=thr]  # short only non-flagged
            pnls=[]
            for s in keep['sym']:
                d=load_for_pnl(s); v=short_stop_pnl(d)
                if v is not None: pnls.append(v)
            if len(pnls)<5: continue
            p=np.array(pnls); (lo,hi),pgt=bootci(p)
            # compare to shorting ALL (no gate) in same test set
            allp=[short_stop_pnl(load_for_pnl(s)) for s in te['sym']]; allp=np.array([x for x in allp if x is not None])
            print(f'  {name:<14} {mdl:<5} skip top-{1-q:.0%} pmoon: n={len(p):>2} mean {p.mean():+.4f} '
                  f'hit {(p>0).mean():3.0%} CI[{lo:+.3f},{hi:+.3f}] P(>0)={pgt:.0%} | ALL mean {allp.mean():+.4f}')

df.to_parquet('/home/yuqing/ctaNew/agents_system/research/scripts/iter039_model_feats.parquet')
print('\nsaved feats.')
