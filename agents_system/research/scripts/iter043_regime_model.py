"""
iter-043 — regime-as-MODEL-FEATURE. Last structurally-distinct new-listing angle.
i039 built a moonshot CLASSIFIER (failed). Here: predict the forward-30d return
(ret30_d3) from early-life features PLUS the forward-knowable regime state (alt30 =
trailing-30d alt-index return at entry), pooled, COHORT-OOS. Hypothesis: with the
regime as a feature the model learns "fade hard in alt-bear, flat in alt-bull" and a
model-gated short transports. Expected failure: only ~2-3 cohort years with little
regime variation → the alt30 interaction can't be learned OOS (it memorizes the
year). Honest gates: OOS rank-IC(pred,realized), model-gated short P(mean>0)>=95%,
beat shuffled-pred placebo >=p95, sign-consistent transport across cohorts.
"""
import pandas as pd, numpy as np
np.seterr(all='ignore')
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

m=pd.read_parquet('agents_system/research/scripts/iter039_model_feats.parquet')
g=pd.read_parquet('agents_system/research/scripts/iter040_events_gated.parquet')[['sym','alt30']]
df=m.merge(g,on='sym',how='left')
df=df[df['year'].isin([2023,2024,2025])].copy()
df=df.dropna(subset=['ret30_d3'])
FEATS=['ret_1d','ret_3d','rv_3d','maxrunup_3d','maxdd_3d','alt30']
df[FEATS]=df[FEATS].fillna(df[FEATS].median())
TGT='ret30_d3'
print(f'events {len(df)}  by yr {df.groupby("year").size().to_dict()}')

def fit_pred(tr,te,model='gbm'):
    Xtr,ytr=tr[FEATS].values,tr[TGT].values; Xte=te[FEATS].values
    if model=='gbm':
        mdl=GradientBoostingRegressor(n_estimators=150,max_depth=2,learning_rate=0.03,subsample=0.8,random_state=0)
    else:
        mdl=Ridge(alpha=5.0)
    mdl.fit(Xtr,ytr); return mdl.predict(Xte)

def short_pnl(pred,real):
    # short events the model predicts will FALL (pred<0): pnl = -1*real ; else flat
    pos=(pred<0).astype(float)*(-1.0)
    return pos*real, pos

# cohort-OOS splits
splits=[('2023->2024',[2023],2024),('2023+24->2025',[2023,2024],2025)]
for mdl in ['ridge','gbm']:
    print(f'\n=== MODEL={mdl} ===')
    allp=[]; allr=[]
    for nm,tryrs,teyr in splits:
        tr=df[df['year'].isin(tryrs)]; te=df[df['year']==teyr]
        if len(te)<8: continue
        pred=fit_pred(tr,te,mdl); real=te[TGT].values
        ic=spearmanr(pred,real).correlation
        pnl,pos=short_pnl(pred,real); traded=pos!=0
        mn=pnl[traded].mean() if traded.sum()>0 else np.nan
        # naive: short ALL
        naive=(-1.0*real).mean()
        print(f'  {nm:>15}: n_te={len(te)} OOS rankIC {ic:+.3f} | gated-short n={int(traded.sum())} mean {mn:+.3f} | naive-short-all mean {naive:+.3f}')
        allp.append(pred); allr.append(real)
    # pooled OOS gated short + bootstrap + shuffle placebo
    if allp:
        P=np.concatenate(allp); R=np.concatenate(allr)
        pnl,pos=short_pnl(P,R); tr=pos!=0; v=pnl[tr]
        if len(v)>=8:
            rng=np.random.default_rng(0)
            bse=np.array([rng.choice(v,len(v),replace=True).mean() for _ in range(2000)])
            pgt0=(bse>0).mean()
            # shuffle-pred placebo: permute predictions vs realized, recompute gated-short mean
            ph=[]
            for sd in range(500):
                rs=np.random.default_rng(sd); Ps=rs.permutation(P)
                pn,po=short_pnl(Ps,R); t2=po!=0
                if t2.sum()>0: ph.append(pn[t2].mean())
            ph=np.array(ph); pct=100*(ph<v.mean()).mean()
            print(f'  POOLED gated-short: n={len(v)} mean {v.mean():+.4f} P(mean>0)={pgt0:.3f} | shuffle-placebo rank p{pct:.0f} (p95 {np.percentile(ph,95):+.4f})')
