"""Reconstruct gain/loss diagnosis; qualify chronology and coverage conclusions.

No new policy or return forecast. Reuse saved loss heads, reproduce the five
market-grouped domain classifier folds, preserve their scores, and report
absolute score distributions as diagnostics only. No new cutoff or selection.
Inspect existing selection composition; do not replace its G2 acceptance.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import lightgbm as lgb
import pf_gain_loss_diag as D

GL, J, JF, F, C = D.GL, D.J, D.JF, D.F, D.C
ROOT = Path(__file__).resolve().parents[2]
OUT = GL.OUT.parent/'gain_loss_diagnosis_review_20261117'
REVIEW = ROOT/'predictfun/reviews/gain_loss_diagnosis_review_20261117'


def equal(a, b):
    C.require(np.allclose(a, b, atol=1e-9, rtol=0, equal_nan=True), 'numeric mismatch')


def chronological(train_days, test_day):
    C.require(max(train_days)<test_day, 'future training day')


def refuses(fn):
    try: fn()
    except (ValueError, AssertionError): return True
    raise ValueError('known-bad input accepted')


def controls():
    equal([1., np.nan], [1., np.nan]); chronological(['20261023'], '20261024')
    train = np.linspace(0., .1, 100); test = np.linspace(.8, .99, 100)
    # Even the lowest test quintile can have no overlap with training scores.
    C.require(test[:20].min()>train.max(), 'relative familiarity toy failed')
    return dict(equality_positive=True, corrupt_value_refused=refuses(lambda: equal([1.], [2.])),
                chronological_positive=True, future_fold_refused=refuses(lambda: chronological(['20261025'], '20261023')),
                relative_familiarity_does_not_imply_overlap=True)


def run():
    C.require(not OUT.exists(), 'review output exists'); OUT.mkdir()
    checks=controls(); GL.selftest(); J.verify(); inputs={}
    def hashed(path, expected=None):
        digest=C.sha(path)
        if expected is not None: C.require(digest==expected, 'artifact changed '+str(path))
        inputs[str(path)]=digest
        return digest
    def read(path):
        hashed(path); return json.loads(Path(path).read_text())
    def receipt(directory):
        r=read(directory/'receipt.json')
        for name,digest in r.get('outputs',{}).items(): hashed(directory/name,digest)
        return r
    for commit,name in [('fe325d18','pf_gain_loss.py'),('9b987f65','pf_gain_loss_diag.py')]:
        p=ROOT/'predictfun/research'/name
        committed=subprocess.check_output(['git','show',commit+':predictfun/research/'+name],cwd=ROOT)
        hashed(p,hashlib.sha256(committed).hexdigest())
    fit=receipt(GL.OUT/'fit'); er=receipt(GL.OUT/'eval')
    for art in fit['artifacts'].values(): hashed(art['path'],art['sha256'])
    published=read(ROOT/'predictfun/reviews/gain_loss_20261117/gain_loss_diag.json')
    hashed(D.OUT/'gain_loss_diag.json',C.sha(ROOT/'predictfun/reviews/gain_loss_20261117/gain_loss_diag.json'))
    T=pd.concat([JF.read(day)[0] for day in F.TRAIN],ignore_index=True)
    E=pd.read_parquet(GL.OUT/'eval/set_L.parquet')
    C.require(len(T)==fit['n'] and len(E)==er['set_L_n'],'population count')
    for frame in (T,E):
        frame['w']=frame.y_w.astype(float);frame['day']=frame.day.astype(str)
        g,l1,l2=GL.components(frame.w.values)
        frame['gain']=g;frame['loss']=l1+l2;frame['excess']=l2
    feat=GL.feature_list('full')
    pred=GL.predict(E,'full',fit['artifacts'])
    for k in ('g','l1','l2','g_p','l1_p','l2_p'): equal(pred[k],E['GL_'+k])
    equal(pred['w'],E.GL)
    E['p_loss']=pred['l1']+pred['l2']
    folds={}
    for held in F.TRAIN:
        train_days=[d for d in F.TRAIN if d!=held]; te=T[T.day==held]
        arts={key:str(D.OUT/('fold_'+held)/(key+'.txt')) for key in ('l1_p','l1_amt','l2_p','l2_amt')}
        for p in arts.values(): hashed(p)
        pr=D.loss_predict(te,feat,arts); ratio=float((pr['l1']+pr['l2']).sum()/te.loss.sum())
        equal(ratio,published['next_day_ratios'][held])
        folds[held]=dict(train_days=train_days,n=len(te),strictly_forward=max(train_days)<held,loss_ratio=ratio)
    cells={}
    for (day,kind),g in E.groupby(['day','kind']):
        event=g.loss>1e-9;severe=g.excess>1e-9
        cells[day+':'+kind]=dict(n=len(g),loss_ratio=float(g.p_loss.sum()/g.loss.sum()),
             severe_ratio=float(g.GL_l2.sum()/g.excess.sum()),
             predicted_loss_event_n=float(g.GL_l1_p.sum()),actual_loss_event_n=int(event.sum()),
             predicted_severe_event_n=float(g.GL_l2_p.sum()),actual_severe_event_n=int(severe.sum()))
    # Exact reproduction of the original domain-classifier design, not forward validation.
    A=pd.concat([T[feat].assign(y=0.,grp=T.day+':'+T.m.astype(str)),
                 E[feat].assign(y=1.,grp=E.day+':'+E.m.astype(str))],ignore_index=True)
    groups=A.grp.unique();rng=np.random.default_rng(20261117)
    mapping=dict(zip(groups,rng.integers(0,5,len(groups))))
    fold=A.grp.map(mapping).to_numpy();score=np.full(len(A),np.nan)
    for k in range(5):
        tr=fold!=k;te=fold==k
        model=lgb.train({**C.V.PARAMS,'objective':'binary'},lgb.Dataset(A.loc[tr,feat].to_numpy(np.float32),label=A.y[tr].to_numpy()),num_boost_round=200)
        score[te]=model.predict(A.loc[te,feat].to_numpy(np.float32),num_threads=1)
        model.save_model(str(OUT/f'domain_fold_{k}.txt'))
    C.require(np.isfinite(score).all(),'missing OOF score')
    auc=JF.binary_metrics(A.y.to_numpy(),score)['auc'];equal(auc,published['separation_auc'])
    A[['y','grp']].assign(fold=fold,score=score).to_parquet(OUT/'domain_scores.parquet',index=False)
    st=score[:len(T)];se=score[len(T):]
    q=pd.qcut(pd.Series(se).rank(method='first'),5,labels=False).to_numpy()
    quant=lambda x:{str(p):float(np.quantile(x,p)) for p in (0,.1,.5,.9,1)}
    familiarity={}
    for i in range(5):
        g=E[q==i];ratio=float(g.p_loss.sum()/g.loss.sum())
        equal(ratio,published['familiarity_ratios'][f'Q{i+1}'])
        familiarity[f'Q{i+1}']=dict(n=len(g),scores=quant(se[q==i]),loss_ratio=ratio)
    # G2 uses only day/kind counts; expose its unmatched market and side composition.
    composition={}
    for (day,kind),g in E.groupby(['day','kind']):
        for rate in GL.RATES:
            n=int(np.ceil(rate*len(g))); selected={}
            tie=[J.V.PATH.keyhash(day,int(m),float(t)) for m,t in zip(g.m,g.decision_ms)]
            for method in ('GL','DIRECT'):
                z=g.assign(tie=tie).sort_values([method,'tie'],ascending=[False,True]).head(n)
                selected[method]=z.groupby(['m','side']).size()
            joined=pd.concat(selected,axis=1).fillna(0)
            composition[f'{day}:{kind}:{rate}']=dict(n_each=n,
                market_side_cells_different=int((joined.GL!=joined.DIRECT).sum()),
                total_market_side_cells=len(joined),count_l1=int((joined.GL-joined.DIRECT).abs().sum()))
    labels={day:receipt(GL.OUT/('label_'+day)) for day in GL.EVAL_DAYS}
    result=dict(asof=C.stamp(),original_asof=published['asof'],source_sha256=C.sha(__file__),
        controls=checks,inputs=inputs,training_n=len(T),evaluation_n=len(E),cells=cells,folds=folds,
        strictly_forward_fold_n=sum(x['strictly_forward'] for x in folds.values()),
        coverage=dict(auc=auc,train_scores=quant(st),evaluation_scores=quant(se),familiarity=familiarity,
                      relative_quintiles_are_not_support_certificates=True),
        original_predicates=published['predicates'],selection_composition=composition,
        training_zero_wealth=int((T.w.abs()<=1e-9).sum()),training_zero_quantity=int((T.y_q<=1e-9).sum()),
        training_filled_zero_wealth=int(((T.y_q>1e-9)&(T.w.abs()<=1e-9)).sum()),
        canonical_checks={day:dict(n=r['canonical_checked'],positive_n=r['canonical_positive'],statuses=r['statuses']) for day,r in labels.items()},
        output_hashes={p.name:C.sha(p) for p in OUT.iterdir() if p.is_file()},
        limits=['Descriptive review of consumed dates; no new return model or policy.',
                'OOF domain scores are distribution diagnostics, not calibrated overlap probabilities.',
                'No inference that missing day intensity causes the loss shortfall.',
                'Original G1/G2 PASS is a relative focal-value screen, not successful loss calibration or profitability.'])
    C.dump(REVIEW/'review_v1.json',result)
    print(json.dumps(dict(asof=result['asof'],folds=folds,coverage=result['coverage'],
                         checks=checks,zero_wealth=result['training_zero_wealth'],
                         zero_quantity=result['training_zero_quantity'])),flush=True)


if __name__=='__main__':run()
