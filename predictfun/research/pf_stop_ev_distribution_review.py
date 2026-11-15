"""Post hoc tail and training/test distribution diagnosis, consumed data only.

No fits, tuning, null significance tests, policies, raw data or forward days.
All five original chronological folds; all 83 declared features. Declare:
1. Per-day/fold fill frequency, size, per-share value and tail severity.
2. Tail = below that fold's training ALL-action 1st percentile, fixed before
   test scoring; evaluate existing Huber/L2/B2/fill risk scores at descriptive
   top 1/5/10% action ranks, plus original kept-set residual tail loss.
3. Feature drift: every feature's train/test missing rates, train min/max
   exceedances, and total variation of train-decile bins (not an OOD proof).
4. Exact common-cell mean decomposition into composition and within-cell
   changes. Decision-time strata side/rung x price/sigma training terciles;
   timing strata side/rung/certificate; filled-only descriptive strata
   side/rung/E2 fill fraction/filled-size training terciles. Train minimum
   50 actions (25 for filled-only); report excluded support explicitly.
5. Repeat main outcome comparisons on certified-estimated actions. Never use
   certificate, E2 composition or realized tail membership as model inputs.
No selected threshold, inferential confidence or claim that all tails can be
avoided. Matched random-selection expectation is computed analytically as a
descriptive comparator, not a sampled null or significance test.
"""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import resource
import subprocess

import numpy as np
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew/data/predict_fun/research')
SRC=ROOT/'stop_ev_model_20261115';CERT=ROOT/'stop_ev_timing_certified_20261115_v1'
E2=ROOT/'stop_ev_e2sep_20261115';CAUSE=ROOT/'stop_ev_causes_20261115_v1'
DAYS=('20261023','20261024','20261025','20261026','20261027','20261031','20261101','20261102','20261103')
TESTS=DAYS[4:];KEY=['day','m','side','rung','generation','decision_ms']


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def guard():require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'RSS cap')


def metric(p):
    if not len(p):return dict(n=0)
    y=p.value_cents.to_numpy(float);q=p.filled_quantity.to_numpy(float);f=q>0
    neg=y<0;pos=y>0;n=int(f.sum());gross=-y[neg].sum()
    return dict(n=len(p),filled=n,fill_rate=float(f.mean()),value_mean=float(y.mean()),
        filled_value_mean=float(y[f].mean()) if n else None,
        filled_quantity_mean=float(q[f].mean()) if n else None,
        value_per_filled_share=float(y.sum()/q.sum()) if q.sum()>0 else None,
        negative_rate=float(neg.mean()),negative_mean=float(y[neg].mean()) if neg.any() else None,
        positive_rate=float(pos.mean()),positive_mean=float(y[pos].mean()) if pos.any() else None,
        gross_loss=float(gross),gross_gain=float(y[pos].sum()),
        at_price_mean=float(p.at_price_value_cents.mean()),through_price_mean=float(p.through_price_value_cents.mean()),
        e2_fraction_filled_shares=float(p.fill_e2.sum()/q.sum()) if q.sum()>0 else None,
        strict_fraction=float(p.certified_ws.mean()),estimated_fraction=float(p.certified_estimated.mean()))


def decompose(a,b,key,min_train=50):
    """Exact Oaxaca-style mean identity on explicitly reported common cells."""
    def agg(x):return x.groupby(key,dropna=False).value_cents.agg(['size','mean'])
    aa,bb=agg(a),agg(b);j=aa.join(bb,lsuffix='_tr',rsuffix='_te',how='outer')
    good=(j.size_tr>=min_train)&(j.size_te>0);g=j[good]
    if not len(g):return dict(train_n=len(a),test_n=len(b),common_cells=0)
    nt=float(g.size_tr.sum());ne=float(g.size_te.sum());wt=g.size_tr/nt;we=g.size_te/ne
    mt=float((wt*g.mean_tr).sum());me=float((we*g.mean_te).sum())
    mix=float(((we-wt)*g.mean_tr).sum());within=float((we*(g.mean_te-g.mean_tr)).sum())
    require(abs((me-mt)-mix-within)<1e-8,'mean decomposition does not reconcile')
    return dict(train_n=len(a),test_n=len(b),train_matched=int(nt),test_matched=int(ne),
        train_excluded=len(a)-int(nt),test_excluded=len(b)-int(ne),common_cells=len(g),
        train_mean=mt,test_mean=me,gap=me-mt,composition=mix,within=within,
        test_matched_fraction=ne/len(b),cells=[{**dict(zip(key,k if isinstance(k,tuple) else (k,))),
            **{c:float(v) for c,v in row.items()}} for k,row in g.iterrows()])


def bucket(tr,te,n=3):
    tr=np.asarray(tr,float);te=np.asarray(te,float);finite=np.isfinite(tr)
    edges=np.unique(np.quantile(tr[finite],np.arange(1,n)/n)) if finite.any() else np.array([])
    def b(x):return np.where(np.isfinite(x),np.searchsorted(edges,x,side='right'),-1)
    return b(tr),b(te),edges.tolist()


def drift(tr,te,F):
    result={};any_out=np.zeros(len(te),bool)
    for c in F:
        a=tr[c].to_numpy(float);b=te[c].to_numpy(float);fa=np.isfinite(a);fb=np.isfinite(b)
        ba,bb,edges=bucket(a,b,10)
        labels=np.arange(-1,len(edges)+1)
        pa=np.array([(ba==i).mean() for i in labels]);pb=np.array([(bb==i).mean() for i in labels])
        outside=fb & ((b<a[fa].min())|(b>a[fa].max())) if fa.any() else fb
        any_out|=outside
        result[c]=dict(train_missing=float((~fa).mean()),test_missing=float((~fb).mean()),
            test_outside_train_range=float(outside.mean()),bin_total_variation=float(np.abs(pa-pb).sum()/2),
            train_median=float(np.median(a[fa])) if fa.any() else None,test_median=float(np.median(b[fb])) if fb.any() else None,
            train_edges=edges)
    return dict(features=result,any_feature_outside_train_range=float(any_out.mean()),
                interpretation='Marginal drift only; min/max or quantile bins do not establish joint support or conditional stability.')


def tail_metric(p,cut):
    y=p.value_cents.to_numpy(float);tail=y<=cut;loss=np.maximum(-y,0.);total=loss.sum()
    return dict(n=len(p),cut=float(cut),tail_n=int(tail.sum()),tail_rate=float(tail.mean()),
        tail_mean=float(y[tail].mean()) if tail.any() else None,tail_loss=float(loss[tail].sum()),
        tail_loss_fraction=float(loss[tail].sum()/total) if total else None)


def score_capture(p,cut,col,sign):
    loss=np.maximum(-p.value_cents.to_numpy(float),0.);tail=p.value_cents.to_numpy(float)<=cut
    out={};score=sign*p[col].to_numpy(float)
    require(np.isfinite(score).all(),'nonfinite score')
    ix=np.argsort(-score,kind='stable')
    for fraction in (.01,.05,.10):
        n=max(1,int(np.ceil(fraction*len(p))));sel=np.zeros(len(p),bool);sel[ix[:n]]=True
        expected_tail=expected_loss=0.
        for _,g in p.assign(selected=sel).groupby(['side','rung']):
            ratio=float(g.selected.mean());expected_tail+=ratio*int((g.value_cents<=cut).sum())
            expected_loss+=ratio*float(np.maximum(-g.value_cents,0.).sum())
        out[str(fraction)]=dict(selected=n,tail_selected=int((tail&sel).sum()),tail_total=int(tail.sum()),
            tail_capture=float((tail&sel).sum()/tail.sum()) if tail.any() else None,
            tail_capture_matched_expectation=float(expected_tail/tail.sum()) if tail.any() else None,
            gross_loss_selected=float(loss[sel].sum()),gross_loss_total=float(loss.sum()),
            gross_loss_matched_expectation=float(expected_loss),
            foregone_positive_value=float(np.maximum(p.value_cents.to_numpy(float)[sel],0.).sum()))
    return out


def selftest():
    a=pd.DataFrame(dict(cell=['a']*100+['b']*100,value_cents=[0.]*100+[-10.]*100))
    b=pd.DataFrame(dict(cell=['a']*20+['b']*80,value_cents=[0.]*20+[-10.]*80))
    r=decompose(a,b,['cell']);require(abs(r['composition']+3)<1e-9 and abs(r['within'])<1e-9,'mix control')
    b=pd.DataFrame(dict(cell=['a']*50+['b']*50,value_cents=[-2.]*50+[-12.]*50))
    r=decompose(a,b,['cell']);require(abs(r['within']+2)<1e-9 and abs(r['composition'])<1e-9,'within control')
    toy=pd.DataFrame(dict(side=[0]*100,rung=[0]*100,value_cents=[-10.]*10+[1.]*90,risk=[1.]*10+[0.]*90))
    x=score_capture(toy,-5,'risk',1)['0.1'];require(x['tail_capture']==1. and abs(x['tail_capture_matched_expectation']-.1)<1e-9,'tail capture control')
    ba,bb,_=bucket([0,1,2],[np.nan,5]);require(bb[0]==-1,'missing bucket')
    corrupt=toy.copy();corrupt.loc[0,'risk']=np.nan
    try:score_capture(corrupt,-5,'risk',1)
    except ValueError:return dict(composition=True,within=True,tail_capture=True,missing_bucket=True,bad_score_refused=True)
    raise ValueError('bad score accepted')


def run(out):
    out.mkdir(parents=True,exist_ok=False);hashes={};ctl=selftest();panels={};excluded={};records={}
    def read(p,cols=None):
        h=sha(p);require(str(p) not in hashes or hashes[str(p)]==h,'input changed across reads');hashes[str(p)]=h
        return pd.read_parquet(p,columns=cols) if p.suffix=='.parquet' else json.loads(p.read_text())
    for day in DAYS:
        d=read(SRC/f'ev_x_{day}.parquet');require(not d.duplicated(KEY).any(),'duplicate action')
        e=read(E2/f'e2_{day}_0.parquet',KEY+['status','value_cents','fill_e2','reach_ws','reach_e2','act_ws','act_e2'])
        j=d.merge(e,on=KEY,how='left',validate='one_to_one',indicator=True,suffixes=('','_e'))
        require((j._merge=='both').all() and np.array_equal(j.value_cents,j.value_cents_e),'E2 label identity')
        require((j.status==j.status_e).all(),'E2 status identity')
        for col in ['fill_e2','reach_ws','reach_e2','act_ws','act_e2']:d[col]=j[col].values
        require((d.fill_e2>=0).all() and (d.fill_e2<=d.filled_quantity+1e-7).all(),'E2 fill exceeds quantity')
        for mode in ['certified_ws','certified_estimated']:
            cr=read(CERT/day/'receipt.json');path=CERT/day/(mode+'.parquet');c=read(path,KEY+['status','value_cents'])
            require(hashes[str(path)]==cr['artifacts'][path.name],'certificate hash mismatch');c.day=c.day.astype('int64')
            j=d[KEY+['value_cents']].merge(c,on=KEY,how='left',validate='one_to_one',indicator=True,suffixes=('','_c'))
            require((j._merge=='both').all(),'certificate action missing');ok=j.status.str.startswith('eligible')
            require(np.array_equal(j.loc[ok,'value_cents'],j.loc[ok,'value_cents_c']),'certificate value differs');d[mode]=ok.values
        d['timing_group']=np.where(d.certified_ws,'strict',np.where(d.certified_estimated,'estimated','uncertified'))
        d['fill_source']=np.where(d.fill_e2<=1e-8,'ws_only',np.where(d.fill_e2>=d.filled_quantity-1e-8,'e2_only','mixed'))
        # This source composition is an outcome descriptor, never a feature.
        panels[day]=d;excluded[day]=read(SRC/f'ev_x_{day}.json')['status'];guard()
    for day in TESTS:
        fit=read(SRC/f'fit_{day}.json');F=fit['features'];td=fit['train_days']
        require(all(d in DAYS for d in td) and max(td)<fit['cal']<day,'fold chronology')
        tr=pd.concat([panels[d] for d in td],ignore_index=True);te=panels[day].copy()
        # Bins are fitted to training input values only.
        for col in ['o_price','sigma']:
            tr[col+'_bin'],te[col+'_bin'],_=bucket(tr[col],te[col])
        tf=tr[tr.filled_quantity>0].copy();ef=te[te.filled_quantity>0].copy()
        tf['quantity_bin'],ef['quantity_bin'],qedges=bucket(tf.filled_quantity,ef.filled_quantity)
        cut=float(tr.value_cents.quantile(.01));require(cut<0,'negative training tail absent')
        cp=CAUSE/day/'predictions.parquet';p=read(cp)
        require(p[KEY].equals(te[KEY]) and np.array_equal(p.value_cents,te.value_cents),'saved scores differ')
        captures={name:score_capture(p,cut,col,sign) for name,col,sign in [('huber','pred',-1),('l2','l2_pred',-1),('order_only','b2',-1),('fill_probability','p_fill',1)]}
        a,b=metric(tr),metric(te)
        fill_mix=(b['fill_rate']-a['fill_rate'])*a['filled_value_mean']
        severity=b['fill_rate']*(b['filled_value_mean']-a['filled_value_mean'])
        require(abs(b['value_mean']-a['value_mean']-fill_mix-severity)<1e-8,'fill/severity identity')
        rec=dict(train_days=td,train=a,test=b,gap=b['value_mean']-a['value_mean'],
            fill_frequency_component=fill_mix,filled_value_component=severity,tail_train=tail_metric(tr,cut),tail_test=tail_metric(te,cut),
            original_kept_tail=tail_metric(p[p.pred>0],cut),feature_drift=drift(tr,te,F),
            decision_state=decompose(tr,te,['side','rung','o_price_bin','sigma_bin']),
            timing_mix=decompose(tr,te,['side','rung','timing_group']),
            filled_composition=decompose(tf,ef,['side','rung','fill_source','quantity_bin'],25),
            filled_size_edges=qedges,
            certified=dict(train=metric(tr[tr.certified_estimated]),test=metric(te[te.certified_estimated])),
            filled_by_source=dict(train={g:metric(q) for g,q in tf.groupby('fill_source')},test={g:metric(q) for g,q in ef.groupby('fill_source')}),
            tail_capture=captures)
        records[day]=rec;guard();print(json.dumps(dict(day=day,gap=rec['gap'],fill_frequency=fill_mix,severity=severity,
            decision_state={k:v for k,v in rec['decision_state'].items() if k!='cells'},filled_composition={k:v for k,v in rec['filled_composition'].items() if k!='cells'})),flush=True)
    daily={d:metric(x) for d,x in panels.items()};pooled_capture={}
    for name in records[TESTS[0]]['tail_capture']:
        pooled_capture[name]={}
        for fraction in ('.01','.05','.1'):
            key=str(float(fraction));parts=[v['tail_capture'][name][key] for v in records.values()]
            totals={c:sum(z[c] for z in parts) for c in ['selected','tail_selected','tail_total','gross_loss_selected','gross_loss_total','gross_loss_matched_expectation','foregone_positive_value']}
            totals['tail_capture']=totals['tail_selected']/totals['tail_total']
            totals['tail_capture_matched_expectation']=sum(z['tail_capture_matched_expectation']*z['tail_total'] for z in parts)/totals['tail_total']
            pooled_capture[name][key]=totals
    for p,h in hashes.items():require(sha(Path(p))==h,'input changed')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        controls=ctl,input_sha256=hashes,daily=daily,exclusion_statuses=excluded,folds=records,pooled_tail_capture=pooled_capture,
        n_unique_test_actions=sum(len(panels[d]) for d in TESTS),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        interpretation='Descriptive consumed-data decomposition. Common cells and fill/source groups do not prove conditional exchangeability. Training tails and state bins fixed before test use; top-risk shares are retrospective diagnostic ranks, not deployed thresholds. No model/policy selected.')
    with (out/'report.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(asof=result['asof'],n=result['n_unique_test_actions'],pooled_tail_capture=pooled_capture)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
