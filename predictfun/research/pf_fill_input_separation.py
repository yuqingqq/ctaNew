"""Input-first good/bad fill audit; no learner or policy fit.

Nine consumed days. Four early days set feature bin edges/direction only;
five chronological later days are descriptive checks, not fresh validation.
Primary = certified_estimated neutral actions. Filled good/bad subsets are
DIAGNOSTICS, never a training/action population. Good y>0, bad y<0, filled
zero separate; also report the predeclared +/-0.5 c/share material subset.

Every existing input (83) + five declared physical scores. Raw-direction AUC
and dev-oriented AUC, per day/side/timing subgroup; conditional AUC within
day/side/rung/floor(edge)/fixed P(fill) bands, plus hour as a finer check.
Pair-weighted conditional AUC reports covered good/bad actions and pairs.
Five dev-quantile bins and missing bucket, all features, no selected feature.
Four predeclared logical flags, all-action value vs an exact analytic matched
selection expectation (day/hour/side/rung/remaining/edge/fill-probability and
feature availability). No sampled null, significance test or confidence claim.

All stored inputs independently rebuilt from event-derived raw feature rows
and client-known action fields, with exact action/time matches and float32
parity. This does not reread the collector tape; earlier raw-tape recomputation
and future-perturbation receipts are preserved and identified separately.
No horizon, active guard, model, fitted thresholds or protected data changed.
"""
import argparse
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import resource
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import pyarrow.parquet as pq

from pf_stop_ev_generalization_review import validate, selftest as parity_controls

ROOT=Path('/home/yuqing/ctaNew/data/predict_fun/research')
SRC=ROOT/'stop_ev_model_20261115';RAW=ROOT/'stop_model_20261114';ACT=ROOT/'stop_ev_20261115'
DEV=('20261023','20261024','20261025','20261026')
TEST=('20261027','20261031','20261101','20261102','20261103');DAYS=DEV+TEST
KEY=['day','m','side','rung','generation','decision_ms'];RK=['m','side','k']
ORDER=['o_rung','o_depth','o_price','o_age','o_rem','o_q_rel','o_shown_rel','o_q0_rel','o_q_frac','o_qunk','o_edge','o_horizon']
X5=['adv_ofi250','adv_ofi1000','adv_dd3_250','adv_dd3_1000','adv_bnd_imb5','adv_bnd_imb20','adv_d_bnd_imb5_250','adv_bnd_ofi250','adv_bnd_ofi1000','bnd_age']
DERIVED=['fair_deficit','fair_shock_over_edge','bn50_shock_over_edge','bn250_shock_over_edge','micro_deficit']
MATCH=['day','side','rung','edge_band','pfill_band']
ECON=['day','hour','side','rung','x_known_remaining','edge_band','pfill_band']


def require(ok,msg):
    if not bool(ok):raise ValueError(msg)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def guard():require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2048*1024,'RSS cap')


def eq(a,b,msg):require(np.array_equal(np.asarray(a),np.asarray(b),equal_nan=True),msg)


def auc(y,x):
    y=np.asarray(y,bool);x=np.asarray(x,float);p=int(y.sum());n=len(y)-p
    return float((rankdata(x)[y].sum()-p*(p+1)/2)/(p*n)) if p and n else None


def categories(d):
    y=d.value_cents.to_numpy(float);q=d.filled_quantity.to_numpy(float)
    require(np.isfinite(y).all() and np.isfinite(q).all() and (q>=0).all(),'invalid outcome')
    require((np.abs(y[q==0])<=1e-8).all(),'nonzero no-fill value')
    return np.where(q<=0,'no_fill',np.where(y< -1e-8,'bad',np.where(y>1e-8,'good','filled_zero')))


def population(d):
    y=d.value_cents.to_numpy(float);q=d.filled_quantity.to_numpy(float);cat=categories(d)
    return dict(n=len(d),classes={k:int((cat==k).sum()) for k in ['no_fill','good','bad','filled_zero']},
        value_cents=float(y.sum()),value_mean=float(y.mean()) if len(y) else None,
        good_value_cents=float(y[y>0].sum()),bad_value_cents=float(y[y<0].sum()),
        value_per_filled_share=float(y.sum()/q.sum()) if q.sum() else None)


def add_logic(d):
    d=d.copy();d['fair_deficit']=d.gap_adv-d.o_edge
    d['fair_shock_over_edge']=d.dfv250-d.o_edge
    d['bn50_shock_over_edge']=d.sens*d.adv_bn_r50-d.o_edge
    d['bn250_shock_over_edge']=d.sens*d.adv_bn_r250-d.o_edge
    d['micro_deficit']=d.adv_pf_micro_dev-d.o_edge
    d['class']=categories(d);d['edge_band']=np.floor(d.o_edge).astype(int)
    return d


def matched_auc(d,x,extra_hour=False):
    x=np.asarray(x,float);cat=d['class'].to_numpy();ok=np.isfinite(x)&np.isin(cat,['good','bad'])
    ix=np.flatnonzero(ok);z=d.iloc[ix].copy();z['score']=x[ix]
    pairs=concordant=ng=nb=groups=0
    for _,g in z.groupby(MATCH+(['hour'] if extra_hour else []),dropna=False,sort=False):
        yy=g['class'].to_numpy()=='bad';p=int(yy.sum());n=len(g)-p
        if not p or not n:continue
        pairs+=p*n;concordant+=auc(yy,g.score)*p*n;ng+=n;nb+=p;groups+=1
    return dict(auc=float(concordant/pairs) if pairs else None,pairs=int(pairs),good=int(ng),bad=int(nb),groups=groups,
        usable_good=int((cat[ok]=='good').sum()),usable_bad=int((cat[ok]=='bad').sum()))


def signal_stats(d,x,edges,conditional=True):
    x=np.asarray(x,float);y=d.value_cents.to_numpy(float);q=d.filled_quantity.to_numpy(float);cat=d['class'].to_numpy()
    fin=np.isfinite(x);fg=fin&np.isin(cat,['good','bad']);vps=np.divide(y,q,out=np.zeros(len(y)),where=q>0)
    strong=fin&(q>0)&(np.abs(vps)>=.5)
    b=np.where(fin,np.searchsorted(edges,x,side='right'),-1)
    bins={str(k):population(d.iloc[np.flatnonzero(b==k)]) for k in np.unique(b)}
    stats=dict(finite=int(fin.sum()),missing=int((~fin).sum()),
        missing_by_class={c:float((~fin[cat==c]).mean()) if (cat==c).any() else None for c in ['no_fill','good','bad','filled_zero']},
        auc_bad_good=auc(cat[fg]=='bad',x[fg]),bad_good_n=int(fg.sum()),
        auc_fill=auc(q[fin]>0,x[fin]),
        strong_auc=auc(vps[strong]<0,x[strong]),strong_n=int(strong.sum()),
        median_good=float(np.median(x[fin&(cat=='good')])) if (fin&(cat=='good')).any() else None,
        median_bad=float(np.median(x[fin&(cat=='bad')])) if (fin&(cat=='bad')).any() else None,bins=bins)
    if conditional:
        stats['matched']=matched_auc(d,x);stats['matched_hour']=matched_auc(d,x,True)
    return stats


def logic_rules(d):
    return {
        'fair_below_order':(np.isfinite(d.fair_deficit),d.fair_deficit>0),
        'fair_shock_exceeds_edge':(np.isfinite(d.fair_shock_over_edge),d.fair_shock_over_edge>0),
        'joint_adverse_pressure':(np.isfinite(d[['dfv250','adv_ofi250','adv_bn_flow250']]).all(axis=1),
            (d.dfv250>0)&(d.adv_ofi250>0)&(d.adv_bn_flow250>0)),
        'thin_adverse_book':(np.isfinite(d[['adv_pf_imb1','adv_d_bnimb250','adv_ofi250']]).all(axis=1),
            (d.adv_pf_imb1>=1/3)&(d.adv_d_bnimb250>0)&(d.adv_ofi250>0))}


def rule_stats(d,available,flag):
    av=np.asarray(available,bool);sel=av&np.asarray(flag,bool);y=d.value_cents.to_numpy(float);expected=expbad=expgood=0.;vary=0
    z=d.copy();z['available']=av
    for ix in z.groupby(ECON+['available'],dropna=False,sort=False).indices.values():
        ix=np.asarray(ix);n=int(sel[ix].sum());ratio=n/len(ix)
        expected+=ratio*y[ix].sum();expbad+=ratio*np.maximum(-y[ix],0).sum();expgood+=ratio*np.maximum(y[ix],0).sum()
        vary+=int(0<n<len(ix))
    chosen=population(d.iloc[np.flatnonzero(sel)])
    return dict(available=int(av.sum()),selected=chosen,unselected=population(d.iloc[np.flatnonzero(~sel)]),
        expected_selected_value_cents=float(expected),expected_gross_loss_cents=float(expbad),expected_gross_gain_cents=float(expgood),
        extra_loss_avoidance_vs_matched_cents=float(expected-y[sel].sum()),varying_strata=vary,
        interpretation='A flag is a diagnostic selection, not a deployed cancellation; same counts and availability within known-state strata.')


def rebuild_inputs(day,d,F,read):
    market=[c for c in F if c not in ORDER]
    rawcols=[c[2:] if c.startswith('r_') else c for c in market if c not in X5]
    r=read(RAW/f'rows_{day}.parquet',list(dict.fromkeys(RK+['t','best','scale_w30']+rawcols)))
    require(not r.duplicated(RK).any(),'duplicate feature row')
    index=d[RK].merge(r[RK].assign(pos=np.arange(len(r))),on=RK,how='left',validate='many_to_one')
    require(index.pos.notna().all(),'feature row missing');pos=index.pos.to_numpy(int);rs=r.iloc[pos].reset_index(drop=True)
    eq(rs.t,d.decision_ms,'feature time mismatch')
    x=read(RAW/f'rows_{day}_x5.parquet',RK+X5)
    require(x[RK].equals(r[RK]),'x5 source keys differ');xs=x.iloc[pos].reset_index(drop=True)
    sc=rs.scale_w30.to_numpy(float);require((sc>0).all(),'invalid scale');rebuilt={}
    for c in market:
        if c in X5:a=xs[c].to_numpy(float)
        elif c.startswith('r_'):
            raw=c[2:];a=rs[raw].to_numpy(float);a=a-np.log(sc) if raw in ['own_l1','opp_l1'] else a/sc
        else:a=rs[c].to_numpy(float)
        rebuilt[c]=a.astype(np.float32);eq(rebuilt[c],d[c],f'raw-to-input {c}')
    feat=['side','rung','depth','price','age_ms','known_remaining','submitted_quantity','queue_ahead_known','shown_known',
          'queue_initial_known','queue_initial_unknown','effective_horizon_ms']
    a=read(ACT/f'actions_{day}_all_anchors_sensitivity.parquet',KEY+['status','value_cents']+['x_'+c for c in feat])
    require(not a.duplicated(KEY).any(),'duplicate source action')
    j=d[KEY+['value_cents']].merge(a,on=KEY,how='left',validate='one_to_one',indicator=True,suffixes=('','_source'))
    require((j._merge=='both').all(),'source action missing');eq(j.value_cents,j.value_cents_source,'source label differs')
    sg=np.where(j.x_side==0,1.,-1.);sh=j.x_shown_known.to_numpy(float);qa=j.x_queue_ahead_known.to_numpy(float)
    with np.errstate(divide='ignore',invalid='ignore'):
        vals=dict(o_rung=j.x_rung,o_depth=j.x_depth,o_price=j.x_price,o_age=np.minimum(j.x_age_ms,60000.),
            o_rem=j.x_known_remaining/j.x_submitted_quantity,o_q_rel=qa/sc,o_shown_rel=sh/sc,o_q0_rel=j.x_queue_initial_known/sc,
            o_q_frac=np.where(np.isfinite(qa)&np.isfinite(sh)&(sh>0),qa/np.where(sh>0,sh,1.),np.nan),
            o_qunk=j.x_queue_initial_unknown,o_edge=sg*(rs.pf_mid.to_numpy(float)-j.x_price),o_horizon=j.x_effective_horizon_ms/1000.)
    for c,v in vals.items():
        a=np.asarray(v,float);a=np.where(np.isfinite(a),a,np.nan);eq(a.astype(np.float32),d[c],f'order input {c}')
    eq((sg*(rs.best.to_numpy()-j.x_price.to_numpy())).astype(np.float32),d.o_depth,'depth/side mismatch')
    st=read(RAW/f'rows_{day}.json');xst=read(RAW/f'rows_{day}_x5.json')
    out=dict(actions=len(d),fields=len(F),mismatches=0,feature_times_exact=True,raw_feature_rows=len(r),
        upstream_future_perturbation=st['falsifier'],upstream_x5=dict(pair_checked=xst['pair_checked'],pair_differ=xst['pair_differ'],
        depth_checked=xst['depth_checked'],depth_differ=xst['depth_differ']))
    del r,x,rs,xs,a,j;gc.collect();guard();return out


def selftest():
    checks=parity_controls()
    d=pd.DataFrame(dict(value_cents=[0.,2.,-3.,0.],filled_quantity=[0.,1.,1.,2.]))
    require(categories(d).tolist()==['no_fill','good','bad','filled_zero'],'category control')
    bad=d.copy();bad.loc[0,'value_cents']=1
    try:categories(bad)
    except ValueError:checks['invalid_no_fill_refused']=True
    else:raise ValueError('invalid no-fill accepted')
    require(auc([False,False,True,True],[0,0,1,1])==1. and auc([False,False,True,True],[1,1,1,1])==.5,'AUC controls')
    toy=pd.DataFrame({c:[0]*4 for c in MATCH+['hour']});toy['class']=['good','bad','good','bad']
    require(matched_auc(toy,[0,1,0,1])['auc']==1.,'matched positive control')
    # Signal entirely confounded by group: within-group discrimination must vanish.
    toy['side']=[0,0,1,1];require(matched_auc(toy,[0,0,1,1])['auc']==.5,'confounding control')
    sg=np.array([1.,-1.]);mid=np.array([50.,50.]);price=np.array([49.,51.]);future=np.array([48.,52.]);edge=sg*(mid-price)
    value=sg*(future-price);adverse=sg*(mid-future)
    eq(value,edge-adverse,'economic sign identity');require((value<0).all(),'ask/bid sign control')
    checks.update(category_control=True,auc_positive=True,confounding_control=True,economic_sign_identity=True);return checks


def run(out):
    require(not out.exists(),'output exists');out.mkdir(parents=True);controls=selftest();hashes={};panels={};lineage={};statuses={}
    def read(p,cols=None):
        h=sha(p);require(str(p) not in hashes or hashes[str(p)]==h,'input changed');hashes[str(p)]=h
        return pd.read_parquet(p,columns=cols) if p.suffix=='.parquet' else json.loads(p.read_text())
    F=read(SRC/f'fit2_{TEST[0]}.json')['features'];require(len(F)==83,'feature contract')
    for day in DAYS:
        d=read(SRC/f'ev_x2_{day}.parquet');require(not d.duplicated(KEY).any(),'duplicate action')
        lineage[day]=rebuild_inputs(day,d,F,read)
        statuses[day]=dict(original=read(SRC/f'ev_x_{day}.json')['status'],certified=read(SRC/f'ev_x2_{day}.json')['checks'])
        if day in TEST:
            p=read(SRC/f'pred2_{day}_{day}.parquet');validate(d,p);d['p_fill']=p.p_fill;d['existing_value_score']=-p.v_fill
            d['pfill_band']=np.searchsorted([.05,.1,.2,.4],d.p_fill)
        d=d[d.cert_est].reset_index(drop=True);panels[day]=add_logic(d);guard()
        print(json.dumps(dict(day=day,lineage=lineage[day],population=population(d))),flush=True)
    dev=pd.concat([panels[d] for d in DEV],ignore_index=True);test=pd.concat([panels[d] for d in TEST],ignore_index=True)
    names=F+DERIVED;bins={};direction={};dev_stats={}
    for c in names:
        x=dev[c].to_numpy(float);v=x[np.isfinite(x)];edges=np.unique(np.quantile(v,[.2,.4,.6,.8])).tolist() if len(v) else []
        bins[c]=edges;z=signal_stats(dev,x,edges,False);dev_stats[c]=z;direction[c]=-1 if z['auc_bad_good'] is not None and z['auc_bad_good']<.5 else 1
    daily={};pooled={}
    for day in TEST:
        d=panels[day];scores={}
        for c in names:
            z=signal_stats(d,d[c],bins[c]);z['dev_direction']=direction[c]
            z['oriented_auc']=z['auc_bad_good'] if direction[c]==1 or z['auc_bad_good'] is None else 1-z['auc_bad_good']
            scores[c]=z
        daily[day]=dict(population=population(d),features=scores,
            rules={k:rule_stats(d,*v) for k,v in logic_rules(d).items()})
        (out/f'day_{day}.json').write_text(json.dumps(daily[day],indent=2,allow_nan=False)+'\n')
    for c in names:
        z=signal_stats(test,test[c],bins[c]);z['dev_direction']=direction[c]
        z['oriented_auc']=z['auc_bad_good'] if direction[c]==1 or z['auc_bad_good'] is None else 1-z['auc_bad_good']
        z['oriented_positive_days']=sum(daily[d]['features'][c]['oriented_auc'] is not None and daily[d]['features'][c]['oriented_auc']>.5 for d in TEST)
        z['by_side']={str(s):signal_stats(test[test.side==s].reset_index(drop=True),test.loc[test.side==s,c],bins[c]) for s in [0,1]}
        z['strict']=signal_stats(test[test.cert_ws].reset_index(drop=True),test.loc[test.cert_ws,c],bins[c])
        pooled[c]=z
    logic={k:rule_stats(test,*v) for k,v in logic_rules(test).items()}
    byclass={}
    for c,g in test.groupby('class'):
        filled=g.filled_quantity.to_numpy()>0;y=g.value_cents.to_numpy(float);q=g.filled_quantity.to_numpy(float)
        byclass[c]=dict(population=population(g),edge_median=float(g.o_edge.median()),
            retrospective_adverse_move_median=float(np.median(g.o_edge.to_numpy()[filled]-y[filled]/q[filled])) if filled.any() else None)
    # Diagnostic traces: fixed rank positions within each class/day, never signal-selected.
    casecols=KEY+['class','filled_quantity','value_cents','o_edge','o_depth','o_q_frac','pf_age','bn_age','bnd_age',
        'dfv250','gap_adv','sens','adv_bn_r50','adv_ofi250','adv_pf_imb1','fair_deficit','fair_shock_over_edge','cert_ws']
    cases=[]
    for day in TEST:
        d=panels[day]
        for kind in ['good','bad']:
            q=d[d['class']==kind].sort_values('value_cents')
            if len(q):cases.append(q.iloc[np.unique(np.floor(np.array([.1,.5,.9])*(len(q)-1)).astype(int))][casecols])
    pd.concat(cases,ignore_index=True).to_csv(out/'cases.csv',index=False)
    raw_check=read(RAW/'s5check2_20261115.json')
    for p,h in hashes.items():require(sha(p)==h,'input changed during run')
    result=dict(asof=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip(),
        controls=controls,input_sha256=hashes,lineage=lineage,statuses=statuses,
        population=dict(dev=population(dev),test=population(test),strict_test=population(test[test.cert_ws]),by_class=byclass),
        features=names,dev_edges=bins,dev_directions=direction,dev=dev_stats,pooled=pooled,rules=logic,
        prior_raw_recomputation=dict(asof=raw_check.get('asof'),path=str(RAW/'s5check2_20261115.json'),sha256=sha(RAW/'s5check2_20261115.json')),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
        interpretation='Input associations, no model fitted. All 83 inputs plus five predefined scores; direction/bin edges from four consumed early dates; five later consumed dates. Conditional subsets are diagnostic, not training populations. No confidence or causal sufficiency claim.')
    (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(asof=result['asof'],population=result['population'],rules=logic,peak_rss_mib=result['peak_rss_mib'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);run(ap.parse_args().out)
