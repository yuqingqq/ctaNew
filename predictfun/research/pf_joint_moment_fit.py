"""Pinned joint-moment prediction screen; no controller or live execution changes."""
import argparse
import gc
import json
import time
import pf_joint_moment as J
import pf_value_bridge_run as R
import pf_flow_control_checks as CHECK
F,C,np,pd,I,D=J.F,J.C,J.np,J.pd,J.I,J.D


def read(day):
    src=J.OUT/day;r=json.loads((src/'receipt.json').read_text())
    C.require(r['source_manifest']==C.sha(J.FREEZE) and not r['mismatches'],'invalid label receipt')
    C.require(C.sha(src/'actions.parquet')==r['outputs']['actions.parquet'],'data changed')
    a=pd.read_parquet(src/'actions.parquet');return a[a.status=='eligible'].copy(),r


def feature_contract(names):
    C.require(set(names)<=set(J.FEATURES) and len(set(names))==len(names),'feature whitelist')
    C.require(not any(n.startswith(('actual_','y_','pred_')) for n in names),'target feature leak')


def binary_metrics(y,p):
    y=np.asarray(y,float);p=np.asarray(p,float)
    C.require(np.isin(y,[0.,1.]).all() and np.isfinite(p).all() and ((0<=p)&(p<=1)).all(),'binary metric contract')
    n1=int(y.sum());n0=len(y)-n1;rank=pd.Series(p).rank(method='average').to_numpy()
    return dict(n=len(y),positives=n1,brier=float(np.mean((p-y)**2)),
        logloss=float(-np.mean(y*np.log(np.clip(p,1e-12,1))+(1-y)*np.log(np.clip(1-p,1e-12,1)))),
        auc=float((rank[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n0 and n1 else None,
        mean_predicted=float(p.mean()),actual_rate=float(y.mean()))


def value_metrics(g):
    error=g.predicted-g.actual;tail=np.where(g.kind=='post',1.,-1.)*g.actual.to_numpy() < -30.
    return dict(n=len(g),rmse_cents=float(np.sqrt(np.mean(error**2))),mae_cents=float(error.abs().mean()),
        zero_prediction_rmse_cents=float(np.sqrt(np.mean(g.actual**2))),spearman=float(g.predicted.corr(g.actual,method='spearman')),
        mean_prediction=float(g.predicted.mean()),mean_actual=float(g.actual.mean()),severe_n=int(tail.sum()),
        severe_rmse_cents=float(np.sqrt(np.mean(error.to_numpy()[tail]**2))) if tail.any() else None,
        positive_n=int((g.predicted>0).sum()),selected_actual_sum_cents=float(g.loc[g.predicted>0,'actual'].sum()),
        matched_null=R.matched_null(g))


def selftest():
    feature_contract(J.FEATURES);feature_contract(J.MECHANICAL)
    q,m=J.bounded(np.array([-1.,5.,50.]),np.array([10.,-1000.,1000.]),np.array([30.,30.,10.]),np.array([50.,10.,90.]),np.array([0,0,1]))
    C.require(np.array_equal(q,[0.,5.,10.]) and np.array_equal(m,[0.,-50.,900.]),'physical moment bounds')
    C.require(binary_metrics([0,1],[0.,1.])['auc']==1 and binary_metrics([0,1],[1.,0.])['auc']==0,'tail score positive control')
    qa=np.array([0.,30.]);ma=np.array([5.,-5.]);mb=-ma
    C.require(qa.mean()*ma.mean()==qa.mean()*mb.mean() and (qa*ma).mean()!=(qa*mb).mean(),'joint target dependence')
    W=I.M.toy_lane_window([],[],[],n=40)
    for s,w in W.items():w['ws']=0.;w['kf']=w['tf']+10.;w['jk']=np.searchsorted(w['kf'],w['tg'],side='right')-1
    import copy
    o=dict(pt=50.,ts=0.,ta=58.,tc=1e30,tr=1e30,ref=False,rung=0,fl=[(100.,5.),(180.,7.)],rem=18.,ahead=0.,ci=0)
    ST={s:dict(cl=[],cur=[None]*6,orders=[[] for _ in range(6)]) for s in (0,1)}
    ST[0].update(cl=[o],cur=[o]+[None]*5,orders=[[o]]+[[] for _ in range(5)])
    state=J.V.P.snapshot(W,5,250.,ST,5.);market=np.zeros(len(C.MARKET))
    x=J.features(W,5,250.,state,0,market,100.);latent=J.label_state(W,5,250.,ST,5.,state)
    C.require(state['orders'][0,3]==25. and latent['orders'][0,3]==18. and latent['position']==12.,'latent/current state separation')
    changed=copy.deepcopy(ST);changed[0]['cl'][0]['fl'].append((249.,2.));changed[0]['cl'][0]['rem']=1.
    after=J.V.P.snapshot(W,5,250.,changed,5.);y=J.features(W,5,250.,after,0,market,100.)
    C.require(np.array_equal(x,y,equal_nan=True),'unreceived fills entered features')
    return dict(joint_target_detects_dependence=True,moment_bounds=True,tail_score_controls=True,
        latent_state_labels_only=True,unreceived_fill_inputs_invariant=True,
        leaked_feature_refused=CHECK.refused(lambda:feature_contract(J.FEATURES+['y_m']),'label input'),
        invalid_tail_probability_refused=CHECK.refused(lambda:binary_metrics([0,1],[-1.,2.]),'invalid tail probability'))


def fit():
    J.verify();selftest();start=time.time();out=J.OUT/'fit';C.require(not out.exists(),'fit exists');out.mkdir(parents=True)
    frames=[];inputs={}
    for day in F.TRAIN:
        a,r=read(day);frames.append(a);inputs[day]=C.sha(J.OUT/day/'receipt.json')
    train=pd.concat(frames,ignore_index=True);C.require(not train.uid.duplicated().any(),'duplicate training action')
    C.require(set(train.day.astype(str))==set(F.TRAIN),'training chronology');artifacts={};importance={}
    configs=[('full_q','y_q',J.FEATURES,30.),('full_m','y_m',J.FEATURES,30.),('direct_w','y_w',J.FEATURES,30.),
        ('mechanical_q','y_q',J.MECHANICAL,30.),('mechanical_m','y_m',J.MECHANICAL,30.),
        ('full_tail','y_severe',J.FEATURES,1.),('mechanical_tail','y_severe',J.MECHANICAL,1.),
        ('remaining','y_remaining',J.FEATURES,1.)]
    for name,target,features,scale in configs:
        feature_contract(features);z=train[train.kind=='cancel'] if name=='remaining' else train
        X=z[features].to_numpy(np.float32);y=z[target].to_numpy(float)/scale
        C.require(np.isfinite(y).all() and not np.isinf(X).any(),'invalid fit values')
        path=out/(name+'.txt');model=J.FM.fit(X,y,features,path,'binary' if target=='y_severe' else 'regression')
        artifacts[name]=dict(path=str(path),sha256=C.sha(path),features=features,scale=scale,n=len(z),target=target)
        importance[name]=dict(zip(features,map(float,model.feature_importance(importance_type='gain'))))
        del X,y,z,model;gc.collect();C.guard('moment fit')
        print(json.dumps(dict(stage='fit',head=name,seconds=round(time.time()-start))),flush=True)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(J.FREEZE),training_days=F.TRAIN,inputs=inputs,artifacts=artifacts,
        n=len(train),counts=train.groupby(['day','kind','side']).size().to_dict(),
        zero_affected_fills=int((train.y_q<=1e-9).sum()),severe=int(train.y_severe.sum()),
        unreceived_fill_cancel_n=int(((train.kind=='cancel')&(train.y_remaining<1.-1e-9)).sum()),
        feature_min=train[J.FEATURES].min().to_dict(),feature_max=train[J.FEATURES].max().to_dict(),importance=importance,seconds=time.time()-start))


def evaluate(day):
    J.verify();a,receipt=read(day);r=json.loads((J.OUT/'fit/receipt.json').read_text());C.require(r['source_manifest']==C.sha(J.FREEZE),'fit contract')
    start=time.time();src=D.OUT/day;dr=json.loads((src/'receipt.json').read_text())
    C.require(C.sha(src/'values.parquet')==dr['outputs']['values.parquet'],'baseline changed')
    base=pd.read_parquet(src/'values.parquet');base=base[(base.case=='SOFT_JOINT')&(base.channel=='ordinary')].set_index('uid')
    a=a.set_index('uid');C.require(set(a.index)==set(base.index),'evaluation support changed');base=base.loc[a.index]
    C.require(np.allclose(a.y_value,base.actual,rtol=0,atol=1e-7),'evaluation target identity')
    predictions={}
    import lightgbm as lgb
    for name,art in r['artifacts'].items():
        C.require(C.sha(art['path'])==art['sha256'],'model changed '+name)
        model=lgb.Booster(model_file=art['path']);x=a[art['features']].to_numpy(np.float32)
        predictions[name]=np.asarray(model.predict(x,num_threads=1))*art['scale'];del model,x
    q,m=J.bounded(predictions['full_q'],predictions['full_m'],a.known_remaining.to_numpy(),a.mid.to_numpy(),a.side.to_numpy())
    qm,mm=J.bounded(predictions['mechanical_q'],predictions['mechanical_m'],a.known_remaining.to_numpy(),a.mid.to_numpy(),a.side.to_numpy())
    sign=1-2*a.side.to_numpy();edge=sign*(a.mid-a.price).to_numpy();orientation=np.where(a.kind=='post',1.,-1.)
    low=np.minimum(-sign*a.price.to_numpy(),sign*(100.-a.price.to_numpy()))*a.known_remaining.to_numpy()
    high=np.maximum(-sign*a.price.to_numpy(),sign*(100.-a.price.to_numpy()))*a.known_remaining.to_numpy()
    direct=np.clip(predictions['direct_w'],low,high);fraction=np.where(a.kind=='cancel',np.clip(predictions['remaining'],0.,1.),1.)
    old_focal=(base.pred_focal_edge+base.pred_focal_motion).to_numpy();rest=base.predicted.to_numpy()-old_focal
    values=dict(SOFT_JOINT=base.predicted.to_numpy(),JOINT_MOMENT=rest+orientation*(edge*q+m),
        MECHANICAL_MOMENT=rest+orientation*(edge*qm+mm),DIRECT_WEALTH=rest+orientation*direct,REMAINDER_ONLY=rest+old_focal*fraction)
    detail=a.copy();detail['baseline_predicted']=base.predicted.to_numpy()
    for name,value in predictions.items():detail['raw_'+name]=value
    detail['q_full']=q;detail['m_full']=m;detail['q_mechanical']=qm;detail['m_mechanical']=mm;detail['remaining_fraction']=fraction
    frames=[]
    for case,pred in values.items():
        frame=base[['day','m','side','rung','generation','kind','decision_ms','actual']].copy()
        frame['case']=case;frame['predicted']=pred;frames.append(frame.reset_index())
    output=pd.concat(frames,ignore_index=True);dest=J.OUT/('eval_'+day);C.require(not dest.exists(),'evaluation exists');dest.mkdir()
    output.to_parquet(dest/'values.parquet',index=False);detail.reset_index().to_parquet(dest/'heads.parquet',index=False)
    C.dump(dest/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(J.FREEZE),fit_receipt_sha256=C.sha(J.OUT/'fit/receipt.json'),
        data_receipt_sha256=C.sha(J.OUT/day/'receipt.json'),baseline_receipt_sha256=C.sha(src/'receipt.json'),
        outputs={p.name:C.sha(p) for p in dest.glob('*.parquet')},seconds=time.time()-start))
    report(day)


def report(day):
    src=J.OUT/('eval_'+day);r=json.loads((src/'receipt.json').read_text())
    for p,h in r['outputs'].items():C.require(C.sha(src/p)==h,'evaluation artifact changed')
    g=pd.read_parquet(src/'values.parquet');a=pd.read_parquet(src/'heads.parquet');metrics={};by_side={};tails={};moments={}
    for (case,kind),z in g.groupby(['case','kind']):metrics[case+':'+kind]=value_metrics(z)
    for (case,kind,side),z in g.groupby(['case','kind','side']):by_side[f'{case}:{kind}:{side}']=value_metrics(z)
    for kind,z in a.groupby('kind'):
        for variant in ('full','mechanical'):
            tails[variant+':'+kind]=binary_metrics(z.y_severe,z['raw_'+variant+'_tail'])
            moments[variant+':'+kind]=dict(n=len(z),quantity_mae_shares=float((z['q_'+variant]-z.y_q).abs().mean()),
                motion_rmse_cents=float(np.sqrt(np.mean((z['m_'+variant]-z.y_m)**2))))
    z=a[a.kind=='cancel'];err=z.remaining_fraction-z.y_remaining;naive=1.-z.y_remaining;hidden=z.y_remaining<1.-1e-9
    remainder=dict(n=len(z),unreceived_n=int(hidden.sum()),model_rmse_fraction=float(np.sqrt(np.mean(err**2))),
        known_upper_bound_rmse_fraction=float(np.sqrt(np.mean(naive**2))),model_mae_fraction=float(err.abs().mean()),
        model_hidden_rmse=float(np.sqrt(np.mean(err[hidden]**2))) if hidden.any() else None,
        naive_hidden_rmse=float(np.sqrt(np.mean(naive[hidden]**2))) if hidden.any() else None)
    fr=json.loads((J.OUT/'fit/receipt.json').read_text());coverage={}
    for name in J.FEATURES:
        x=a[name].to_numpy();lo=fr['feature_min'][name];hi=fr['feature_max'][name]
        coverage[name]=dict(missing=int((~np.isfinite(x)).sum()),below=int((x<lo).sum()) if lo is not None else None,above=int((x>hi).sum()) if hi is not None else None)
    C.dump(src/'report.json',dict(asof=C.stamp(),day=day,n=a.uid.nunique(),metrics=metrics,by_side=by_side,tail_scores=tails,
        moment_scores=moments,remainder=remainder,coverage=coverage,receipt_sha256=C.sha(src/'receipt.json'),limits=J.verify()['limits']))
    print(json.dumps(C.clean(dict(day=day,metrics=metrics,tail_scores=tails,remainder=remainder))),flush=True)


def audit():
    J.verify();checks=selftest();files=['fit/receipt.json'];population={};screen={}
    for day in F.TRAIN+(F.CAL,F.TEST):
        a,r=read(day);population[day]=dict(n=len(a),statuses=r['action_statuses'],canonical_checked=r['canonical_checked'],canonical_positive=r['canonical_positive'])
        C.require(r['canonical_checked']>0 and not r['mismatches'],'label checks missing')
        C.require(np.allclose(a.y_w,a.jm_edge*a.y_q+a.y_m,rtol=0,atol=2e-5),'moment label algebra')
        C.require((a.y_q>=-1e-7).all() and (a.y_q<=a.known_remaining+1e-7).all(),'label quantity bound')
        files.append(day+'/receipt.json')
    for day in (F.CAL,F.TEST):
        src=J.OUT/('eval_'+day);r=json.loads((src/'receipt.json').read_text())
        for p,h in r['outputs'].items():C.require(C.sha(src/p)==h,'result artifact changed')
        g=pd.read_parquet(src/'values.parquet');rep=json.loads((src/'report.json').read_text())
        for (case,kind),z in g.groupby(['case','kind']):C.require(C.clean(value_metrics(z))==rep['metrics'][case+':'+kind],'metric/null mismatch')
        for (case,kind,side),z in g.groupby(['case','kind','side']):C.require(C.clean(value_metrics(z))==rep['by_side'][f'{case}:{kind}:{side}'],'side metric mismatch')
        heads=pd.read_parquet(src/'heads.parquet')
        for kind,z in heads.groupby('kind'):
            for variant in ('full','mechanical'):
                C.require(C.clean(binary_metrics(z.y_severe,z['raw_'+variant+'_tail']))==rep['tail_scores'][variant+':'+kind],'tail metric mismatch')
        for kind in ('cancel','post'):
            a=rep['metrics']['JOINT_MOMENT:'+kind];b=rep['metrics']['SOFT_JOINT:'+kind];m=rep['metrics']['MECHANICAL_MOMENT:'+kind]
            screen[day+':'+kind]=dict(n=a['n'],value_improved=a['rmse_cents']<b['rmse_cents'],
                tail_no_worse=a['severe_rmse_cents']<=b['severe_rmse_cents'],market_beats_mechanical=a['rmse_cents']<m['rmse_cents'])
        files += ['eval_'+day+'/'+n for n in ('receipt.json','report.json')]
    result=dict(asof=C.stamp(),controls=checks,populations=population,comparisons=screen,
        value_tail_screen_passed=all(x['value_improved'] and x['tail_no_worse'] for x in screen.values()),
        market_incremental_screen_passed=all(x['market_beats_mechanical'] for x in screen.values()),
        role='Consumed prediction screen only; no controller integration or promotion',source_manifest=C.sha(J.FREEZE))
    C.dump(J.REVIEW/'audit_v1.json',result)
    for name in files:
        p=J.REVIEW/name;C.require(not p.exists(),'review output exists');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((J.OUT/name).read_bytes())
    C.dump(J.REVIEW/'index.json',dict(asof=C.stamp(),source_manifest=C.sha(J.FREEZE),outputs={p:C.sha(J.REVIEW/p) for p in files+['audit_v1.json']},limits=J.verify()['limits']))
    print(json.dumps(C.clean(result)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['selftest','freeze','fit','evaluate','audit']);p.add_argument('day',nargs='?');a=p.parse_args()
    if a.mode=='freeze':J.freeze()
    elif a.mode=='selftest':print(json.dumps(selftest()),flush=True)
    elif a.mode=='fit':fit()
    elif a.mode=='evaluate':evaluate(a.day)
    else:audit()
