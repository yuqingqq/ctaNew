"""Controlled training history, all-action calibration and multivariate coverage."""
import argparse
import gc
import json
import time
import lightgbm as lgb
from scipy.spatial import cKDTree
import pf_value_diagnosis as D
import pf_value_bridge_run as R
J,F,C,np,pd,FIT=D.J,D.F,D.C,D.np,D.pd,D.FIT
HEADS=[('full_q','y_q',J.FEATURES,30.),('full_m','y_m',J.FEATURES,30.),
       ('direct_w','y_w',J.FEATURES,30.),('mechanical_q','y_q',J.MECHANICAL,30.),
       ('mechanical_m','y_m',J.MECHANICAL,30.),('full_tail','y_severe',J.FEATURES,1.),
       ('mechanical_tail','y_severe',J.MECHANICAL,1.),('full_fill','fill',J.FEATURES,1.),
       ('mechanical_fill','fill',J.MECHANICAL,1.)]


def selected(train,name):
    z=train[train.day.astype(str).isin(D.HISTORY[name])].copy()
    if name=='H3_MATCHED_N1':
        target=train[train.day.astype(str)=='20261025'].groupby(['kind','side','phase']).size()
        z['_hash']=z.uid.map(D.identity_hash);parts=[]
        for key,g in z.groupby(['kind','side','phase']):parts.append(g.sort_values('_hash').head(int(target.get(key,0))))
        z=pd.concat(parts,ignore_index=True).drop(columns='_hash')
        C.require(z.groupby(['kind','side','phase']).size().equals(target),'matched history counts')
    C.require(not z.uid.duplicated().any(),'duplicate training identity')
    return z


def geometry(train):
    x=train[D.GEOMETRY].to_numpy(float);med=np.nanmedian(x,axis=0)
    scale=np.nanquantile(x,.75,axis=0)-np.nanquantile(x,.25,axis=0)
    return np.where(np.isfinite(med),med,0.),np.where(np.isfinite(scale)&(scale>1e-9),scale,1.)


def transform(a,med,scale):
    x=a[D.GEOMETRY].to_numpy(float);missing=~np.isfinite(x)
    return np.c_[np.arcsinh((np.where(missing,med,x)-med)/scale),missing.astype(float)]


def cells(a):
    depth=pd.cut(a.jm_depth,[-np.inf,0.,2.,np.inf],labels=['at_or_inside','near','deep']).astype(str)
    mid=pd.cut(a.mid,[-np.inf,20.,80.,np.inf],labels=['low','middle','high']).astype(str)
    return a.kind.astype(str)+':'+a.side.astype(str)+':'+a.phase.astype(str)+':'+depth+':'+mid


def coverage(train,evaluation,med,scale):
    out=evaluation[['uid','kind','side','m']].copy().set_index('uid');reference={}
    count=train.assign(cell=cells(train)).groupby('cell').agg(n=('uid','size'),severe=('y_severe','sum'))
    for key,g in evaluation.groupby(['kind','side']):
        tr=train[(train.kind==key[0])&(train.side==key[1])].copy();C.require(len(tr)>32,'insufficient reference geometry')
        tx=transform(tr,med,scale);tree=cKDTree(tx);distance,_=tree.query(transform(g,med,scale),k=1,workers=1)
        anchors=tr.assign(_hash=tr.uid.map(D.identity_hash)).sort_values('_hash').head(500)
        d,ix=tree.query(transform(anchors,med,scale),k=32,workers=1)
        other=tr.m.to_numpy()[ix]!=anchors.m.to_numpy()[:,None]
        C.require(other.any(axis=1).all(),'no distinct-market neighbor')
        ref=np.min(np.where(other,d,np.inf),axis=1);cut=float(np.quantile(ref,.95))
        out.loc[g.uid,'nearest_distance']=distance;out.loc[g.uid,'reference_q95']=cut
        out.loc[g.uid,'outside_reference95']=distance>cut
        reference[str(key)]=dict(n=len(tr),queries=len(anchors),q95=cut)
    cs=cells(evaluation)
    out.loc[evaluation.uid,'cell_n']=cs.map(count.n).fillna(0).to_numpy()
    out.loc[evaluation.uid,'cell_severe_n']=cs.map(count.severe).fillna(0).to_numpy()
    return out.reset_index(),reference


def predict(a,artifacts):
    p={}
    for name,art in artifacts.items():
        C.require(C.sha(art['path'])==art['sha256'],'model changed')
        model=lgb.Booster(model_file=art['path'])
        p[name]=np.asarray(model.predict(a[art['features']].to_numpy(np.float32),num_threads=1))*art['scale']
    return p


def attach(a,p,base=None):
    a=a.copy();sign=1-2*a.side.to_numpy();edge=sign*(a.mid-a.price).to_numpy()
    orientation=np.where(a.kind=='post',1.,-1.)
    for name,pred in p.items():a['raw_'+name]=pred
    for name in ('full','mechanical'):
        q,m=J.bounded(p[name+'_q'],p[name+'_m'],a.known_remaining.to_numpy(),a.mid.to_numpy(),a.side.to_numpy())
        a['q_'+name]=q;a['m_'+name]=m;a['w_'+name]=edge*q+m
    low=np.minimum(-sign*a.price,sign*(100-a.price))*a.known_remaining
    high=np.maximum(-sign*a.price,sign*(100-a.price))*a.known_remaining
    a['w_direct']=np.clip(p['direct_w'],low,high)
    if base is not None:
        b=base.set_index('uid').loc[a.uid];D.check_equal(a.y_value,b.actual,'evaluation target')
        old=(b.pred_focal_edge+b.pred_focal_motion).to_numpy();rest=b.predicted.to_numpy()-old
        a['value_soft']=b.predicted.to_numpy()
        for name in ('full','mechanical','direct'):a['value_'+name]=rest+orientation*a['w_'+name].to_numpy()
    return a


def group_summary(g,pred='predicted',actual='actual'):
    x=g[pred].to_numpy(float);y=g[actual].to_numpy(float)
    return dict(n=len(g),predicted_mean=float(x.mean()),actual_mean=float(y.mean()),
                bias=float((x-y).mean()),rmse=float(np.sqrt(np.mean((x-y)**2))))


def bins(g,pred,actual,edges):
    z=g.copy();z['_bin']=pd.cut(z[pred],edges,include_lowest=True).astype(str)
    return {str(k):group_summary(x,pred,actual) for k,x in z.groupby('_bin',sort=False)}


def calibration(a,training=False):
    result={}
    groups=[('kind',['kind']),('side',['kind','side']),('phase',['kind','phase']),('depth',['kind','depth_band'])]
    a=a.copy();a['fill']=(a.y_q>1e-9).astype(float)
    a['depth_band']=pd.cut(a.jm_depth,[-np.inf,0.,2.,np.inf],labels=['at_or_inside','near','deep']).astype(str)
    for level,fields in groups:
        for key,g in a.groupby(fields):
            d={}
            for name in ('full','mechanical'):
                q=g['q_'+name];m=g['m_'+name];p=g['raw_'+name+'_fill']
                d[name]=dict(n=len(g),fill=FIT.binary_metrics(g.fill,p),tail=FIT.binary_metrics(g.y_severe,g['raw_'+name+'_tail']),
                    quantity=group_summary(g,'q_'+name,'y_q'),motion=group_summary(g,'m_'+name,'y_m'),
                    focal_wealth=group_summary(g,'w_'+name,'y_w'),
                    predicted_shares_per_fill=float(q.sum()/p.sum()) if p.sum()>1e-12 else None,
                    actual_shares_per_fill=float(g.y_q.sum()/g.fill.sum()) if g.fill.sum() else None,
                    predicted_fill_weighted_move=float(m.sum()/q.sum()) if q.sum()>1e-12 else None,
                    actual_fill_weighted_move=float(g.y_m.sum()/g.y_q.sum()) if g.y_q.sum()>1e-12 else None,
                    probability_quantity_incoherent_n=int((q>p*g.known_remaining+1e-7).sum()))
                if level in ('kind','side'):
                    d[name]['fill_bins']=bins(g,'raw_'+name+'_fill','fill',D.PROB_BINS)
                    d[name]['tail_bins']=bins(g,'raw_'+name+'_tail','y_severe',D.PROB_BINS)
                    d[name]['focal_bins']=bins(g,'w_'+name,'y_w',D.VALUE_BINS)
            d['direct']=dict(focal_wealth=group_summary(g,'w_direct','y_w'))
            result[level+':'+str(key)]=d
    return dict(role='in_sample_training' if training else 'consumed_later_actions',groups=result)


def economic_report(a):
    rows=[]
    for case in ('soft','full','mechanical','direct'):
        z=a[['uid','kind','side','m','y_value','phase','nearest_distance','outside_reference95','cell_n']].copy()
        z['predicted']=a['value_'+case].to_numpy();z=z.rename(columns={'y_value':'actual'});z['case']=case;rows.append(z)
    values=pd.concat(rows,ignore_index=True);metrics={};grouped={}
    for (case,kind),g in values.groupby(['case','kind']):metrics[case+':'+kind]=FIT.value_metrics(g)
    for (case,kind,side),g in values.groupby(['case','kind','side']):grouped[f'{case}:{kind}:{side}']=FIT.value_metrics(g)
    cal={};cover={}
    for (case,kind),g in values.groupby(['case','kind']):
        cal[case+':'+kind]=bins(g,'predicted','actual',D.VALUE_BINS)
        cover[case+':'+kind]={str(k):group_summary(z) for k,z in g.groupby('outside_reference95')}
    return values,dict(n=a.uid.nunique(),metrics=metrics,by_side=grouped,value_bins=cal,coverage_groups=cover)


def selftest():
    frame=pd.DataFrame(dict(predicted=[1.,1.,-1.,-1.],actual=[2.,0.,-2.,0.]))
    cal=bins(frame,'predicted','actual',D.VALUE_BINS)
    C.require(all(abs(x['bias'])<1e-9 for x in cal.values()),'calibrated group positive control')
    bad=frame.copy();bad['predicted']+=10.
    C.require(group_summary(bad)['bias']==10.,'miscalibration detector')
    tree=cKDTree([[0.,0.],[1.,1.]])
    C.require(tree.query([[0.,0.]])[0][0]==0 and tree.query([[10.,10.]])[0][0]>10,'novelty positive control')
    return dict(group_calibration_positive=True,miscalibration_detected=True,novelty_controls=True,
        corrupt_number_refused=FIT.CHECK.refused(lambda:D.check_equal([1.],[2.],'bad'),'bad equality'))


def run():
    D.verify();controls=selftest();start=time.time();out=D.OUT/'calibration';C.require(not out.exists(),'calibration exists');out.mkdir(parents=True)
    frames=[];source={};statuses={}
    for day in F.TRAIN:
        a,r=FIT.read(day);frames.append(a);source[day]=C.sha(J.OUT/day/'receipt.json');statuses[day]=r['action_statuses']
    train=pd.concat(frames,ignore_index=True);train['fill']=(train.y_q>1e-9).astype(float);med,scale=geometry(train)
    original=json.loads((J.OUT/'fit/receipt.json').read_text());summary={}
    evaluations={};bases={}
    for day in (F.CAL,F.TEST):
        a,r=FIT.read(day);evaluations[day]=a;source[day]=C.sha(J.OUT/day/'receipt.json');statuses[day]=r['action_statuses']
        D.checked_receipt(J.D.OUT/day);b=pd.read_parquet(J.D.OUT/day/'values.parquet')
        bases[day]=b[(b.case=='SOFT_JOINT')&(b.channel=='ordinary')]
    for history in D.HISTORY:
        z=selected(train,history);dest=out/history;dest.mkdir();artifacts={}
        z[['uid','day','kind','side','phase']].to_parquet(dest/'training_ids.parquet',index=False)
        for name,target,features,unit in HEADS:
            if history=='H3' and name in original['artifacts']:artifacts[name]=original['artifacts'][name];continue
            path=dest/(name+'.txt');y=z[target].to_numpy(float)/unit
            model=J.FM.fit(z[features].to_numpy(np.float32),y,features,path,'binary' if target in ('fill','y_severe') else 'regression')
            artifacts[name]=dict(path=str(path),sha256=C.sha(path),features=features,scale=unit,n=len(z),target=target)
            del model;gc.collect();C.guard('history fit')
        trained=attach(z,predict(z,artifacts));C.dump(dest/'training_calibration.json',dict(asof=C.stamp(),n=len(z),**calibration(trained,True)))
        days={}
        for day,a in evaluations.items():
            heads=attach(a,predict(a,artifacts),bases[day]);cov,ref=coverage(z,a,med,scale)
            heads=heads.merge(cov.drop(columns=['kind','side','m']),on='uid',validate='one_to_one')
            if history=='H3':
                D.checked_receipt(J.OUT/('eval_'+day));old=pd.read_parquet(J.OUT/('eval_'+day)/'heads.parquet').set_index('uid').loc[heads.uid]
                for name in original['artifacts']:
                    if name in artifacts:D.check_equal(heads['raw_'+name],old['raw_'+name],'three-day saved predictions')
            heads.to_parquet(dest/(day+'_heads.parquet'),index=False)
            values,economic=economic_report(heads);values.to_parquet(dest/(day+'_values.parquet'),index=False)
            report=dict(asof=C.stamp(),history=history,day=day,n=len(a),economic=economic,calibration=calibration(heads),
                coverage=dict(reference=ref,far_n=int(heads.outside_reference95.sum()),empty_cell_n=int((heads.cell_n==0).sum()),
                    median_cell_n=float(heads.cell_n.median()),median_cell_severe_n=float(heads.cell_severe_n.median())),
                source_statuses=statuses[day])
            C.dump(dest/(day+'_report.json'),report);days[day]=dict(n=len(a),far_n=report['coverage']['far_n'],metrics=economic['metrics'])
        C.dump(dest/'receipt.json',dict(asof=C.stamp(),n=len(z),training_days=D.HISTORY[history],artifacts=artifacts,outputs=D.files(dest),source_manifest=C.sha(D.FREEZE)))
        summary[history]=dict(n=len(z),days=days)
        print(json.dumps(dict(stage='history',history=history,n=len(z),seconds=round(time.time()-start))),flush=True)
    comparisons={}
    for day in evaluations:
        for kind in ('cancel','post'):
            v={h:summary[h]['days'][day]['metrics']['full:'+kind]['rmse_cents'] for h in D.HISTORY}
            comparisons[day+':'+kind]=dict(rmse=v,monotonic_history_improvement=v['H3']<v['H2']<v['H1'],
                matched_broader_history_improves=v['H3_MATCHED_N1']<v['H1'],full_more_rows_improves=v['H3']<v['H3_MATCHED_N1'])
    C.dump(out/'report.json',dict(asof=C.stamp(),controls=controls,histories=summary,comparisons=comparisons,source_statuses=statuses,
        geometry=dict(features=D.GEOMETRY,median=med.tolist(),scale=scale.tolist()),limits=D.verify()['limits']))
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(D.FREEZE),inputs=source,
        children={h:C.sha(out/h/'receipt.json') for h in D.HISTORY},outputs=D.files(out),seconds=time.time()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','selftest','freeze']);a=p.parse_args()
    if a.mode=='freeze':D.freeze()
    elif a.mode=='selftest':print(json.dumps(selftest()))
    else:run()
