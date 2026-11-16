"""Chronological residual distillation and delayed replay for entry anticipation."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_entry_arrival as A
np,pd,lgb=A.np,A.pd,A.lgb
L,P,V,G=A.L,A.P,A.V,A.G


def rows(day,root):
    d=root/'data'/day;j=json.loads((d/'receipt.json').read_text())
    L.require(not j['smoke_only'] and j['features']==V.FULL_IN and L.sha(d/'rows.npz')==j['rows_sha256'],'data identity')
    z=np.load(d/'rows.npz',allow_pickle=False)
    return {k:z[k] for k in z.files},j


def metrics(y,p):
    L.require(len(y)==len(p) and len(y)>0 and np.isfinite(y).all() and np.isfinite(p).all(),'metric population')
    wrong=(y<0)!=(p<0)
    return dict(n=len(y),mae=float(np.mean(np.abs(y-p))),mse=float(np.mean((y-p)**2)),
        sign_accuracy=float(np.mean(~wrong)),teacher_positive=int((y>=0).sum()),predicted_positive=int((p>=0).sum()),
        teacher_value_weighted_disagreement_cents=float(np.sum(np.abs(y)*wrong)),
        false_post_cost_cents=float(np.sum(-y[(y<0)&(p>=0)])),false_wait_cost_cents=float(np.sum(y[(y>0)&(p<0)])))


def controls():
    y=np.array([-10.,1.,0.]);good=metrics(y,y);bad=metrics(y,-y)
    L.require(good['teacher_value_weighted_disagreement_cents']==0 and bad['teacher_value_weighted_disagreement_cents']==11,'metric positive/negative control')
    caught=False
    try:metrics(y,np.array([np.nan,1.,0.]))
    except ValueError:caught=True
    L.require(caught,'nonfinite prediction not refused')
    return dict(perfect_score_control=True,reversed_score_control=True,nonfinite_refused=True)


def fit(day,root=A.ROOT):
    out=root/'fit'/day;L.require(day in L.DAYS and not out.exists(),'fit day/output');out.mkdir(parents=True)
    start=time.time();checks=controls();heads,tfit=A.teacher(day);train=tfit['train_days'];A.test_chronology(train,day)
    chunks=[];ys=[];base=[];ids=[];receipts={}
    for d in train:
        z,j=rows(d,root);receipts[d]=L.sha(root/'data'/d/'receipt.json')
        y=A.future_scores(heads,z['future'],z['status']);b=A.score(heads,z['current']);ok=np.isfinite(y)&(z['status']!='missing_current_inputs')
        chunks.append(z['current'][ok]);ys.append(y[ok]);base.append(b[ok]);ids.append(z['ids'][ok]);del z;gc.collect()
    x=np.concatenate(chunks);y=np.concatenate(ys);b=np.concatenate(base);ii=np.concatenate(ids)
    del chunks,ys,base,ids;gc.collect()
    xt=A.student_matrix(x,b);models={};info={}
    params=dict(V.PARAMS,objective='regression',metric='l2',num_threads=1)
    params.pop('is_unbalance',None);params.pop('scale_pos_weight',None)
    # Parameters/rounds are inherited without a per-experiment search.
    for side in (0,1):
        sel=ii[:,1]==side;L.require(sel.any(),'missing training side')
        bst=lgb.train(params,lgb.Dataset(xt[sel],label=(y-b)[sel],feature_name=list(A.FEATURES)),num_boost_round=V.ROUNDS)
        p=out/f'side{side}.txt';bst.save_model(str(p));loaded=lgb.Booster(model_file=str(p))
        sample=xt[np.flatnonzero(sel)[::max(1,int(sel.sum())//1000)]]
        err=float(np.max(np.abs(bst.predict(sample,num_threads=1)-loaded.predict(sample,num_threads=1))))
        L.require(err<=1e-9,'saved student changed predictions');models[side]=loaded
        info[str(side)]=dict(n_train=int(sel.sum()),path=str(p),sha256=L.sha(p),reload_max_error=err)
    del x,xt,ii,y,b;gc.collect()
    z,j=rows(day,root);receipts[day]=L.sha(root/'data'/day/'receipt.json')
    y=A.future_scores(heads,z['future'],z['status']);b=A.score(heads,z['current']);pred=np.full(len(b),np.nan)
    ok=np.isfinite(y)&(z['status']!='missing_current_inputs');xx=A.student_matrix(z['current'],b)
    result={}
    for side in (0,1):
        side_mask=z['ids'][:,1]==side;pred[side_mask]=b[side_mask]+models[side].predict(xx[side_mask],num_threads=1)
        sel=side_mask&ok;result[str(side)]={'persistence':metrics(y[sel],b[sel]),'forecast':metrics(y[sel],pred[sel])}
    result['pooled']={'persistence':metrics(y[ok],b[ok]),'forecast':metrics(y[ok],pred[ok])}
    np.savez_compressed(out/'test_predictions.npz',target=y,baseline=b,forecast=pred,ids=z['ids'],status=z['status'])
    receipt=dict(asof=A.stamp(),test_day=day,train_days=train,unused_calibration_day=tfit['cal'],source_commit=A.commit(),
        source_sha256=L.sha(__file__),features=A.FEATURES,params=params,rounds=V.ROUNDS,heads=info,teacher_fit_sha256=L.sha(Path(V.OUTD)/f'fit3_{day}.json'),
        data_receipts=receipts,test_statuses=dict(Counter(z['status'])),metrics=result,controls=checks,
        predictions_sha256=L.sha(out/'test_predictions.npz'),seconds=time.time()-start,
        limits='Teacher-score imitation on neutral placements, not realized fill value or untouched validation. Separate bid/ask entry only; no threshold selection. Missing target support excluded with status; activation rejections retained as zero.')
    A.dump(out/'receipt.json',receipt);print(json.dumps(dict(day=day,metrics=result,seconds=receipt['seconds'])),flush=True)


def fresh_scores(heads,students,W,X):
    keys,x,_,_=A.vectors(W,X);b=A.score(heads,x);pred=b.copy()
    for s in (0,1):
        sel=keys[:,0]==s
        if sel.any():pred[sel]+=students[s].predict(A.student_matrix(x[sel],b[sel]),num_threads=1)
    out={n:np.full((2,P.NR,W[0]['n']),np.nan) for n in ['baseline','forecast']}
    for k,bb,pp in zip(keys,b,pred):out['baseline'][tuple(k)]=bb;out['forecast'][tuple(k)]=pp
    return out


class EntryPolicy(L.Policy):
    """Only entry changes. Resting order decisions call the original full-EV head."""
    def __init__(self,heads,W,X,entry):
        super().__init__(heads,W,X,'EV',(0.,1.),Counter(),fresh=entry)


def replay(day,root=A.ROOT):
    out=root/'replay'/day;L.require(day in L.DAYS and not out.exists(),'replay day/output');out.mkdir(parents=True)
    start=time.time();checks=controls();heads,tfit=A.teacher(day)
    fit_path=root/'fit'/day/'receipt.json';f=json.loads(fit_path.read_text())
    L.require(tuple(f['features'])==A.FEATURES and f['teacher_fit_sha256']==L.sha(Path(V.OUTD)/f'fit3_{day}.json'),'fit contract')
    students={}
    for s in (0,1):
        a=f['heads'][str(s)];L.require(L.sha(a['path'])==a['sha256'],'student changed');students[s]=lgb.Booster(model_file=a['path'])
    data_receipt=json.loads((root/'data'/day/'receipt.json').read_text())
    L.require(not data_receipt['smoke_only'],'smoke oracle')
    old_path=Path(A.K.IO)/'ev_lag_gap_20261116_v1'/day/'windows.parquet'
    old_receipt=json.loads((G.HERE.parent/'reviews/ev_lag_gap_20261116/results_v1/report.json').read_text())
    L.require(L.sha(old_path)==old_receipt['window_hashes'][day],'prior median windows changed')
    old=pd.read_parquet(old_path);old=old[old.cfg=='medians_58_37'].set_index('m')
    D,windows,_=A.load(day);counts=Counter();output=[]
    for W,X,SR,IM in windows():
        m=int(W[0]['m']);fresh=fresh_scores(heads,students,W,X)
        op=root/'data'/day/'oracle'/f'{m}.npz';L.require(L.sha(op)==data_receipt['oracle_hashes'][op.name],'oracle cache changed')
        oracle=np.load(op)['score'];fresh['arrival_oracle']=np.where(np.isfinite(oracle),oracle,fresh['baseline'])
        counts['oracle_fallback_candidates']+=int((~np.isfinite(oracle)&np.isfinite(fresh['baseline'])).sum())
        with G.clocks(A.PLACE,A.CANCEL):
            for name in ['baseline','arrival_oracle','forecast']:
                pol=EntryPolicy(heads,W,X,fresh[name]);r=L.simulate(W,{(s,d):pol.dec(s,d) for s in (0,1) for d in range(P.NR)},
                    {(s,d):pol.gate(s,d) for s in (0,1) for d in range(P.NR)},50.)
                v,live=L.value(W,r,SR,IM);row=dict(day=day,m=m,arm=name,**v,**{f'live{j}':a for j,a in enumerate(live)})
                if name=='baseline':G.parity(row,old.loc[m]);counts['baseline_window_parity']+=1
                if counts['windows']==0 and name=='baseline':
                    ref=L.Policy(heads,W,X,'EV',(0.,1.),Counter())
                    rr=L.simulate(W,{(s,d):ref.dec(s,d) for s in (0,1) for d in range(P.NR)},
                        {(s,d):ref.gate(s,d) for s in (0,1) for d in range(P.NR)},50.)
                    L.same_trajectory(r,rr);counts['baseline_trajectory_parity']+=1
                output.append(row);counts['replays']+=1
        counts['windows']+=1
        if counts['windows']%36==0:
            print(json.dumps(dict(day=day,stats=dict(counts),seconds=round(time.time()-start),rss_mib=A.K.rss_mib())),flush=True);A.E.mem_guard('entry arrival replay')
    L.require(counts['baseline_window_parity']==len(old)==counts['windows'],'incomplete replay')
    frame=pd.DataFrame(output);frame.to_parquet(out/'windows.parquet',index=False);totals={}
    for arm,g in frame.groupby('arm'):
        z={k:float(g[k].sum()) for k in ['pnl','shares','orders','pulls','fill_events','markout1_cents','markout5_cents','imputed_pnl','imputed_shares','pend','pend_shares','qunk','qunk_shares']}
        z.update(cents_per_share=100*z['pnl']/z['shares'] if z['shares'] else None,windows=len(g),max_abs_position=float(g.max_abs_position.max()),live_share=float(g[[f'live{j}' for j in range(P.NR)]].to_numpy().mean()))
        totals[arm]=z
    receipt=dict(asof=A.stamp(),day=day,source_commit=A.commit(),source_sha256=L.sha(__file__),counts=dict(counts),controls=checks,totals=totals,
        fit_sha256=L.sha(fit_path),data_receipt_sha256=L.sha(root/'data'/day/'receipt.json'),prior_median_windows_sha256=L.sha(old_path),
        windows_sha256=L.sha(out/'windows.parquet'),seconds=time.time()-start,
        limits='All fills, same 58/37 execution and current resting guard. Arrival oracle uses future zero-feed fresh-order score only; not a full zero-latency policy or attainable bound. Consumed data, no inference/promotion.')
    A.dump(out/'receipt.json',receipt);print(json.dumps(dict(day=day,totals=totals,counts=dict(counts),seconds=receipt['seconds'])),flush=True)


def report(root=A.ROOT):
    out=A.REVIEW/'results_v1';L.require(not out.exists(),'report exists');out.mkdir(parents=True)
    fits={d:json.loads((root/'fit'/d/'receipt.json').read_text()) for d in L.DAYS}
    reps={d:json.loads((root/'replay'/d/'receipt.json').read_text()) for d in L.DAYS}
    data={d:json.loads((root/'data'/d/'receipt.json').read_text()) for d in A.DAYS}
    totals={};mm={};diff={}
    for d,r in reps.items():
        L.require(L.sha(root/'replay'/d/'windows.parquet')==r['windows_sha256'],'replay output changed')
        L.require(L.sha(root/'fit'/d/'receipt.json')==r['fit_sha256'],'replay fit identity')
        L.require(L.sha(root/'data'/d/'receipt.json')==r['data_receipt_sha256'],'replay data identity')
        frame=pd.read_parquet(root/'replay'/d/'windows.parquet')
        L.require(not frame.duplicated(['day','m','arm']).any(),'duplicate replay windows')
        for arm,g in frame.groupby('arm'):
            for k in ['pnl','shares','orders','pulls']:
                L.require(abs(float(g[k].sum())-r['totals'][arm][k])<1e-7,'output reconciliation')
        diff[d]={a:r['totals'][a]['pnl']-r['totals']['baseline']['pnl'] for a in ['arrival_oracle','forecast']}
    for arm in ['baseline','arrival_oracle','forecast']:
        cells=[r['totals'][arm] for r in reps.values()]
        z={k:sum(c[k] for c in cells) for k in ['pnl','shares','orders','pulls','fill_events','markout1_cents','windows']}
        z.update(cents_per_share=100*z['pnl']/z['shares'] if z['shares'] else None,positive_dates=sum(c['pnl']>0 for c in cells))
        totals[arm]=z
    for arm in ['persistence','forecast']:
        cells=[f['metrics']['pooled'][arm] for f in fits.values()];n=sum(c['n'] for c in cells)
        mm[arm]={k:sum(c[k]*c['n'] for c in cells)/n for k in ['mae','mse','sign_accuracy']}
        mm[arm].update(n=n,teacher_value_weighted_disagreement_cents=sum(c['teacher_value_weighted_disagreement_cents'] for c in cells))
    predicate=dict(teacher_disagreement_improves=mm['forecast']['teacher_value_weighted_disagreement_cents']<mm['persistence']['teacher_value_weighted_disagreement_cents'],
        forecast_pnl_improves=totals['forecast']['pnl']>totals['baseline']['pnl'],forecast_better_dates=sum(v['forecast']>0 for v in diff.values()),
        oracle_pnl_improves=totals['arrival_oracle']['pnl']>totals['baseline']['pnl'])
    result=dict(asof=A.stamp(),freeze=json.loads((A.REVIEW/'freeze_v1.json').read_text()),data=data,fits=fits,replays=reps,
        totals=totals,prediction_metrics=mm,daily_pnl_changes=diff,computed_descriptive_predicates=predicate,
        controls=controls(),source_sha256=L.sha(__file__),limits='Development test on consumed dates. Teacher-score prediction metrics are not realized value. Entry only, cancellation unchanged. No inference or untouched validation claim.')
    A.dump(out/'report.json',result)
    print(json.dumps(dict(totals=totals,prediction_metrics=mm,daily_pnl_changes=diff,predicates=predicate),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=['selftest','fit','replay','report']);p.add_argument('--day');p.add_argument('--root',type=Path,default=A.ROOT);a=p.parse_args()
    if a.task=='selftest':print(json.dumps(controls()))
    elif a.task=='fit':fit(a.day,a.root)
    elif a.task=='replay':replay(a.day,a.root)
    else:report(a.root)
