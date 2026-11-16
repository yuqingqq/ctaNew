"""Independent artifact audit for the frozen entry-arrival experiment."""
import json
from pathlib import Path
import sys
import time
import math
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pf_entry_arrival as A
import pf_entry_arrival_fit as F
np,L=A.np,A.L


def target_invariants(x,f,ids,status):
    cols=A.V.FULL_IN;ok=status=='supported'
    L.require(len(ids)==len(set(map(tuple,ids[:,:4]))),'duplicate action')
    L.require(np.array_equal(x[:,cols.index('o_price')],ids[:,6],equal_nan=True),'current quote price identity')
    L.require(np.array_equal(x[ok,cols.index('o_price')],f[ok,cols.index('o_price')]),'future price changed')
    dt=f[ok,cols.index('t_rel')]-x[ok,cols.index('t_rel')]
    L.require(np.all(np.abs(dt-A.PLACE/1000.)<1e-5),'target time offset')
    L.require(np.all(f[ok,cols.index('o_age')]==0) and np.all(f[ok,cols.index('o_rem')]==1),'future is not fresh')
    sign=np.where(ids[ok,1]==0,1.,-1.)
    mid=f[ok,cols.index('pf_mid')];spread=f[ok,cols.index('pf_spread')];price=f[ok,cols.index('o_price')]
    L.require(np.allclose(f[ok,cols.index('o_edge')],sign*(mid-price),rtol=0,atol=1e-6),'future side edge')
    L.require(np.allclose(f[ok,cols.index('o_depth')],sign*(mid-sign*spread/2-price),rtol=0,atol=1e-6),'future fixed-price depth')
    for name in ['o_q0_rel','o_shown_rel']:
        L.require(np.array_equal(f[ok,cols.index('o_q_rel')],f[ok,cols.index(name)],equal_nan=True),'future fresh queue')
    return dict(rows=len(ids),supported=int(ok.sum()),max_time_error_seconds=float(np.max(np.abs(dt-A.PLACE/1000.))) if ok.any() else None)


def controls():
    x=np.zeros((1,len(A.V.FULL_IN)),np.float32);f=x.copy();ids=np.array([[1,0,0,0,0,0,0]],float);s=np.array(['supported'])
    f[:,A.V.FULL_IN.index('t_rel')]=A.PLACE/1000.;f[:,A.V.FULL_IN.index('o_rem')]=1
    target_invariants(x,f,ids,s);bad=f.copy();bad[:,A.V.FULL_IN.index('o_price')]=1;caught=False
    try:target_invariants(x,bad,ids,s)
    except ValueError:caught=True
    L.require(caught,'changed-price falsifier did not fire')
    bad=f.copy();bad[:,A.V.FULL_IN.index('t_rel')]=.1;late=False
    try:target_invariants(x,bad,ids,s)
    except ValueError:late=True
    L.require(late,'rounded-time falsifier did not fire')
    ids=np.array([[1,0,0],[1,0,0],[2,0,0],[2,0,0]])
    keep=np.array([True,False,False,True]);base=np.array([1.,2.,3.,4.]);matched=matched_keep(ids,base,keep)
    L.require(np.array_equal(matched,[False,True,False,True]),'matched-current positive control')
    refused=False
    try:check_counts(ids,keep,np.array([True,True,False,False]))
    except ValueError:refused=True
    L.require(refused,'global count instead of matched strata not refused')
    return dict(correct_target_passes=True,wrong_price_refused=caught,rounded_100ms_refused=late,matched_current_control=True,unmatched_counts_refused=refused)


def strata(ids):
    return A.pd.DataFrame(ids[:,:3],columns=['m','side','rung']).groupby(['m','side','rung'],sort=False).indices.values()


def check_counts(ids,keep,control):
    for ix in strata(ids):L.require(int(keep[ix].sum())==int(control[ix].sum()),'unmatched window/side/rung count')


def matched_keep(ids,baseline,keep):
    out=np.zeros(len(keep),bool)
    for ix in strata(ids):
        n=int(keep[ix].sum());out[ix[np.argsort(-baseline[ix],kind='stable')[:n]]]=True
    check_counts(ids,keep,out);return out


def check_attribution(v):
    for fields,total in [(('age_lt100','age_100_500','age_ge500'),'pnl'),(('ws','e2'),'pnl'),(('ws_old','e2_old'),'pnl_old')]:
        L.require(abs(sum(v[k] for k in fields)-v[total])<1e-6,'attribution reconciliation')


def replay_attribution():
    """Read existing all-fill outputs; no new policy or selected population."""
    cols=['pnl','pnl_old','shares','shares_young','age_lt100','age_100_500','age_ge500','ws','e2','ws_old','e2_old','pend','pend_shares','imputed_pnl','imputed_shares','qunk','qunk_shares']
    result={};pooled={arm:Counter() for arm in ['baseline','arrival_oracle','forecast']};control=False
    for day in L.DAYS:
        path=A.ROOT/'replay'/day/'windows.parquet';receipt=json.loads((path.parent/'receipt.json').read_text())
        L.require(L.sha(path)==receipt['windows_sha256'],'attribution input hash')
        frame=A.pd.read_parquet(path);result[day]={}
        for arm,g in frame.groupby('arm'):
            v={c:float(g[c].sum()) for c in cols};check_attribution(v)
            if not control:
                bad=dict(v,pnl=v['pnl']+1.)
                try:check_attribution(bad)
                except ValueError:control=True
                L.require(control,'corrupted attribution not refused')
            result[day][arm]=v;pooled[arm].update(v)
    return dict(days=result,pooled={a:dict(v) for a,v in pooled.items()},corrupted_total_refused=control,
        limits='Existing M50/queue-M outputs only. Age/source splits are additive attribution, not exclusion-policy reruns. No M0/M250 or queue-L robustness claim.')


def live_states(day,heads,students):
    """First twelve registry windows, fixed independently of economic results."""
    D,windows,_=A.load(day);count=0;refused=False;nw=0;traces={arm:Counter() for arm in ['baseline','arrival_oracle','forecast']}
    def equal(f,g):
        L.require(all((math.isnan(float(f[c])) and math.isnan(float(g[c]))) or abs(float(f[c])-float(g[c]))<1e-9 for c in A.P.ORDER_KEYS),'live known-state mismatch')
    def check(W,cur,key,t,f):
        nonlocal count,refused
        g=A.AC.known_state(W,cur,key+(0,),t,30.,terminal_ms=W[0]['te']+200.,settings=A.AC.Settings(notification_ms=A.P.L_NOTE,cancel_ms=A.K.L_CANCEL))
        equal(f,g);count+=1
        if not refused:
            bad=dict(f,age_ms=f['age_ms']+1.)
            try:equal(bad,g)
            except ValueError:refused=True
            L.require(refused,'corrupted live age not refused')
    for W,X,_,_ in windows():
        fresh=F.fresh_scores(heads,students,W,X);m=int(W[0]['m'])
        oracle=np.load(A.ROOT/'data'/day/'oracle'/f'{m}.npz')['score']
        fresh['arrival_oracle']=np.where(np.isfinite(oracle),oracle,fresh['baseline'])
        with A.G.clocks(A.PLACE,A.CANCEL):
            for arm in ['baseline','arrival_oracle','forecast']:
                pol=F.EntryPolicy(heads,W,X,fresh[arm]);pol.check=check;seen=set();st=traces[arm]
                def wrapped(side,rung):
                    fn=pol.dec(side,rung)
                    def dec(k,t,o):
                        answer=fn(k,t,o)
                        if id(o) not in seen:
                            seen.add(id(o));st['first_client_reviews']+=1;st['first_review_pull']+=int(answer)
                            st['first_review_physically_remaining']+=int(o['rem']>1e-12)
                            st['first_review_pull_physically_remaining']+=int(answer and o['rem']>1e-12)
                        return answer
                    return dec
                rr=L.simulate(W,{(s,d):wrapped(s,d) for s in (0,1) for d in range(A.P.NR)},
                    {(s,d):pol.gate(s,d) for s in (0,1) for d in range(A.P.NR)},50.)
                for (s,d),orders in rr['order_states'].items():
                    for o in orders:
                        k=int(np.searchsorted(W[s]['tg'],o['ts']));st['submitted_orders']+=1
                        st['posts_with_negative_current_fresh_score']+=int(fresh['baseline'][s,d,k]<0)
                        target=float(oracle[s,d,k]);base=float(fresh['baseline'][s,d,k]);pred=float(fresh['forecast'][s,d,k])
                        if np.isfinite([target,base,pred]).all():
                            st['posts_with_all_scores']+=1;st['posted_teacher_score_sum']+=target
                            st['posted_forecast_score_sum']+=pred;st['posted_current_score_sum']+=base
                            st['posts_with_negative_arrival_teacher']+=int(target<0)
                            st['posted_forecast_absolute_error_sum']+=abs(pred-target)
                            st['posted_current_absolute_error_sum']+=abs(base-target)
                        else:st['posts_missing_a_score']+=1
        nw+=1
        if nw==12:break
    L.require(count>0 and refused,'no independent state control')
    L.require(traces['baseline']['posts_with_negative_current_fresh_score']==0,'baseline entry sign identity')
    return dict(windows=nw,comparisons=count,corrupted_age_refused=refused,posthoc_entry_guard_traces={k:dict(v) for k,v in traces.items()})


def run():
    freeze=A.verify_freeze();result=dict(asof=A.stamp(),controls=controls(),days={},fits={},teacher_identities=0)
    for day in A.DAYS:
        z,j=F.rows(day,A.ROOT)
        L.require(j['source_sha256']==freeze['sources']['pf_entry_arrival.py'],'data builder identity')
        L.require(len(z['ids'])==j['counts']['neutral_posts'],'missing placement status')
        valid=z['status']!='missing_current_inputs'
        result['days'][day]=target_invariants(z['current'][valid],z['future'][valid],z['ids'][valid],z['status'][valid])
        result['days'][day]['all_placements']=len(z['ids'])
        if day not in L.DAYS:continue
        fit_path=A.ROOT/'fit'/day/'receipt.json';fit=json.loads(fit_path.read_text())
        heads,teacher=A.teacher(day);prior=json.loads((A.G.PREVIOUS/f'receipt_{day}_v3.json').read_text())
        teacher_path=Path(A.V.OUTD)/f'fit3_{day}.json'
        for p in [teacher_path]+[Path(teacher['artifacts'][f'full_{h}']['path']) for h in ['p','v']]:
            L.require(L.sha(p)==prior['inputs'][str(p)]['sha256'],'positive-example teacher changed')
            result['teacher_identities']+=1
        L.require(j['teacher_fit_sha256']==fit['teacher_fit_sha256']==L.sha(teacher_path),'oracle/fit teacher identity')
        L.require(fit['train_days']==teacher['train_days'] and max(fit['train_days'])<teacher['cal']<day,'fold chronology')
        L.require(fit['source_sha256']==freeze['sources']['pf_entry_arrival_fit.py'],'fit source changed')
        for d,h in fit['data_receipts'].items():L.require(L.sha(A.ROOT/'data'/d/'receipt.json')==h,'fit data changed')
        predpath=A.ROOT/'fit'/day/'test_predictions.npz';L.require(L.sha(predpath)==fit['predictions_sha256'],'predictions changed')
        pred=np.load(predpath);y=A.future_scores(heads,z['future'],z['status']);b=A.score(heads,z['current']);pp=b.copy()
        students={}
        for s in (0,1):
            a=fit['heads'][str(s)];L.require(L.sha(a['path'])==a['sha256'],'student artifact changed')
            model=A.lgb.Booster(model_file=a['path']);L.require(tuple(model.feature_name())==A.FEATURES,'model feature schema')
            students[s]=model
            sel=z['ids'][:,1]==s;pp[sel]+=model.predict(A.student_matrix(z['current'][sel],b[sel]),num_threads=1)
        for name,left,right in [('target',y,pred['target']),('baseline',b,pred['baseline']),('forecast',pp,pred['forecast'])]:
            L.require(np.allclose(left,right,rtol=0,atol=1e-9,equal_nan=True),'prediction rederivation '+name)
        valid=np.isfinite(y)&(z['status']!='missing_current_inputs')
        subsets={}
        supported=valid&(z['status']=='supported')
        for name,subset in [('supported_changes_sign',supported&((y<0)!=(b<0))),('supported_same_sign',supported&((y<0)==(b<0))),('activation_rejections',valid&(z['status']=='reject_at_activation'))]:
            subsets[name]={key:F.metrics(y[subset],p[subset]) for key,p in [('persistence',b),('forecast',pp)]} if subset.any() else {'n':0}
        keep=pp[valid]>=0;matched=matched_keep(z['ids'][valid],b[valid],keep);yy=y[valid]
        selection=dict(post_count=int(keep.sum()),model_teacher_kept_value_cents=float(yy[keep].sum()),
            matched_current_teacher_kept_value_cents=float(yy[matched].sum()),
            model_minus_matched_current_cents=float(yy[keep].sum()-yy[matched].sum()),
            always_wait_teacher_disagreement_cents=float(np.maximum(yy,0).sum()),
            always_post_teacher_disagreement_cents=float(np.maximum(-yy,0).sum()),
            limits='Post-hoc descriptive comparison after first-date results; same post count within window/side/rung. Teacher value, not realized PnL; no inference or strategy selection.')
        result['fits'][day]=dict(rows=len(y),prediction_rederived=True,teacher_prior_identity=True,chronology=True,posthoc_matched_selection=selection,
            descriptive_posthoc_transition_subsets=subsets,live_state_checks=live_states(day,heads,students))
    # CPU-only marginal predictor cost, not end-to-end execution latency.
    loaded={s:A.lgb.Booster(model_file=fit['heads'][str(s)]['path']) for s in (0,1)}
    batches={s:A.student_matrix(z['current'][z['ids'][:,1]==s][:6],b[z['ids'][:,1]==s][:6]) for s in (0,1)}
    elapsed=[]
    for k in range(520):
        start=time.perf_counter_ns()
        for s in (0,1):loaded[s].predict(batches[s],num_threads=1)
        if k>=20:elapsed.append((time.perf_counter_ns()-start)/1e6)
    result['student_cpu_batch_12_quotes']=dict(samples=len(elapsed),median_ms=float(np.median(elapsed)),p95_ms=float(np.quantile(elapsed,.95)),max_ms=float(max(elapsed)))
    result['replay_attribution']=replay_attribution()
    result['completed_asof']=A.stamp()
    result['source_sha256']=L.sha(__file__)
    result['limits']='Artifact and timing identity checks, not physical clock validation or statistical inference.'
    out=A.REVIEW/'results_v1'/'audit_v1.json';L.require(not out.exists(),'audit output exists');A.dump(out,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
