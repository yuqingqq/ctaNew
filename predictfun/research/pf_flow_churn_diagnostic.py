"""Frozen, descriptive entry-to-first-review attribution; no policy changes."""
import argparse
from collections import Counter
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import pickle
import subprocess
import time
import pf_flow_control_common as F
import pf_flow_control_policy as P
import pf_flow_control_paths as PATH
import pf_flow_control_kernel as K
import pf_flow_control_run as RUN
import pf_flow_control_checks as CHECK
C,np,pd,I=F.C,F.np,F.pd,F.I
OUT=Path(I.K.IO)/'flow_churn_20261116_v1'
REVIEW=F.REVIEW/'churn_v1'
FREEZE=F.REVIEW/'churn_v1_freeze.json'
FACTORS=('market_paths_book','forecast_probabilities','target_queue','portfolio_inventory')
PER_GROUP=8


def uid(m,s,d,t):
    return f'{int(m)}:{int(s)}:{int(d)}:{t:.0f}'


def digest(x):
    return hashlib.sha256((x+':first-review-v1').encode()).hexdigest()


def spec(W,prob,k):
    w=W[0];j=int(w['jk'][k]);t=float(w['tg'][k])
    return dict(phase=min(5,int((t-w['ws'])//60000)+1),
        bid=float(w['P5'][j,0]),spread=float(W[1]['P5'][j,0]-w['P5'][j,0]),
        prices=np.array([W[s]['P5'][j] for s in (0,1)]),
        sizes=np.array([W[s]['S5'][j] for s in (0,1)]),prob=prob[:,k,:].copy())


def context(paths,sp,prob=None,h=1000.):
    pp=PATH.select(paths,sp['phase'],sp['bid'],sp['spread'])
    C.require(pp,'missing counterfactual support')
    packed=K.pack(pp,sp['bid'],2000. if h==2000. else 1000.)
    K.current_book(packed,sp['prices'],sp['sizes'])
    weight,diagnostic=PATH.weights(pp,sp['prob'] if prob is None else prob)
    if h not in (1000.,2000.):
        # Fixed original absolute endpoint: use the current scenario's valid
        # venue book at its shorter remaining horizon, without changing weights.
        marks=[]
        for p in pp:
            ff=p['frames'];valid=(ff[:,2]>=0)&(ff[:,12]>ff[:,2])&(ff[:,0]<=h)
            C.require(valid.any(),'short endpoint has no valid mark')
            f=ff[np.flatnonzero(valid)[-1]]
            marks.append((f[2]+f[12])/2+sp['bid']-p['bid'])
        packed=(*packed[:5],np.asarray(marks,float))
    return packed,weight,sp['bid']+sp['spread']/2,pp,diagnostic


def advantage(orders,actions,position,index,ctx,other_action=2.,h=1000.):
    a=np.asarray(actions,float).copy();a[index]=1.
    b=a.copy();b[index]=other_action
    out,qty=K.evaluate(ctx[0],orders,np.stack([a,b]),position,ctx[2],h)
    v=ctx[1]@P.utility(out,position)
    wealth=ctx[1]@out[:,:,0]
    q=ctx[1]@qty[:,:,index]
    return dict(value=float(v[0]-v[1]),wealth=float(wealth[0]-wealth[1]),
        risk=float(v[0]-v[1]-wealth[0]+wealth[1]),quantity_keep=float(q[0]),quantity_other=float(q[1]))


def prefix(state,selected,target=None):
    a=state['baseline'].copy()
    for item in selected:
        if item['index']==target:break
        a[item['index']]=2. if item['kind']=='cancel' else 1.
    return a


def queue_shown(sp,side,price):
    pp=sp['prices'][side];ss=sp['sizes'][side]
    if pp[4]>=0 and (price<pp[4] if side==0 else price>pp[4]):return 0.,True
    return float(ss[pp==price].sum()),False


def shapley(values):
    C.require(set(values)==set(range(16)),'incomplete attribution cube')
    ans=np.zeros(4)
    for j in range(4):
        for bits in range(16):
            if bits&(1<<j):continue
            n=bits.bit_count();w=math.factorial(n)*math.factorial(3-n)/math.factorial(4)
            ans[j]+=w*(values[bits|(1<<j)]-values[bits])
    C.require(np.isclose(ans.sum(),values[15]-values[0],atol=1e-7),'attribution does not telescope')
    return ans


def decompose(pair,paths):
    e,r=pair['entry'],pair['review'];ei,ri=e['index'],r['index']
    es,rs=e['state'],r['state'];target=rs['orders'][ri].copy();side=int(target[0])
    oldq,unknown=queue_shown(e['spec'],side,target[2]);values={};details={}
    contexts={(m,p):context(paths,[e,r][m]['spec'],[e,r][p]['spec']['prob']) for m in (0,1) for p in (0,1)}
    # Entry portfolio stays exactly as submitted (including entry's other greedy
    # actions). Only the target is repriced as active. These are model-input
    # interventions, not claims that these hybrid states actually occurred.
    for bits in range(16):
        m,p,q,bg=[(bits>>i)&1 for i in range(4)]
        state=rs if bg else es;index=ri if bg else ei
        orders=state['orders'].copy();orders[index]=target
        if not q:orders[index,4]=oldq
        action=r['prefix'] if bg else e['final']
        d=advantage(orders,action,state['position'],index,contexts[m,p])
        details[bits]=d;values[bits]=d['value']
    sh=shapley(values)
    if r['cancel']:
        C.require(np.isclose(-values[15],r['selected_advantage'],atol=1e-7),'cancel value differs from frozen chosen step')
    actual=advantage(es['orders'],e['prefix'],es['position'],ei,contexts[0,0],0.)
    C.require(np.isclose(actual['value'],e['selected_advantage'],atol=1e-7),'entry value differs from frozen chosen step')
    final=advantage(es['orders'],e['final'],es['position'],ei,contexts[0,0],0.)
    fixedq=es['orders'].copy();fixedq[ei,4]=oldq;fixedq[ei,8]=0.
    fixed=advantage(fixedq,e['final'],es['position'],ei,contexts[0,0],0.)
    active=fixedq.copy();active[ei]=target;active[ei,4]=oldq
    active_absent=advantage(active,e['final'],es['position'],ei,contexts[0,0],0.)
    h=1000.-(r['time']-e['time'])
    C.require(0<h<=1000.,'invalid fixed original endpoint')
    short=advantage(rs['orders'],r['prefix'],rs['position'],ri,context(paths,r['spec'],h=h),h=h)
    long=advantage(rs['orders'],r['prefix'],rs['position'],ri,context(paths,r['spec'],h=2000.),h=2000.)
    # Post/wait comparison with the same current price/portfolio/forecast, but
    # a hypothetical fresh replacement. A positive value alongside a negative
    # keep value identifies a queue/activation-dependent cancel/repost incentive.
    new=rs['orders'].copy();new[ri,4:]=[0.,58.,1e30,0.,1.]
    fresh=advantage(new,r['prefix'],rs['position'],ri,contexts[1,1],0.)
    out=dict(uid=pair['uid'],m=pair['m'],side=side,rung=int(target[1]),cancel=r['cancel'],
        entry_selected=actual['value'],entry_final=final['value'],entry_fixed_queue=fixed['value'],
        entry_active_absence=active_absent['value'],frozen_entry_keep=values[0],current_keep=values[15],
        current_wealth=details[15]['wealth'],current_risk=details[15]['risk'],
        current_keep_qty=details[15]['quantity_keep'],current_cancel_qty=details[15]['quantity_other'],
        current_fresh_post=fresh['value'],fixed_original_endpoint_keep=short['value'],h2_keep=long['value'],
        old_queue=oldq,old_queue_unknown=unknown,current_queue=float(target[4]),
        structural_at_first_review=pair['structural'],
        entry_greedy_change=final['value']-actual['value'],
        entry_queue_assumption_change=fixed['value']-final['value'],
        target_activation_change=active_absent['value']-fixed['value'],
        absent_to_cancel_change=values[0]-active_absent['value'],
        **{'shapley_'+k:float(sh[i]) for i,k in enumerate(FACTORS)})
    bridge=sum(out[k] for k in ['entry_greedy_change','entry_queue_assumption_change','target_activation_change','absent_to_cancel_change'])
    C.require(np.isclose(out['entry_selected']+bridge+sh.sum(),out['current_keep'],atol=1e-7),'full bridge reconciliation')
    return out


class Observe(P.Controller):
    def __init__(self,W,paths,prob):
        super().__init__(W,paths,prob,'JOINT');self.pending={};self.entries={};self.rows=[];self.sample={}

    def on_step(self,k,t,ST,kpos):
        original=P.choose;capture={}
        def wrapped(state,packed,weights,mid,h):
            action,selected,count=original(state,packed,weights,mid,h)
            capture.update(state=copy.deepcopy(state),action=action.copy(),selected=copy.deepcopy(selected))
            return action,selected,count
        P.choose=wrapped
        try:super().on_step(k,t,ST,kpos)
        finally:P.choose=original
        if capture:
            state=capture['state'];selected=capture['selected'];sp=spec(self.W,self.prob,k)
            for i,info in enumerate(state['info']):
                if info['kind']!='cancel' or not 0<=t-info['activation']<=50.+1e-7:continue
                key=uid(self.W[0]['m'],info['side'],info['rung'],info['submitted'])
                if key not in self.pending:continue
                e=self.pending.pop(key);self.entries[key]['status']='paired_first_review'
                cancel=[a for a in selected if a['index']==i and a['kind']=='cancel']
                row=state['orders'][i];s=int(row[0]);price=row[2]
                best=sp['prices'][s,0];opp=sp['prices'][1-s,0]
                structural=bool(best<0 or (best-price if s==0 else price-best)>3+row[1] or (price>=opp if s==0 else price<=opp))
                pr=prefix(state,selected,i) if cancel else capture['action'].copy()
                review=dict(time=t,state=state,index=i,prefix=pr,spec=sp,cancel=bool(cancel),selected_advantage=cancel[0]['advantage_cents'] if cancel else None)
                p=dict(uid=key,m=int(self.W[0]['m']),entry=e,review=review,structural=structural)
                oldq,unk=queue_shown(e['spec'],s,price)
                self.rows.append(dict(uid=key,m=p['m'],side=s,rung=int(row[1]),submit_ms=e['time'],review_ms=t,
                    cancel=bool(cancel),structural=structural,entry_advantage=e['selected_advantage'],
                    probability_l1=float(np.abs(e['spec']['prob']-sp['prob']).sum(1).mean()),
                    best_changed=bool(e['spec']['bid']!=sp['bid'] or e['spec']['spread']!=sp['spread']),
                    depth_changed=bool(not np.array_equal(e['spec']['prices'],sp['prices']) or not np.array_equal(e['spec']['sizes'],sp['sizes'])),
                    mid_delta=float(sp['bid']+sp['spread']/2-e['spec']['bid']-e['spec']['spread']/2),
                    old_queue=oldq,current_queue=float(row[4]),old_queue_unknown=unk,
                    inventory_delta=state['position']-e['state']['position'],
                    current_cancel_advantage=cancel[0]['advantage_cents'] if cancel else None))
                group=(s,bool(cancel));a=self.sample.setdefault(group,[]);a.append((digest(key),p));a.sort(key=lambda x:x[0]);del a[PER_GROUP:]
            for item in selected:
                if item['kind']!='post':continue
                key=uid(self.W[0]['m'],item['side'],item['rung'],t)
                self.pending[key]=dict(time=t,state=state,index=item['index'],prefix=prefix(state,selected,item['index']),
                    final=capture['action'].copy(),spec=sp,selected_advantage=item['advantage_cents'])
                self.entries[key]=dict(uid=key,m=int(self.W[0]['m']),side=item['side'],rung=item['rung'],submitted=t,status='not_paired')
        for key,e in list(self.pending.items()):
            if t-e['time']>=100.-1e-7:del self.pending[key]


def selftest():
    c={};v={b:2.+sum((i+1) for i in range(4) if b&(1<<i))+ (3. if b&3==3 else 0.) for b in range(16)}
    np.testing.assert_allclose(shapley(v),[2.5,3.5,3.,4.]);c['attribution_positive']=True
    c['incomplete_cube_refused']=CHECK.refused(lambda:shapley({k:v[k] for k in range(15)}),'missing cube cell')
    np.testing.assert_allclose(shapley({b:1. for b in range(16)}),0.);c['unchanged_inputs_zero']=True
    # A stationary toy mixture can prefer posting while rejecting an already
    # active order: placement skips the bad 40-58ms segment, cancellation can
    # avoid it. This is a positive instrument control, not evidence on real data.
    a=CHECK.toy([(45.,0,49.,30.)],marks=(48.,48.),depth=0.)
    b=CHECK.toy([(150.,0,50.,10.)],marks=(52.,52.),depth=0.)
    packed=K.pack([a,b],50,1000.);ctx=(packed,np.array([.5,.5]),51.)
    new=np.array([[0,0,50,30,0,58,1e30,0,1]],float)
    live=new.copy();live[0,5]=-42;live[0,8]=0
    post=advantage(new,[1],0,0,ctx,0.);keep=advantage(live,[1],0,0,ctx)
    C.require(post['value']>0 and keep['value']<0,'clock reversal positive control')
    c['clock_reversal_detected']=True
    gain=K.pack([b],50,1000.);ctx2=(gain,np.ones(1),51.)
    C.require(advantage(new,[1],0,0,ctx2,0.)['value']>0 and advantage(live,[1],0,0,ctx2)['value']>0,'all good paths should not reverse')
    c['good_path_no_false_reversal']=True
    return c


def freeze():
    F.verify();C.require(not FREEZE.exists(),'diagnostic freeze exists')
    C.dump(FREEZE,dict(asof=C.stamp(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=F.ROOT,text=True).strip(),
        source_sha256=C.sha(Path(__file__)),controller_freeze_sha256=C.sha(F.FREEZE),
        population='Same 12 consumed Oct27 pilot markets, JOINT policy unchanged. Every selected post gets a status. All matching first eligible 42ms-age review pairs retained.',
        sample='Up to8 identity-hash selected pairs per market/side/first-review-model-cancel-status, before any attribution value is computed; no P&L or outcome selection.',
        attribution='Chosen-step post/wait -> final entry portfolio -> fixed current known queue at entry -> active target -> keep/cancel -> 16-cell Shapley of current market/path bank, forecast probabilities, target queue, and other portfolio/inventory. Hybrid input sensitivity, not a claim of real-world causal effects.',
        horizons='Current decision under1s, original absolute entry endpoint (~900ms remaining), fixed2s; no policy retuning. Original endpoint only isolates local endpoint sensitivity under current forecast, not full dynamically conditioned paths.',
        null=dict(draws=200,seed=20261116,groups=['market','side'],metric='Mean forecast L1 change among model cancellations minus kept first-review orders, count-matched label shuffle. Descriptive only.'),
        controls='Exact archived policy trace and fill ledger identity; chosen-step value reconstruction; telescoping/Shapley checks; planted clock reversal, no-change and no-reversal controls; wrong cube/ledger refusals.',
        multiplicity=dict(new_models=0,new_policies=0,attribution_designs=1),limits=['One consumed day; no confidence intervals, significance or promotion','Own-policy diagnostic conditions on observed entry and first-review state; not neutral-path training data','Fixed-entry queue and hybrid portfolio interventions are approximations; unreceived fills remain omitted','No claim that extending horizon supplies queue continuation value'] ))


def matched_null(rows):
    rng=np.random.default_rng(20261116);a=rows.cancel.to_numpy(bool);x=rows.probability_l1.to_numpy(float)
    metric=lambda z:float(x[z].mean()-x[~z].mean())
    groups=list(rows.groupby(['m','side'],sort=False).indices.values());values=[]
    for _ in range(200):
        z=a.copy()
        for ix in groups:z[ix]=rng.permutation(a[ix])
        C.require(all(z[ix].sum()==a[ix].sum() for ix in groups),'shuffle count mismatch');values.append(metric(z))
    return dict(n=len(rows),draws=200,observed=metric(a),null_mean=float(np.mean(values)),null_min=float(min(values)),null_max=float(max(values)))


def report(rows,attrib,entries):
    result=dict(asof=C.stamp(),test_day=F.TEST,markets=int(entries.m.nunique()),entry_statuses=dict(Counter(entries.status)),
        selected_posts=len(entries),first_review_pairs=len(rows),matched_null=matched_null(rows),groups={})
    for name,cancel in [('cancel',True),('keep',False)]:
        x=rows[rows.cancel==cancel];a=attrib[attrib.cancel==cancel]
        result['groups'][name]=dict(n=len(x),sample_n=len(a),
            best_unchanged=int((~x.best_changed).sum()),depth_unchanged=int((~x.depth_changed).sum()),
            probability_l1_mean=float(x.probability_l1.mean()),probability_l1_median=float(x.probability_l1.median()),
            inventory_unchanged=int((x.inventory_delta.abs()<1e-7).sum()),queue_unchanged=int((np.abs(x.old_queue-x.current_queue)<1e-7).sum()),
            structural=int(x.structural.sum()),entry_advantage_median=float(x.entry_advantage.median()),
            means={k:float(a[k].mean()) for k in a.select_dtypes('number').columns if k not in ('m','side','rung')},
            counts=dict(entry_final_negative=int((a.entry_final<-1e-9).sum()),
                frozen_entry_keep_negative=int((a.frozen_entry_keep<-1e-9).sum()),
                fresh_post_positive=int((a.current_fresh_post>1e-9).sum()),
                fresh_post_positive_and_keep_negative=int(((a.current_fresh_post>1e-9)&(a.current_keep<-1e-9)).sum()),
                wealth_only_keep_negative=int((a.current_wealth<-1e-9).sum()),
                fixed_original_endpoint_keep_negative=int((a.fixed_original_endpoint_keep<-1e-9).sum()),
                h2_keep_negative=int((a.h2_keep<-1e-9).sum())))
    return result


def run():
    start=time.time();F.verify();f=json.loads(FREEZE.read_text());C.require(C.sha(Path(__file__))==f['source_sha256'],'diagnostic source changed')
    controls=selftest();C.require(not OUT.exists(),'diagnostic output exists');OUT.mkdir(parents=True)
    pilot=F.OUT/'pilot';receipt=json.loads((pilot/'receipt.json').read_text())
    for n,h in receipt['outputs'].items():C.require(C.sha(pilot/n)==h,'changed pilot '+n)
    for p,h in receipt['inputs'].items():C.require(C.sha(p)==h,'changed raw input')
    prior=pd.read_parquet(pilot/'decisions.parquet');prior=prior[prior.case=='JOINT']
    fills=pd.read_parquet(pilot/'fills.parquet');fills=fills[fills.case=='JOINT']
    paths=PATH.load();r=json.loads((C.OUT/F.TEST/'receipt.json').read_text())
    selected=receipt['selected_markets'];reg=pd.read_parquet(Path(I.K.IO)/'step1'/F.TEST/'registry.parquet')
    windows={int(x.mid):float(x.ws)*1000 for x in reg[reg.mid.isin(selected)].itertuples()}
    fair,_=I.fair_fix(F.TEST,windows);hours=I.X5.depth_hours(F.TEST)
    clock=json.loads((Path(I.K.IO)/'step1'/F.TEST/'clock_rl_d2.json').read_text())
    depth,_=I.X5.depth_arrays(hours,float(clock['bn_in'])+float(clock['d_dec']))
    data=I.load_day(F.TEST,set(selected),r['imputation_offset_ms'])
    rows=[];pairs=[];attribution=[];entries=[];seen=[]
    for W in I.pairs(data):
        m=int(W[0]['m']);X,_=I.market_inputs(W,fair[m],depth);prob=P.forecasts(W,X)
        pol=Observe(W,paths,prob);result=RUN.run_lane(W,pol);seen.append(m)
        actual=pd.DataFrame(pol.trace);old=prior[prior.m==m].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[old.columns].reset_index(drop=True),old,check_dtype=False,atol=1e-9,rtol=1e-10)
        g=fills[fills.m==m];oldfills=Counter(g[['side','rung','fill_ms','price_cents','shares']].itertuples(index=False,name=None))
        C.require(Counter(result['fl'])==oldfills,'observer altered archived fills')
        actualorders={uid(m,s,d,o['ts']):o for (s,d),oo in result['order_states'].items() for o in oo}
        for key,e in pol.entries.items():
            if e['status']=='paired_first_review':continue
            o=actualorders.get(key)
            e['status']='not_placed' if o is None else ('activation_rejected' if o['ref'] else ('structural_cancel_before_first_review' if o['tc']<o['ts']+100. else 'no_eligible_context_at_first_review'))
        sample=[p for a in pol.sample.values() for _,p in a]
        attribution.extend(decompose(p,paths) for p in sample);pairs.extend(sample);rows.extend(pol.rows);entries.extend(pol.entries.values())
        C.guard('churn diagnostic market')
        print(json.dumps(dict(m=m,pairs=len(pol.rows),sample=len(sample),seconds=round(time.time()-start))),flush=True)
    C.require(set(seen)==set(selected),'missing diagnostic markets')
    rows=pd.DataFrame(rows);entries=pd.DataFrame(entries);attrib=pd.DataFrame(attribution)
    C.require(not rows.uid.duplicated().any() and not attrib.uid.duplicated().any(),'duplicate generations')
    C.require(set(attrib.uid)<=set(rows.uid),'sample outside paired population')
    fast=prior[(prior.kind=='cancel')&(prior.time-prior.activation<=50.+1e-7)]
    C.require(int(rows.cancel.sum())==len(fast),'first-review cancellation identity coverage')
    controls['archived_trace_and_fills_exact']=True
    controls['corrupt_fill_ledger_refused']=CHECK.refused(lambda:C.require(oldfills==oldfills+Counter({(0,0,0.,1.,1.):1}),'bad ledger'),'corrupt fill ledger')
    for name,df in [('pairs',rows),('entries',entries),('attribution',attrib)]:df.to_parquet(OUT/(name+'.parquet'),index=False)
    (OUT/'states.pkl').write_bytes(pickle.dumps(pairs,protocol=5))
    rep=report(rows,attrib,entries);C.dump(OUT/'report.json',rep)
    C.dump(OUT/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(FREEZE),source_sha256=C.sha(Path(__file__)),
        pilot_receipt_sha256=C.sha(pilot/'receipt.json'),controller_freeze_sha256=C.sha(F.FREEZE),controls=controls,
        selected_markets=selected,seconds=time.time()-start,outputs={p.name:C.sha(p) for p in OUT.iterdir() if p.is_file()}))
    print(json.dumps(C.clean(rep)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','selftest','run']);a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='selftest':print(json.dumps(selftest()))
    else:run()
