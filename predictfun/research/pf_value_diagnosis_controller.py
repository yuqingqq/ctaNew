"""SOFT_JOINT first-review attribution and complement invariance diagnostics."""
import argparse
from collections import Counter, OrderedDict
import copy
import gc
import json
import pickle
import time
from unittest.mock import patch
import pf_value_diagnosis as D
J,CH,F,C,np,pd,I,V=D.J,D.CH,D.F,D.C,D.np,D.pd,D.I,D.V
P,K,PATH,RUN=CH.P,CH.K,CH.PATH,CH.RUN


def context(paths,sp,prob=None,h=1000.):
    pp=V.pool(paths,sp['phase'],sp['bid'],sp['spread'])
    C.require(pp,'missing SOFT counterfactual support')
    packed=K.pack(pp,sp['bid'],2000. if h==2000. else 1000.)
    K.current_book(packed,sp['prices'],sp['sizes'])
    weight,diagnostic=PATH.weights(pp,sp['prob'] if prob is None else prob)
    if h not in (1000.,2000.):
        marks=[]
        for p in pp:
            ff=p['frames'];valid=(ff[:,2]>=0)&(ff[:,12]>ff[:,2])&(ff[:,0]<=h)
            C.require(valid.any(),'short endpoint missing mark');f=ff[np.flatnonzero(valid)[-1]]
            marks.append((f[2]+f[12])/2+sp['bid']-p['bid'])
        packed=(*packed[:5],np.asarray(marks,float))
    return packed,weight,sp['bid']+sp['spread']/2,pp,diagnostic


def mirror_prices(prices):
    p=np.asarray(prices,float)
    return np.where(p>=0,100.-p,p)


def mirror_path(p):
    z=copy.deepcopy(p);f=p['frames'];g=f.copy()
    g[:,2:7]=mirror_prices(f[:,12:17]);g[:,7:12]=f[:,17:22]
    g[:,12:17]=mirror_prices(f[:,2:7]);g[:,17:22]=f[:,7:12]
    z['frames']=g;t=p['trades'].copy();t[:,1]=1-t[:,1];t[:,2]=100-t[:,2];z['trades']=t
    z['bid']=100.-p['bid']-p['spread'];z['marks']=[100.-m for m in p['marks']]
    z['min_price']=100.-p['max_price'];z['max_price']=100.-p['min_price'];z['classes']=list(reversed(p['classes']))
    return z


def mirror_spec(sp):
    z=copy.deepcopy(sp);z['bid']=100.-sp['bid']-sp['spread']
    z['prices']=mirror_prices(sp['prices'][::-1]);z['sizes']=sp['sizes'][::-1].copy();z['prob']=sp['prob'][::-1].copy()
    return z


def mirror_orders(orders):
    z=orders.copy();z[:,0]=1.-z[:,0];z[:,2]=100.-z[:,2];return z


def mirror_context(ctx,sp):
    # Transform the already translated kernel representation. Re-anchoring a
    # mirrored donor to current bid would shift it again when spreads differ.
    fo,to,f,t,delta,mark=ctx[0];g=f.copy();tr=t.copy()
    g[:,2:7]=mirror_prices(f[:,12:17]);g[:,7:12]=f[:,17:22]
    g[:,12:17]=mirror_prices(f[:,2:7]);g[:,17:22]=f[:,7:12]
    tr[:,1]=1-tr[:,1];tr[:,2]=100-tr[:,2]
    packed=(fo.copy(),to.copy(),g,tr,-delta.copy(),100.-mark)
    return packed,ctx[1].copy(),100.-ctx[2],[mirror_path(p) for p in ctx[3]],ctx[4]


class Observe(CH.Observe):
    def __init__(self,W,paths,prob):
        super().__init__(W,paths,prob);self.case='SOFT_JOINT';self.cache=OrderedDict()

    def context(self,k,t):
        return V.Controller.context(self,k,t)

    def on_step(self,k,t,ST,kpos):
        begin=len(self.rows);super().on_step(k,t,ST,kpos)
        if len(self.rows)==begin:return
        samples={p['uid']:p for group in self.sample.values() for _,p in group if p['review']['time']==t}
        for row in self.rows[begin:]:
            s=int(row['side']);matches=[o for o in ST[s]['cl'] if o['rung']==row['rung'] and o['ts']==row['submit_ms']]
            C.require(len(matches)==1,'first-review order identity')
            order=matches[0];known=30.-sum(q for ft,q in order['fl'] if ft+100.<=t)
            actual=30.-sum(q for ft,q in order['fl'] if ft<=t)
            row.update(known_remaining=known,actual_remaining=max(0.,actual),unreceived_fill_shares=max(0.,known-actual))
            if row['uid'] in samples:
                r=samples[row['uid']]['review']
                r['latent_label_state']=J.label_state(self.W,k,t,ST,kpos,r['state'])


def extra(pair,paths):
    e,r=pair['entry'],pair['review'];s=r['state'];i=r['index'];sp=r['spec'];orders=s['orders']
    ctx=context(paths,sp);value=lambda o,c,pos=s['position']:CH.advantage(o,r['prefix'],pos,i,c)['value']
    actual=value(orders,ctx);oldp=context(paths,sp,e['spec']['prob']);oldm=context(paths,e['spec'],sp['prob'])
    queue=orders.copy();queue[i,4]=CH.queue_shown(e['spec'],int(orders[i,0]),orders[i,2])[0]
    latent=r['latent_label_state'];true_state=value(latent['orders'],ctx,latent['position'])
    mirrored=mirror_context(ctx,sp);mirror_value=value(mirror_orders(orders),mirrored,-s['position'])
    D.check_equal(actual,mirror_value,'complement kernel value invariant')
    query=mirror_spec(sp);newpaths=V.pool(paths,query['phase'],query['bid'],query['spread'])
    status='eligible' if newpaths else 'no_mirrored_query_support'
    query_value=value(mirror_orders(orders),context(paths,query),-s['position']) if newpaths else np.nan
    oldids={p['uid'] for p in oldm[3]};ids={p['uid'] for p in ctx[3]}
    return dict(uid=pair['uid'],m=pair['m'],side=int(orders[i,0]),cancel=r['cancel'],current_keep=actual,
        old_forecast_keep=value(orders,oldp),old_market_keep=value(orders,oldm),old_queue_keep=value(queue,ctx),
        true_current_state_keep=true_state,hidden_state_delta=true_state-actual,
        actual_remaining=float(latent['orders'][i,3]),known_remaining=float(orders[i,3]),
        kernel_mirror_value=mirror_value,mirror_query_status=status,mirror_query_keep=query_value,
        mirror_query_delta=query_value-actual,bank_changed=oldids!=ids,bank_jaccard=len(oldids&ids)/len(oldids|ids),
        spread_changed=e['spec']['spread']!=sp['spread'],phase_changed=e['spec']['phase']!=sp['phase'])


def summarize(rows,attrib,entries,sens):
    result=CH.report(rows,attrib,entries);result['case']='SOFT_JOINT';result['sensitivity']={}
    for key,g in sens.groupby(['cancel','side']):
        z=g[g.mirror_query_status=='eligible']
        result['sensitivity'][str(key)]=dict(n=len(g),mirror_statuses=g.mirror_query_status.value_counts().to_dict(),
            bank_changed=int(g.bank_changed.sum()),spread_changed=int(g.spread_changed.sum()),
            old_forecast_still_cancel=int((g.old_forecast_keep<-1e-9).sum()),old_market_still_cancel=int((g.old_market_keep<-1e-9).sum()),
            old_queue_still_cancel=int((g.old_queue_keep<-1e-9).sum()),true_state_still_cancel=int((g.true_current_state_keep<-1e-9).sum()),
            hidden_focal_fill_n=int((g.actual_remaining<g.known_remaining-1e-9).sum()),
            mean_hidden_state_delta=float(g.hidden_state_delta.mean()),mean_mirror_query_delta=float(z.mirror_query_delta.mean()),
            mean_abs_mirror_query_delta=float(z.mirror_query_delta.abs().mean()),
            mirrored_preference_changed=int(((z.current_keep<-1e-9)!=(z.mirror_query_keep<-1e-9)).sum()))
    result['by_side']={}
    for (cancel,side),g in rows.groupby(['cancel','side']):
        result['by_side'][str((cancel,side))]=dict(n=len(g),hidden_focal_fill_n=int((g.unreceived_fill_shares>1e-9).sum()),
            adverse_mid_n=int(((1-2*g.side)*g.mid_delta<0).sum()),
            unchanged_book_n=int((~g.depth_changed).sum()),entry_median_cents=float(g.entry_advantage.median()))
    return result


def selftest():
    controls=CH.selftest()
    path=CH.CHECK.toy([(150.,0,50.,10.),(160.,1,52.,8.)],marks=(52.,53.),depth=0.)
    path.update(uid='toy',min_price=48.,max_price=54.,phase=1,classes=[0,1],base_weight=1.)
    sp=dict(phase=1,bid=50.,spread=2.,prices=np.array([[50,49,48,47,46],[52,53,54,55,56]],float),
        sizes=np.zeros((2,5)),prob=np.full((2,15),1/15))
    packed=K.pack([path],50.,1000.);K.current_book(packed,sp['prices'],sp['sizes']);ctx=(packed,np.ones(1),51.,[path],{})
    orders=np.array([[0,0,50,30,0,-50,1e30,0,0],[1,0,52,30,0,-50,1e30,0,0]],float)
    for i in (0,1):
        a=CH.advantage(orders,[1,1],5.,i,ctx)['value']
        b=CH.advantage(mirror_orders(orders),[1,1],-5.,i,mirror_context(ctx,sp))['value']
        D.check_equal(a,b,'toy mirror')
    twice=mirror_path(mirror_path(path));D.check_equal(twice['frames'],path['frames'],'mirror involution')
    D.check_equal(twice['trades'],path['trades'],'trade involution')
    controls.update(complement_value_positive=True,complement_involution=True,
        corrupt_mirror_refused=CH.CHECK.refused(lambda:D.check_equal(a,b+1.,'corrupted mirror'),'corrupt mirror'))
    return controls


def run():
    D.verify();start=time.time();controls=selftest();out=D.OUT/'controller'
    C.require(not out.exists(),'controller diagnostic exists');out.mkdir(parents=True)
    pilot=V.OUT/'pilot';receipt=D.checked_receipt(pilot)
    prior=pd.read_parquet(pilot/'decisions.parquet');prior=prior[prior.case=='SOFT_JOINT']
    fills=pd.read_parquet(pilot/'fills.parquet');fills=fills[fills.case=='SOFT_JOINT']
    selected,data,fair,depth,raw=J.A.load(F.TEST);C.require(selected==receipt['selected_markets'],'pilot markets changed')
    paths=PATH.load();rows=[];entries=[];pairs=[];attribution=[];sensitivity=[];seen=[]
    for W in I.pairs(data):
        m=int(W[0]['m']);X,_=I.market_inputs(W,fair[m],depth);prob=P.forecasts(W,X)
        pol=Observe(W,paths,prob);result=RUN.run_lane(W,pol);seen.append(m)
        actual=pd.DataFrame(pol.trace);old=prior[prior.m==m].reset_index(drop=True)
        pd.testing.assert_frame_equal(actual[old.columns].reset_index(drop=True),old,check_dtype=False,atol=1e-9,rtol=1e-10)
        g=fills[fills.m==m];oldfills=Counter(g[['side','rung','fill_ms','price_cents','shares']].itertuples(index=False,name=None))
        C.require(Counter(result['fl'])==oldfills,'observer changed archived fills')
        orders={CH.uid(m,s,d,o['ts']):o for (s,d),group in result['order_states'].items() for o in group}
        for key,e in pol.entries.items():
            if e['status']=='paired_first_review':continue
            o=orders.get(key);e['status']='not_placed' if o is None else ('activation_rejected' if o['ref'] else
                ('structural_cancel_before_first_review' if o['tc']<o['ts']+100. else 'no_eligible_context_at_first_review'))
        sample=[p for group in pol.sample.values() for _,p in group]
        with patch.object(CH,'context',context):attribution.extend(CH.decompose(p,paths) for p in sample)
        sensitivity.extend(extra(p,paths) for p in sample);pairs.extend(sample);rows.extend(pol.rows);entries.extend(pol.entries.values())
        C.dump(out/'progress.json',dict(asof=C.stamp(),markets=len(seen),pairs=len(rows),sample=len(pairs),seconds=time.time()-start))
        print(json.dumps(dict(stage='controller',market=m,markets=len(seen),pairs=len(rows),seconds=round(time.time()-start))),flush=True)
        del pol,result,X,prob;gc.collect();C.guard('soft controller diagnostic')
    C.require(set(seen)==set(selected),'missing pilot markets')
    frames=dict(pairs=pd.DataFrame(rows),entries=pd.DataFrame(entries),attribution=pd.DataFrame(attribution),sensitivity=pd.DataFrame(sensitivity))
    for name,a in frames.items():
        C.require(not a.uid.duplicated().any(),'duplicate '+name);a.to_parquet(out/(name+'.parquet'),index=False)
    fast=prior[(prior.kind=='cancel')&(prior.time-prior.activation<=50.+1e-7)]
    C.require(int(frames['pairs'].cancel.sum())==len(fast),'rapid cancellation coverage')
    (out/'states.pkl').write_bytes(pickle.dumps(pairs,protocol=5))
    controls['archived_trace_and_fills_exact']=True
    controls['corrupt_ledger_refused']=CH.CHECK.refused(lambda:C.require(oldfills==oldfills+Counter({(0,0,0.,1.,1.):1}),'bad ledger'),'bad ledger')
    report=summarize(frames['pairs'],frames['attribution'],frames['entries'],frames['sensitivity'])
    report.update(controls=controls,limits=D.verify()['limits']);C.dump(out/'report.json',report)
    C.dump(out/'receipt.json',dict(asof=C.stamp(),source_manifest=C.sha(D.FREEZE),pilot_receipt_sha256=C.sha(pilot/'receipt.json'),
        selected_markets=selected,outputs=D.files(out),seconds=time.time()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','selftest']);a=p.parse_args()
    if a.mode=='selftest':print(json.dumps(selftest()))
    else:run()
