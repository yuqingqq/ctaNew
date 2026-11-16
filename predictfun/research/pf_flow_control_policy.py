"""Causal state, coherent branch values, and an explicit research policy."""
from collections import Counter
import json
import math
import lightgbm as lgb
import pf_flow_control_common as F
import pf_flow_control_paths as PATH
import pf_flow_control_kernel as EXEC
import pf_flow_price_model as FM
import pf_stop_ev_play as P
C,np,pd,I=F.C,F.np,F.pd,F.I


def forecasts(W,X):
    fit=C.OUT/('fit_'+F.TEST);r=json.loads((fit/'receipt.json').read_text())
    C.require(r['training_days']==list(F.TRAIN) and r['calibration_day']==F.CAL and
              r['test_day']==F.TEST and r['source_manifest']==C.sha(C.REVIEW/'freeze_v1.json'),'forecast chronology/contract')
    a=r['artifacts']['joint'];C.require(C.sha(a['path'])==a['sha256'],'joint artifact changed')
    b=lgb.Booster(model_file=a['path']);temperature=r['temperature']['joint']['value']
    rows=[];keys=[]
    for k,t in enumerate(W[0]['tg']):
        for s in (0,1):
            idx=X[s][0][k]
            if idx<0:continue
            state=P.fresh_state(W[s],k,float(t),s,0,W[0]['ws']+300000.)
            if state is None:continue
            x=X[s][1][idx];scale=X[s][2][idx];mid=X[s][3][idx]
            extra=[30.,state['known_remaining'],state['queue_ahead_known'],state['shown_known'],scale,
                   min(5,int((t-W[0]['ws'])//60000)+1)]
            row=np.r_[x,P.order_vector(state,scale,mid),extra]
            C.require(len(row)==len(C.FEATURES),'forecast feature alignment')
            rows.append(row);keys.append((s,k))
    result=np.full((2,W[0]['n'],15),np.nan)
    if rows:
        matrix=np.asarray(rows,np.float32)
        probabilities=FM.temperature(b.predict(matrix,num_threads=1),temperature)
        # Batched and singleton calls must agree at actual queried state identities.
        ix=np.linspace(0,len(matrix)-1,min(17,len(matrix))).astype(int)
        for i in ix:C.require(np.allclose(probabilities[i],FM.temperature(b.predict(matrix[i:i+1],num_threads=1),temperature)[0],atol=1e-10),'forecast batch identity')
        for key,value in zip(keys,probabilities):result[key]=value
    return result


def snapshot(W,k,t,ST,kpos):
    """Read only client-known fills, frames, submissions and cancellations."""
    rows=[];info=[];held=set();active={};cur_ids={id(o):(s,d) for s in (0,1) for d,o in enumerate(ST[s]['cur']) if o is not None}
    for s in (0,1):
        for o in ST[s]['cl']:
            filled=sum(q for ft,q in o['fl'] if ft+100.<=t)
            if o['ts']>t or (o['ta']<=t and o['ref']) or filled>=30.-1e-9:continue
            removal=(max(o['tc']+37.,o['ts']+95.) if o['tc']<o['ta'] else o['tc']+37.) if o['tc']<1e25 else 1e30
            if removal+100.<=t:continue
            # Do not initialize a received-book queue cache before activation.
            # Confirmed partial fills imply front priority. An initially unknown
            # queue stays zero under the declared unknown-depth-zero assumption.
            ahead=0.
            if o['ta']<=t and filled<=1e-12:
                state=P.order_state(W[s],o,k,t,s,o['rung'],W[0]['ws']+300000.)
                estimate=state['queue_ahead_known']
                if math.isfinite(state['queue_initial_known']) and math.isfinite(estimate):ahead=float(estimate)
            rows.append([s,o['rung'],o['pt'],30.-filled,ahead,o['ta']-t,removal-t,o['ts']-t,float(o['ta']>t)])
            switch=id(o) in cur_ids and o['ta']<=t and o['tc']>t
            info.append(dict(side=s,rung=o['rung'],kind='cancel' if switch else 'pending',
                             activation=o['ta'],submitted=o['ts'],generation=len(ST[s]['orders'][o['rung']])-1))
            held.add((s,o['pt']))
            if id(o) in cur_ids:active[s,o['rung']]=len(rows)-1
    existing=len(rows)
    for s in (0,1):
        for d in range(6):
            if (s,d) in active:continue
            state=P.fresh_state(W[s],k,t,s,d,W[0]['ws']+300000.)
            if state is None or (s,state['price']) in held:continue
            rows.append([s,d,state['price'],30.,0.,58.,1e30,0.,1.])
            info.append(dict(side=s,rung=d,kind='post',activation=t+58.,submitted=t,generation=-1))
    orders=np.asarray(rows,float).reshape(-1,9)
    baseline=np.r_[np.ones(existing),np.zeros(len(rows)-existing)]
    candidates=[i for i,v in enumerate(info) if v['kind'] in ('cancel','post')]
    return dict(orders=orders,info=info,baseline=baseline,candidates=candidates,position=float(kpos),existing=existing)


def utility(out,position,risk=F.RISK):
    return out[:,:,0]-risk*(out[:,:,2]**2-position**2)


def feasible(state,action):
    # Submitted/cancel-pending orders retain reservation until acknowledgment.
    orders=state['orders'];existing=state['existing'];position=state['position']
    for side in (0,1):
        reserved=sum(o[3] for i,o in enumerate(orders) if o[0]==side and (i<existing or action[i]==1))
        if (position if side==0 else -position)+reserved>F.CAP+1e-7:return False
    return True


def choose(state,packed,weights,mid,horizon):
    base=state['baseline'].copy();remaining=list(state['candidates']);selected=[];evaluations=0
    if len(base)==0:return base,selected,dict(branch_evaluations=0,advantage_cents=0.)
    total_advantage=0.
    while remaining:
        branches=[base.copy()];keys=[]
        for i in remaining:
            action=base.copy();action[i]=2. if state['info'][i]['kind']=='cancel' else 1.
            # A cancellation is always available even if a preexisting state exceeds a reservation bound.
            if state['info'][i]['kind']=='post' and not feasible(state,action):continue
            branches.append(action);keys.append(i)
        if len(branches)==1:break
        out,qty=EXEC.evaluate(packed,state['orders'],np.asarray(branches),state['position'],mid,horizon)
        values=weights@utility(out,state['position']);evaluations+=len(branches)
        best=int(np.argmax(values));advantage=float(values[best]-values[0])
        if best==0 or advantage<=1e-9:break
        idx=keys[best-1];base=branches[best];remaining.remove(idx);total_advantage+=advantage
        selected.append(dict(index=idx,advantage_cents=advantage,**state['info'][idx]))
    return base,selected,dict(branch_evaluations=evaluations,advantage_cents=total_advantage)


class Controller:
    def __init__(self,W,paths,prob,case):
        self.W=W;self.paths=paths;self.prob=prob;self.case=case
        self.horizon=2000. if case=='JOINT_H2' else 1000.
        self.cancel=set();self.post=set();self.stats=Counter();self.trace=[];self.cache={}

    def context(self,k,t):
        w=self.W[0];j=int(w['jk'][k]);phase=min(5,int((t-w['ws'])//60000)+1)
        if j<0:return None,'no_current_book'
        bid=float(w['P5'][j,0]);ask=float(self.W[1]['P5'][j,0])
        if not 0<=bid<ask<=100:return None,'invalid_current_book'
        if w['ws']+300000.-t<2000.:return None,'horizon_near_expiry'
        key=(phase,bid,ask-bid,self.horizon)
        if key not in self.cache:
            paths=PATH.select(self.paths,phase,bid,ask-bid)
            self.cache[key]=(paths,EXEC.pack(paths,bid,self.horizon) if paths else None)
        paths,packed=self.cache[key]
        if not paths:return None,'no_scenario_support'
        EXEC.current_book(packed,[self.W[s]['P5'][j] for s in (0,1)],
                          [self.W[s]['S5'][j] for s in (0,1)])
        p=None if self.case=='EMPIRICAL' else self.prob[:,k,:]
        if p is not None and not np.isfinite(p).all():return None,'no_model_row'
        weight,diagnostic=PATH.weights(paths,p)
        return (packed,weight,(bid+ask)/2,diagnostic),'eligible'

    def on_step(self,k,t,ST,kpos):
        self.cancel=set();self.post=set();self.stats['steps']+=1
        # Every arm gets the same conservative last-two-seconds shutdown.
        if self.W[0]['ws']+300000.-t<2000.:
            self.cancel={(s,d) for s in (0,1) for d in range(6)};self.stats['horizon_near_expiry']+=1;return
        if self.case=='KEEP_ALL_CAP':
            self.post={(s,d) for s in (0,1) for d in range(6)};return
        context,status=self.context(k,t);self.stats[status]+=1
        if context is None:return  # keep / wait, no invented forecast
        packed,weights,mid,diagnostic=context
        state=snapshot(self.W,k,t,ST,kpos)
        action,selected,counts=choose(state,packed,weights,mid,self.horizon)
        self.stats.update(counts)
        for key,v in diagnostic.items():self.stats[key+'_sum']+=v
        for item in selected:
            key=(item['side'],item['rung'])
            (self.cancel if item['kind']=='cancel' else self.post).add(key)
            self.trace.append(dict(k=k,time=t,case=self.case,m=int(self.W[0]['m']),**item,**diagnostic))

    def decide(self,s,d):return lambda k,t,cur:(s,d) in self.cancel
    def gate(self,s,d):return lambda k:(s,d) not in self.post
