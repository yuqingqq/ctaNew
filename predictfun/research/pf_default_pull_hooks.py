"""Actual-order default quoting and separate cancellation policy layer."""
import collections
import copy
import hashlib
import numpy as np
from . import pf_default_pull_spec as S
from . import pf_order_policy_hooks as OLD
from .pf_order_policy_inputs import FEATURES,MARKET
from .pf_model_spec import SETS
from .pf_own_order_warmup_v3 import CausalWindow
from .pf_default_pull_shadow import paired_leg

EXTRA=['probe_age','reset_q','reset_value1','reset_value5','keep_value','reset_value','first_cancel',
       'score_count','pull','thin_ratio','life_ms','spread_value','spread_missing']
NUMERIC=list(dict.fromkeys(OLD.NUMERIC+EXTRA))
OLD.NUMERIC=NUMERIC  # schema only, process-local; frozen file remains unchanged
OLD.Window=CausalWindow

def remaining(o):return max(0.,o['size']-o.get('kq',0.))

def probe_age(gid):
    return S.PROBE_AGES[int.from_bytes(hashlib.sha256(gid.encode()).digest()[:4],'big')%len(S.PROBE_AGES)]

def pressure(x,thin):
    req=['pf_imb1','bn_imb','bn_flow1000','d_bnimb250','activity_ratio']
    # The signed flow feature is verified against the feature schema at startup.
    if not all(np.isfinite(x[k]) for k in req) or not np.isfinite(thin):return None
    sg=1 if x['side']==0 else -1
    return bool(thin<=.8 or sg*x['pf_imb1']<=-1/3 or sg*x['d_bnimb250']<=-.3
                or (x['activity_ratio']>=2 and sg*x['bn_flow1000']<0))

def thin_ratio(w,t,side,p):
    i=w.index(t);j=w.index(t-250)
    if i<0 or j<0:return np.nan
    now=w.shown(i,side,p);old=w.shown(j,side,p)
    if not np.isfinite(now) or not np.isfinite(old) or old<=0:return np.nan
    return now/old

class Manager(OLD.Manager):
    def __init__(self,*args,models=None,**kwargs):
        super().__init__(*args,**kwargs);self.models=models or {}
    def window(self,mid,ws,rules,F,kf,trades,events,guard,value,line):
        mode=next((k for k,v in S.MODES.items() if v and v in rules),'reference')
        return Hook(self,mode,mid,ws,F,kf,trades,events,guard,value,line)

class Hook(OLD.Hook):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.probes=[];self.cool=[-np.inf,-np.inf];self.last_score={};self.accepted={}
        self.presence=0.;self.last_presence=self.ws;self.present_sides=0
        self.calls=collections.Counter()
        if self.mode!='reference':self.wakeups=np.arange(self.ws,self.ws+240000,50.).tolist()

    def seed_flag(self,*args):return False
    def seed_size(self,t,side,normal):return normal
    def allow_new(self,*args):return True
    def action(self,t,lane,p,size,cur,pending,net,refs):return p,size,False

    def sent(self,rec,net=0.):
        super().sent(rec,net)
        r=self.records[rec['_gid']];r.update(probe_age=probe_age(rec['_gid']),score_count=0.,pull=0.)

    def activated(self,o,rec):
        super().activated(o,rec);self.accepted[o['_gid']]=o

    def state(self,t,side,o,net,refs):
        j=self.w.row(t);i=self.w.index(t)
        if j<0 or i<0:return None
        f=self.w.F[i]
        if not 0<f['b1_p']<f['a1_p']<100:return None
        x=dict(zip(SETS['ALL'],map(float,self.w.x[j])))
        if not np.isfinite(x['bn_age']) or x['bn_age']+t-self.w.rt[j]>1000:return None
        v=self.value(t,side)
        if v is None or not np.isfinite(v):return None
        if o is None:
            o=dict(side=side,pt=f['b1_p'] if side==0 else f['a1_p'],size=5.,t0=t,kq=0.)
        other=sum(remaining(r) for r in refs if r['side']!=side)
        x=self.w.features(t,o,net,other,v,self.line(t))
        if x is None:return None
        tr=thin_ratio(self.w,t,side,o['pt']);pr=pressure(x,tr)
        if pr is None:return None
        return x,tr,pr

    def price_safe(self,t,side,p):
        i=self.w.index(t)
        if i<0 or not self.ws<=t<self.ws+240000:return False
        f=self.w.F[i];b,a=f['b1_p'],f['a1_p'];sg=1 if side==0 else -1
        if not 0<b<a<100 or not 1<=p<=99:return False
        return bool((p<a if side==0 else p>b) and sg*((b if side==0 else a)-p)<=2)

    def record(self,t,o,net,refs,state,status):
        row=self.records[o['_gid']];x,tr,pr=state
        row.update(x);row.update(decision=t,known_remaining=remaining(o),thin_ratio=tr,status=status)
        return row

    def targets(self,t,original,known,inflight,refs,net):
        if self.mode=='reference':return tuple(original)
        # Presence is a client-observed diagnostic, not venue uptime.
        self.presence+=(t-self.last_presence)*self.present_sides
        self.last_presence=t;self.present_sides=sum(o is not None for o in known.values())
        result=[(None,0.)]*4
        for side in (0,1):
            fl=[r for r in inflight[side] if not r['cancelled']]
            o=fl[-1] if fl else known[side];pending=bool(fl);sg=1 if side==0 else -1
            state=self.state(t,side,None if pending else o,net,refs)
            if state is None:self.calls['unknown']+=1;continue
            x,tr,pr=state
            guard=self.guard(t)[side] or x['guard_margin']<=0
            trigger=pr if self.mode=='pressure' else x['guard_v']<=-.10 if self.mode=='floor' else guard if self.mode=='matched_guard' else False
            if trigger:
                self.cool[side]=max(self.cool[side],t+(0 if self.mode=='matched_guard' else S.COOLDOWN))
                self.calls['trigger']+=1;continue
            if o is not None and not self.price_safe(t,side,o['pt']):o=None;pending=False
            rem=remaining(o) if o is not None else 0.
            others=sum(remaining(r) for r in refs if r is not o and r['side']==side)
            capacity=10-sg*net-others
            if t>=self.ws+60000:
                capacity=min(capacity,max(0.,-sg*net-others))
            if o is not None and rem>capacity+1e-9:
                self.calls['capacity_pull']+=1;continue
            if o is not None and not pending and remaining(o)>1e-9:
                # Recompute state for the actual old price after a price replacement decision.
                state=self.state(t,side,o,net,refs)
                if state is None:continue
                age=t-o['t0'];gid=o['_gid'];row=self.records[gid]
                if self.parent.phase=='build' and self.mode=='default' and not np.isfinite(row['decision']) and age>=row['probe_age']:
                    row=self.record(t,o,net,refs,state,'eligible')
                    row['cancel_effective']=t+self.c['cancel'];self.probes.append((row['cancel_effective'],o))
                if self.mode in ('market','queue','queue_cost') and age>=100 and t-self.last_score.get(gid,-np.inf)>=100:
                    self.last_score[gid]=t;x=state[0]
                    which='market' if self.mode=='market' else 'queue';cols=MARKET if which=='market' else FEATURES
                    score=float(self.parent.models[which].predict(np.array([[x[k] for k in cols]]),num_threads=1)[0])
                    row['score_count']+=1
                    threshold=-.01*x['remaining'] if self.mode=='queue_cost' else 0.
                    pull=score<threshold
                    if not np.isfinite(row['decision']) or pull:
                        row=self.record(t,o,net,refs,state,'pulled' if pull else 'scored_keep');row['score']=score
                    if pull:
                        row['pull']=1.;self.cool[side]=t+S.COOLDOWN;self.calls['model_pull']+=1;continue
            if t<self.cool[side]:continue
            if o is not None:
                result[side]=(o['pt'],o['size']);continue
            # Full pending/cancel reservations remain until the engine's known lifecycle resolves them.
            z=min(5.,capacity)
            if z<=1e-9:continue
            i=self.w.index(t);p=self.w.F[i]['b1_p'] if side==0 else self.w.F[i]['a1_p']
            if (p if side==0 else 100-p)*z<100-1e-9:continue
            result[side]=(p,z)
        return tuple(result)

    def cancelling(self,cur,t_rm):
        row=self.records[cur['_gid']]
        if not np.isfinite(row['first_cancel']):row['first_cancel']=t_rm
        return False

    def snapshot(self,t,cur,others,pending):
        row=self.records[cur['_gid']]
        self.snapshots.append((row,copy.copy(cur),[copy.copy(o) for o in others if o is not cur and o['side']==cur['side']],
            [copy.copy(o) for o in pending if o['side']==cur['side']],t))

    def label(self,row,target,others,pending,start):
        end=min(row['decision']+S.LABEL_MS,self.ws+240000)
        for t in np.arange(row['decision']+50,end+1,50.):
            if not self.price_safe(t,target['side'],target['pt']) or self.state(t,target['side'],target,row['inventory'],[]) is None:
                end=min(end,t);break
        end+=self.c['cancel'];row['shadow_end']=end;row['postcancel_rem']=target['rem']
        lo=np.searchsorted(self.tt,start);hi=np.searchsorted(self.tt,end,side='left')
        trades=self.trades[lo:hi]
        keep=paired_leg(self.w,target,others,pending,trades,start,end,self.c['queue'])
        reset_send=row['decision']+S.COOLDOWN
        # Only unavoidable prefix fills can have happened on the CANCEL branch.
        notified=sum(q for t,_,_,q in self.genfills[row['id']] if t<start and t+self.c['notice']<=reset_send)
        reset_size=max(0.,row['initial_size']-notified)
        reset=paired_leg(self.w,target,others,pending,trades,start,end,self.c['queue'],reset_send+self.c['place'],reset_size)
        values=[]
        for fs,pre in [(keep,'label'),(reset,'reset')]:
            row[pre+'_q']=sum(q for _,q,_,_ in fs)
            for h in (1,5):
                terms=[(1 if target['side']==0 else -1)*q*(self.w.mid(t+h*1000)-p) for t,q,p,_ in fs]
                row[f'{pre}_value{h}']=sum(terms) if all(np.isfinite(terms)) else np.nan
            dq=(1 if target['side']==0 else -1)*row[pre+'_q'];net=row['inventory']
            values.append(.5*(row[pre+'_value1']+row[pre+'_value5'])-S.RISK_C*((net+dq)**2-net**2))
        row['keep_value'],row['reset_value']=values
        row['label']=values[0]-values[1]
        if not np.isfinite(row['label']):row['status']='missing_label_mark'

    def finish(self,*args):
        # Add diagnostics before base finish flushes rows. Labels/accounting use base independent bookkeeping.
        for gid,row in self.records.items():
            fs=self.genfills[gid];stop=self.ws+240100
            effective=row['first_cancel'];filled=max((t for t,_,_,_ in fs),default=np.inf)
            if sum(q for _,_,_,q in fs)<row['initial_size']-1e-9:filled=np.inf
            row['life_ms']=max(0.,min(stop,effective if np.isfinite(effective) else stop,filled)-row['t_act']) if gid in self.accepted else 0.
            terms=[(1 if s==0 else -1)*q*(self.w.mid(t-.001)-p) for t,s,p,q in fs]
            row['spread_missing']=sum(not np.isfinite(v) for v in terms)
            row['spread_value']=sum(terms) if all(np.isfinite(terms)) else np.nan
            if self.mode=='default' and self.parent.phase=='build' and not np.isfinite(row['decision']) and row['status']!='postonly_refused':row['status']='ended_before_probe_or_unknown'
        super().finish(*args)
        self.parent.windows[-1].update(client_presence_s=self.presence/1000,model_pulls=self.calls['model_pull'],unknown_decisions=self.calls['unknown'])
