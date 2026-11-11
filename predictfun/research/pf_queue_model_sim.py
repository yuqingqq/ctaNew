"""Small independent order-episode simulator and causal order-state features."""
import bisect
from dataclasses import dataclass
import numpy as np
from .pf_model_spec import SETS
from .pf_queue_model_spec import SIZE, LEASE

MARKET = SETS['ALL'] + ['side','order_price','known_remaining','inventory','other_reserved',
                         'd_bnimb250','d_pfimb250','activity_ratio']
QUEUE = ['order_age_ms','queue_initial','queue_min_seen','queue_now','queue_progress']


@dataclass
class Tape:
    tf: np.ndarray
    known: np.ndarray
    bp: np.ndarray
    ap: np.ndarray
    bs: np.ndarray
    ass: np.ndarray
    tt: np.ndarray
    ts: np.ndarray
    tp: np.ndarray
    tq: np.ndarray
    rt: np.ndarray
    x: np.ndarray
    v: np.ndarray

    def validate(self):
        for t in (self.tf,self.known,self.tt,self.rt):
            if not np.isfinite(t).all() or (np.diff(t)<0).any():raise ValueError('nonmonotone or invalid clock')
        if not ((self.ts==0)|(self.ts==1)).all() or not np.isfinite(self.tq).all() or (self.tq<0).any():raise ValueError('invalid trade')
        if not np.isfinite(self.tp).all() or ((self.tp<0)|(self.tp>100)).any():raise ValueError('invalid trade price')
        if (self.known<self.tf).any():raise ValueError('book known before venue event')
        if len({len(self.tt),len(self.ts),len(self.tp),len(self.tq)})!=1:raise ValueError('trade identity lengths')
        if len(self.rt)!=len(self.x) or len(self.rt)!=len(self.v):raise ValueError('feature identity lengths')

    def index(self,t,known=False):
        arr=self.known if known else self.tf
        i=int(np.searchsorted(arr,t,side='right')-1)
        return i if i>=0 and t-arr[i]<=1000 else -1

    def book_ok(self,i):
        return i>=0 and 0<self.bp[i,0]<self.ap[i,0]<100

    def shown(self,i,side,p):
        if i<0:return np.nan
        pp,qq=(self.bp,self.bs) if side==0 else (self.ap,self.ass)
        match=pp[i]==p
        if match.any():return float(qq[i,match][0])
        valid=pp[i][pp[i]>=0]
        if not len(valid):return np.nan
        if (p<valid[-1] if side==0 else p>valid[-1]):return np.nan
        return 0.

    def midpoint(self,t):
        i=self.index(t)
        return float((self.bp[i,0]+self.ap[i,0])/2) if self.book_ok(i) else np.nan

    def row(self,t):
        i=int(np.searchsorted(self.rt,t,side='right')-1)
        return i if i>=0 and t-self.rt[i]<=100 else -1

    def hard_stop(self,side,p,start,end):
        # Same rules for all actions, on the model's known 50ms feature clock.
        s=1 if side==0 else -1
        grid=np.arange(start,end+1,50.)
        if not len(self.rt) or not len(self.known):return start
        j=np.searchsorted(self.rt,grid,side='right')-1;i=np.searchsorted(self.known,grid,side='right')-1
        bad=(j<0)|(i<0)|(grid-self.rt[j]>100)|(grid-self.known[i]>1000)
        bad|=(self.bp[i,0]<=0)|(self.bp[i,0]>=self.ap[i,0])|(self.ap[i,0]>=100)
        best=self.bp[i,0] if side==0 else self.ap[i,0];bad|=s*(best-p)>1
        for name in ('bn_r250','bn_r1000'):
            z=self.x[j,SETS['ALL'].index(name)];bad|=np.isfinite(z)&(s*z<=-.5)
        return float(grid[np.flatnonzero(bad)[0]]) if bad.any() else end


def simulate(tape,side,p,send,decision,config,action,stop=None):
    """Three potential actions share the exact neutral prefix. Returns fills and lifecycle."""
    if side not in (0,1) or action not in ('keep','cancel','rejoin') or not send<decision<send+LEASE:
        raise ValueError('invalid episode')
    c=config;stop=tape.hard_stop(side,p,send,send+LEASE) if stop is None else stop
    orders=[];fills=[]

    def order(at,size,remove):
        i=tape.index(at)
        refused=not tape.book_ok(i) or (p>=tape.ap[i,0] if side==0 else p<=tape.bp[i,0])
        a=tape.shown(i,side,p)
        if not np.isfinite(a):refused=True
        # Pessimistic activation uncertainty is a latent fill assumption, not a model input.
        if i+1<len(tape.tf) and tape.tf[i+1]-at<=100:
            z=tape.shown(i+1,side,p)
            if np.isfinite(z):a=max(a,z)
        o=dict(at=at,remove=remove,rem=size,ahead=a,initial=a,ci=i,refused=bool(refused),size=size)
        orders.append(o);return o

    at=send+c['place']
    # Cancellation before acceptance cannot erase the still-pending create.
    requested=min(stop,decision) if action!='keep' else stop
    remove=max(requested+c['cancel'],send+95.) if requested<at else requested+c['cancel']
    old=order(at,SIZE,remove)
    if c['cancel_ack']<c['cancel']:raise ValueError('cancel acknowledged before effective')
    new_send=decision+c['cancel_ack']
    new_at=new_send+c['place'] if action=='rejoin' and stop>new_send else np.inf
    new_remove=max(stop+c['cancel'],new_send+95.) if stop<new_at else stop+c['cancel']
    ts=tape.tt-c['shift'];lo=np.searchsorted(ts,at);hi=np.searchsorted(ts,send+LEASE+c['cancel'],side='right')
    added=False
    for j in range(lo,hi):
        t=float(ts[j])
        if not added and new_at<=t:
            notified=sum(q for ft,q,_,_,_ in fills if ft+c['notice']<=decision)
            z=max(0.,SIZE-notified);o=order(new_at,z,new_remove)
            o['refused'] |= (p if side==0 else 100-p)*z<100.;added=True
        if tape.ts[j]!=side:continue
        price=float(tape.tp[j]);amount=float(tape.tq[j]);left=amount
        for k,o in enumerate(orders):
            if o['refused'] or not o['at']<=t<o['remove'] or o['rem']<=1e-12:continue
            if c['rule']=='M':
                ci=int(np.searchsorted(tape.tf,t,side='left')-1)
                for ii in range(o['ci']+1,ci+1):
                    z=tape.shown(ii,side,p)
                    if np.isfinite(z):o['ahead']=min(o['ahead'],z)
                o['ci']=max(o['ci'],ci)
            through=(price<p) if side==0 else (price>p)
            if price==p:
                used=min(amount,o['ahead']);o['ahead']-=used
                q=min(max(0.,amount-used),o['rem'],left)
            elif through:q=min(o['rem'],left);o['ahead']=0.
            else:q=0.
            if q>1e-12:
                o['rem']-=q;left-=q
                fills.append((t,q,bool(through),k,float(ts[j]+c['shift'])))
    if not added and np.isfinite(new_at):
        notified=sum(q for ft,q,_,_,_ in fills if ft+c['notice']<=decision)
        z=max(0.,SIZE-notified);o=order(new_at,z,new_remove)
        o['refused'] |= (p if side==0 else 100-p)*z<100.
    return dict(fills=fills,orders=orders,stop=stop,send=send,decision=decision,price=p,side=side)


def features(tape,side,p,send,decision,config,neutral,inventory=0.,other_reserved=SIZE):
    j=tape.row(decision);i=tape.index(decision,True)
    if j<0 or not tape.book_ok(i):return None,'missing_or_stale_state'
    if neutral['orders'][0]['refused']:return None,'postonly_or_depth_refused'
    if neutral['stop']<=decision:return None,'hard_guard_before_decision'
    notified=sum(q for t,q,_,_,_ in neutral['fills'] if t+config['notice']<=decision)
    if notified>=SIZE-1e-9:return None,'fill_known_before_decision'
    birth=tape.index(send+config['place'],True)
    initial=tape.shown(birth,side,p);now=tape.shown(i,side,p)
    vals=[tape.shown(k,side,p) for k in range(max(0,birth),i+1)]
    vals=[v for v in vals if np.isfinite(v)]
    if not vals or not np.isfinite(initial) or not np.isfinite(now):return None,'unknown_queue_estimate'
    x=dict(zip(SETS['ALL'],map(float,tape.x[j])))
    k=int(np.searchsorted(tape.rt,decision-250,side='right')-1)
    lo=int(np.searchsorted(tape.rt,decision-30000,side='left'))
    ix=SETS['ALL'].index('bn_ntr1000');past=tape.x[lo:j,ix];past=past[np.isfinite(past)]
    x.update(side=float(side),order_price=float(p),known_remaining=SIZE-notified,inventory=float(inventory),other_reserved=float(other_reserved),
        d_bnimb250=x['bn_imb']-tape.x[k,SETS['ALL'].index('bn_imb')] if k>=0 else np.nan,
        d_pfimb250=x['pf_imb1']-tape.x[k,SETS['ALL'].index('pf_imb1')] if k>=0 else np.nan,
        activity_ratio=(1+x['bn_ntr1000'])/(1+float(past.mean())) if len(past) else np.nan,
        order_age_ms=decision-send-config['place'],queue_initial=initial,queue_min_seen=min(vals),
        queue_now=now,queue_progress=initial-min(vals))
    return x,'eligible'


def reservation(result,now,config):
    """Client-side outstanding exposure, including uncertainty until late fills can be reported."""
    total=0.
    for k,o in enumerate(result['orders']):
        sent=o['at']-config['place']
        if sent>now:continue
        if o['refused'] and now>=sent+95.:continue
        # A cancel acknowledgment alone does not reveal a delayed fill notification.
        resolved=o['remove']+max(config['notice'],config['cancel_ack']-config['cancel'])
        if now>=resolved:continue
        notified=sum(q for t,q,_,generation,_ in result['fills'] if generation==k and t+config['notice']<=now)
        total+=max(0.,o['size']-notified)
    return total


def marks(tape,result,mark_t,outcome):
    m=tape.midpoint(mark_t)
    s=1 if result['side']==0 else -1;p=result['price'];fills=result['fills'];q=sum(f[1] for f in fills)
    value=s*q*(m-p) if q else 0.
    out=dict(q=q,value=value,settle=s*q*(100*outcome-p) if q else 0.,cash=-s*q*p,net=s*q,
             orders=sum(not o['refused'] and o['size']>0 for o in result['orders']),refused=sum(o['refused'] for o in result['orders']))
    for horizon in (1,5,60):
        terms=[];missing=0
        for t,z,_,_,anchor in fills:
            target=min(anchor+horizon*1000,299900.)
            mid=tape.midpoint(target)
            if not np.isfinite(mid):missing+=1
            else:terms.append(s*z*(mid-p))
        out['mo'+str(horizon)]=sum(terms) if not missing else np.nan
        out['missing_mo'+str(horizon)]=missing
    spread=[]
    for t,z,_,_,_ in fills:
        mid=tape.midpoint(t-.001)
        if np.isfinite(mid):spread.append(s*z*(mid-p))
    out['spread']=sum(spread) if len(spread)==len(fills) else np.nan
    out['fills']=len(fills);out['through_q']=sum(z for _,z,through,_,_ in fills if through)
    return out
