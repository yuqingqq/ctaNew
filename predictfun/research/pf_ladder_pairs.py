"""Research implementation of native f20fdb1 incremental paired-order sizing.

Order snapshots are client-known, never physical future fills. Native request
throttling remains in the engine. This mapper never reserves/net-crosses sides.
"""
import math

OFF=(None,0.)

class PairMapper:
    def __init__(self,min_usd=None):
        self.minimum=min_usd
        self.total=[None]*4
        self.filled=[0.]*4
        self.last_resizes=0
        self.last_withheld=0

    @staticmethod
    def partner(lane):return lane+4 if lane<4 else lane-4

    @staticmethod
    def active(lane,tick,current,pending,intent):
        o=current.get(lane);p=pending.get(lane)
        return (o is not None and o['pt']==tick) or (p is not None and p['key'][0]==tick and intent.get(lane)==p['key'])

    def value_ok(self,lane,tick,size):
        lots=math.floor(size*100.+1e-9)/100.
        return self.minimum is None or (tick if lane%2==0 else 100-tick)*lots/100.>=self.minimum-1e-9

    def wants(self,logical,current,pending,intent):
        if len(logical)!=4:raise ValueError('exactly four logical lanes required')
        if any(not math.isfinite(z) or z<0 or (p is None and z!=0) or (p is not None and (int(p)!=p or not 1<=p<=99)) for p,z in logical):
            raise ValueError('invalid logical quote')
        out=[OFF]*8;self.last_resizes=self.last_withheld=0
        for lane,(tick,total) in enumerate(logical):
            if tick is None:
                self.total[lane]=None;self.filled[lane]=0.;continue
            members=[(m,current[m]) for m in (lane,lane+4) if current.get(m) is not None and current[m]['pt']==tick]
            members.sort(key=lambda p:(p[1]['t_send'],p[0]))
            old=self.total[lane]
            same=old is not None and old[0]==tick and abs(old[1]-total)<1e-9
            active=any(self.active(m,tick,current,pending,intent) for m in (lane,lane+4))
            if same and active:target=total-self.filled[lane]
            else:self.filled[lane]=0.;target=total
            self.last_resizes+=int(old is not None and old[0]==tick and abs(old[1]-total)>=1e-9)
            self.total[lane]=(tick,total)
            if target<=1e-9:continue
            if members and members[0][1]['remaining']<=target+1e-9:
                m,front=members[0];out[m]=(tick,front['size']);other=self.partner(m);rest=target-front['remaining']
                if len(members)>1 and abs(members[1][1]['remaining']-rest)<1e-9 and rest>1e-9:
                    out[other]=(tick,members[1][1]['size'])
                elif rest>1e-9:
                    if rest>=1.-1e-9 and self.value_ok(lane,tick,rest):out[other]=(tick,rest)
                    else:self.last_withheld+=1
            elif self.value_ok(lane,tick,target):out[lane]=(tick,target)
            else:self.last_withheld+=1
        return tuple(out)

    def notify(self,lane,tick,qty,ended=False):
        if lane not in range(8) or not math.isfinite(qty) or qty<=0:raise ValueError('invalid known fill')
        parent=lane%4
        if not ended and self.total[parent] is not None and self.total[parent][0]==tick:self.filled[parent]+=qty

    def reset(self,lane):
        self.total[lane%4]=None;self.filled[lane%4]=0.

    def rung(self,lane,current,pending,intent):
        total=self.total[lane]
        if total is not None and any(self.active(m,total[0],current,pending,intent) for m in (lane,lane+4)):
            return total
        o=current.get(lane)
        return (o['pt'],o['size']) if o is not None else None
