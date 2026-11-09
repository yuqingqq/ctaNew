"""Causal entry-only policies for recorded quote pulls and rapid reopening.

These policies never retain an order after its upstream quote target is pulled.
Cancellation-ack mode reads only whether the modeled acknowledgement has arrived,
not how far away that future acknowledgement is. No fills or price forecasts.
"""
import math

OFF=(None,0.)
ARMS={
    'reference':('pass',0.),
    'cancel_ack':('cancel_ack',0.),
    'reentry25':('reentry',25.),
    'reentry50':('reentry',50.),
    'reentry100':('reentry',100.),
    'all_entry50':('all_entry',50.),
}


class ReentryGate:
    def __init__(self,mode='pass',confirm_ms=0.,rapid_ms=100.):
        if mode not in {'pass','cancel_ack','reentry','all_entry'}:raise ValueError('unknown entry rule')
        if not all(math.isfinite(x) and x>=0 for x in (confirm_ms,rapid_ms)):raise ValueError('invalid delay')
        self.mode=mode;self.confirm_ms=confirm_ms;self.rapid_ms=rapid_ms
        self.now=-math.inf;self.current=[None]*4;self.last_pull=[None]*4;self.actions=[]

    def apply(self,now,lanes,orders=()):
        if not math.isfinite(now) or now<self.now or len(lanes)!=4:raise ValueError('invalid entry clock/shape')
        self.now=now;out=[]
        for lane,(tick,size) in enumerate(lanes):
            if not math.isfinite(size) or size<0 or (tick is None and size!=0) or (tick is not None and (type(tick) is not int or not 1<=tick<=99)):
                raise ValueError('invalid quote intent')
            on=tick is not None and size>0;cur=self.current[lane]
            if cur is not None and (not on or tick!=cur['tick']):
                cur.update(end=now,end_reason='pull' if not on else 'reprice')
                if not on:self.last_pull[lane]=(cur['tick'],now)
                self.current[lane]=None
            if not on:out.append(OFF);continue
            if self.current[lane] is None:
                prior=self.last_pull[lane]
                rapid=cur is None and prior is not None and prior[0]==tick and now-prior[1]<=self.rapid_ms
                item=dict(action=len(self.actions),lane=lane,tick=tick,start=now,end=None,end_reason=None,
                    start_reason='reprice' if cur is not None else 'same_price_reentry' if rapid else 'other_entry',
                    rapid=rapid,admitted_at=None)
                self.current[lane]=item;self.actions.append(item)
            cur=self.current[lane]
            allow=cur['admitted_at'] is not None
            if not allow:
                if self.mode=='all_entry' or (self.mode=='reentry' and cur['rapid']):
                    allow=now-cur['start']>=self.confirm_ms
                elif self.mode=='cancel_ack' and cur['rapid']:
                    allow=not any(o.lane==lane and o.tick==tick and o.cancel_at is not None and now<o.cancel_at and o.rem>1e-9 for o in orders)
                else:allow=True
            if allow and cur['admitted_at'] is None:cur['admitted_at']=now
            out.append((tick,size) if allow else OFF)
        return out

    def finish(self,end):
        if end<self.now:raise ValueError('end precedes last decision')
        for cur in self.current:
            if cur is not None:cur.update(end=end,end_reason='window_censored')
        return self.actions
