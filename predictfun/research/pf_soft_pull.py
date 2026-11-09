"""Bounded retention of accepted small touch orders at a soft prediction exit.

Only research mechanics: relaxed quotes come from the exact native function at
factual state. No placements on probe events and no changed fill feedback.
"""
import math

ARMS={'reference':0.,'exit005':.005,'exit01':.01,'exit02':.02,'exit04':.04}
OFF=[(None,0.)]*4


def lanes(row,prefix='raw'):
    return [(None if row[f'{prefix}_tick{i}']<0 else int(row[f'{prefix}_tick{i}']),float(row[f'{prefix}_size{i}'])) for i in range(4)]


class SoftPull:
    def __init__(self,delta,line=.0492,max_ms=100.):
        if not all(math.isfinite(x) for x in (delta,line,max_ms)) or delta<0 or max_ms<=0:raise ValueError('bad soft-exit settings')
        self.delta=delta;self.line=line;self.max_ms=max_ms
        self.previous=list(OFF);self.holds={};self.actions=[]

    def eligible(self,row,lane,o,opening=False):
        if row['held']:return 'held'
        if row['binance_bid' if lane==0 else 'binance_ask']:return 'binance'
        v=row['v_bid' if lane==0 else 'v_ask']
        if v is None or not math.isfinite(v):return 'missing_score'
        if v<=self.line-self.delta:return 'lower_line'
        if opening and v>self.line:return 'not_soft_exit'
        q=lanes(row,f'd{self.delta:g}')[lane]
        if q[0]!=o.tick or q[1]+1e-9<o.rem:return 'native_price_size_guard'
        raw=lanes(row)
        if raw[lane+2][0]==o.tick and raw[lane+2][1]>0:return 'duplicate_rung'
        return None

    def close(self,lane,now,reason,p,cancel=True):
        h=self.holds.pop(lane);a=self.actions[h['action']]
        o=next((o for o in p.orders if o.id==h['id']),None)
        if cancel and o is not None:p.cancel(o,now,'soft_'+reason)
        a.update(end=now,end_reason=reason,retained_ms=now-a['start'])

    def expire(self,now,p):
        # Called BEFORE integration; scheduling at the deadline lets integrate()
        # account for the exact confirmation time even inside a long event gap.
        for lane,h in list(self.holds.items()):
            if h['deadline']<=now:self.close(lane,h['deadline'],'deadline',p)

    def observe(self,now,row,p):
        for lane,h in list(self.holds.items()):
            o=next((o for o in p.orders if o.id==h['id']),None)
            reason='order_unavailable' if o is None or o.cancel_at is not None or o.rem<=0 else self.eligible(row,lane,o)
            if reason:self.close(lane,now,reason,p)

    def apply(self,now,row,p):
        raw=lanes(row);want=list(raw)
        # A normal quote-on event ends the special grace; ordinary size/price
        # logic resumes and may still cancel this ticket.
        for lane in list(self.holds):
            if raw[lane][0] is not None and raw[lane][1]>0:
                h=self.holds[lane];o=next(o for o in p.orders if o.id==h['id'])
                self.close(lane,now,'same_price_reopen' if o.tick==raw[lane][0] else 'other_price_reopen',p,cancel=False)
        self.observe(now,row,p)
        for lane in (0,1):
            if self.previous[lane][0] is None or raw[lane][0] is not None:continue
            aa=dict(action=len(self.actions),lane=lane,start=now,old_tick=self.previous[lane][0],
                v=row['v_bid' if lane==0 else 'v_ask'],midpoint=row['midpoint'],input_index=row['input_index'],
                spread=row['ask']-row['bid'],status='reference_cancel',end=now,end_reason='not_retained',retained_ms=0.)
            self.actions.append(aa)
            if self.delta==0:continue
            available=[o for o in p.orders if o.lane==lane and o.cancel_at is None and o.accepted<=now and o.slot=='base' and 0<o.rem<=5 and o.size<=5 and o.tick==aa['old_tick']]
            if len(available)!=1:aa['status']='no_accepted_small_base';continue
            o=available[0];reason=self.eligible(row,lane,o,opening=True)
            if reason:aa['status']=reason;continue
            aa.update(status='retained',order_id=o.id,accepted=o.accepted,end=None,end_reason=None,retained_ms=None)
            self.holds[lane]=dict(id=o.id,deadline=now+self.max_ms,action=aa['action'])
        for lane,h in self.holds.items():
            o=next(o for o in p.orders if o.id==h['id'])
            if o.cancel_at is not None or o.accepted>now:raise ValueError('retaining unavailable order')
            want[lane]=(o.tick,o.size)
        before=len(p.commands);p.plan(now,want)
        for c in p.commands[before:]:
            if c['kind']=='place' and raw[c['lane']][0] is None:raise ValueError('new order during guard-off grace')
        self.previous=raw

    def finish(self,end,p):
        for lane in list(self.holds):self.close(lane,end,'window_censored',p,cancel=False)
        return self.actions
