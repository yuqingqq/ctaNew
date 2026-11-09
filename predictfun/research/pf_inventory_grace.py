"""Inventory-aware restrictions on the previously frozen soft-exit policy."""
import math
from .pf_soft_pull import SoftPull


def reserved_side(p,side,now):
    """Same-side amount which can still fill/report, including pending cancels."""
    return sum(o.rem for o in p.orders if o.lane%2==side and o.rem>0 and o.sent<=now and (o.release_at is None or now<o.release_at))


def inventory_reason(mode,net,lane,remaining,reserved):
    if mode not in ['all','individual','reserved']:raise ValueError('unknown inventory mode')
    if not all(math.isfinite(x) for x in [net,remaining,reserved]) or remaining<0 or reserved+1e-9<remaining:raise ValueError('invalid inventory input')
    if mode=='all':return None
    sign=1 if lane%2==0 else -1
    if net*sign>=0:return 'inventory_not_reducing'
    if remaining>abs(net)+1e-9:return 'inventory_individual_overshoot'
    if mode=='reserved' and reserved>abs(net)+1e-9:return 'inventory_reserved_overshoot'
    return None


class InventoryGrace(SoftPull):
    def __init__(self,delta=.01,mode='all',max_ms=100.):
        super().__init__(delta,max_ms=max_ms)
        if mode not in ['all','individual','reserved']:raise ValueError('unknown inventory mode')
        self.mode=mode;self.checks=[];self.p=None;self.at=None

    def eligible(self,row,lane,o,opening=False):
        reason=super().eligible(row,lane,o,opening)
        if reason:return reason
        amount=reserved_side(self.p,lane%2,self.at);net=float(row['net'])
        reason=inventory_reason(self.mode,net,lane,o.rem,amount)
        self.checks.append(dict(t=self.at,lane=lane,order_id=o.id,net=net,remaining=o.rem,reserved=amount,
            input_index=row['input_index'],source_kind=row.get('kind'),source_ordinal=row.get('ordinal'),
            opening=opening,status='allow' if reason is None else reason))
        return reason

    def observe(self,now,row,p):
        self.p=p;self.at=now
        super().observe(now,row,p)

    def apply(self,now,row,p):
        self.p=p;self.at=now;first=len(self.actions)
        before=[reserved_side(p,side,now) for side in [0,1]]
        super().apply(now,row,p)
        for action in self.actions[first:]:
            action.update(start_net=float(row['net']),reserved_before=before[action['lane']])
            if action['status']=='retained':
                o=next(o for o in p.orders if o.id==action['order_id']);action['retained_shares']=o.rem
        # A quote newly placed on another lane can consume reducing room in the
        # same decision. Enforce the aggregate check again after planning.
        if self.mode!='all':self.observe(now,row,p)


def exposure(p,start,end,raw,net):
    """Exact conditional active/off-target share-time and worst-case range.

    Net is factual, not simulated. This is exposure accounting, never variance.
    Opposite sides do not net in the worst-case endpoint calculation.
    """
    if end<start or not math.isfinite(net):raise ValueError('invalid exposure span/state')
    out=dict(off_reducing_share_ms=0.,off_other_share_ms=0.,active_range_extra_share_ms=0.,reserved_range_extra_share_ms=0.)
    for o in p.orders:
        lo=max(start,o.accepted);hi=min(end,o.cancel_at if o.cancel_at is not None else end)
        if hi<=lo or o.rem<=0:continue
        tick,size=raw[o.lane]
        if tick==o.tick and size>0:continue
        key='off_reducing_share_ms' if net*(1 if o.lane%2==0 else -1)<0 else 'off_other_share_ms'
        out[key]+=o.rem*(hi-lo)
    for reserved in [False,True]:
        events=[(start,0.,0.),(end,0.,0.)]
        for o in p.orders:
            if o.rem<=0:continue
            lo=max(start,o.sent if reserved else o.accepted)
            stop=o.release_at if reserved else o.cancel_at
            hi=min(end,stop if stop is not None else end)
            if hi<=lo:continue
            up=o.rem if o.lane%2==0 else 0.;down=o.rem if o.lane%2 else 0.
            events.extend([(lo,up,down),(hi,-up,-down)])
        up=down=total=0.;last=start
        for t,du,dd in sorted(events):
            total+=(t-last)*max(0.,abs(net+up)-abs(net),abs(net-down)-abs(net))
            up+=du;down+=dd;last=t
        out['reserved_range_extra_share_ms' if reserved else 'active_range_extra_share_ms']=total
    return out
