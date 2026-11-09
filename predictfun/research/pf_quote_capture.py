"""Research quote planner: retain base priority, gate re-entry, respect reservations.

No exchange adapter, fill model, or PnL. Targets are externally supplied policy
decisions; a closed guard always cancels immediately. Quantity reductions become
cancels on a no-amend venue; pending cancels remain reserved through their grace.
"""
from dataclasses import dataclass
import math


@dataclass
class Order:
    id: int
    lane: int
    slot: str
    tick: int
    size: float
    rem: float
    sent: float
    accepted: float
    cancel_at: float | None = None
    release_at: float | None = None


def token_price(lane,tick):
    if lane not in range(4) or type(tick) is not int or not 1<=tick<=99:raise ValueError('bad lane/price')
    return (tick if lane%2==0 else 100-tick)/100


def components(lane,tick,target,base=5.,layered=True):
    if not math.isfinite(target) or not math.isfinite(base) or target<0 or base<=0:raise ValueError('bad size')
    p=token_price(lane,tick)
    if target*p<1-1e-9:return {}
    if not layered or target<=base or base*p<1-1e-9:return {'base':target}
    extra=target-base
    # Do not discard part of the target when the second ticket cannot meet min-notional.
    if extra*p<1-1e-9:return {'base':target}
    return {'base':base,'extra':extra}


class EntryGate:
    """Buffer entry only. Existing exits/hard guards are never delayed."""
    def __init__(self,delta=0.):
        if not math.isfinite(delta) or delta<0:raise ValueError('bad buffer')
        self.delta=delta;self.on=[False]*4;self.tick=[None]*4

    def apply(self,lanes,values,line,now,score_at):
        if len(lanes)!=4 or len(values)!=2:raise ValueError('bad decision shape')
        fresh=math.isfinite(score_at) and 0<=now-score_at<=100
        result=[]
        for lane,(tick,size) in enumerate(lanes):
            if tick is None or size<=0:
                self.on[lane]=False;self.tick[lane]=None;result.append((None,0.));continue
            if tick!=self.tick[lane]:self.on[lane]=False
            v=values[lane%2]
            if self.delta==0:
                allow=True  # exact pass-through baseline, including missing reconstructed scores
            elif not fresh or not math.isfinite(v) or v<=line:
                allow=False
            else:
                allow=self.on[lane] or v>line+self.delta
            self.on[lane]=allow;self.tick[lane]=tick
            result.append((tick,size) if allow else (None,0.))
        return result


class QuotePlanner:
    def __init__(self,layered=True,base=5.,cap=150.,inflight=3,place_ms=40.,cancel_ms=22.,pending_cancel_ms=95.,grace_ms=1000.,incremental=False):
        if base<=0 or cap<=0 or inflight<1 or min(place_ms,cancel_ms,pending_cancel_ms,grace_ms)<0:raise ValueError('invalid planner limits')
        if incremental and not layered:raise ValueError('incremental mode requires retention')
        self.layered=layered;self.incremental=incremental;self.base=base;self.cap=cap;self.inflight=inflight
        self.place_ms=place_ms;self.cancel_ms=cancel_ms;self.pending_cancel_ms=pending_cancel_ms;self.grace_ms=grace_ms
        self.orders=[];self.next_id=0;self.now=-math.inf;self.commands=[]
        self.refused={'capacity':0,'inflight':0,'minimum':0}
        self.stats=dict(places=0,cancels=0,base_resize_cancels=0,kept_base_size_switches=0)
        self.previous=[(None,0.)]*4

    def advance(self,now):
        if not math.isfinite(now) or now<self.now:raise ValueError('clock regressed')
        self.now=now
        self.orders=[o for o in self.orders if o.rem>1e-9 and (o.release_at is None or now<o.release_at)]

    def reserved(self,side):
        return sum(o.rem for o in self.orders if o.lane%2==side)

    def fill(self,order_id,size):
        # Synthetic/observed feedback only. Never manufacture a historical fill time.
        matches=[o for o in self.orders if o.id==order_id]
        if len(matches)!=1 or not math.isfinite(size) or not 0<size<=matches[0].rem+1e-9:raise ValueError('invalid exact fill feedback')
        matches[0].rem=max(0.,matches[0].rem-size)

    def cancel(self,o,now,reason):
        if o.cancel_at is not None:return
        end=max(now+self.cancel_ms,o.sent+self.pending_cancel_ms) if now<o.accepted else now+self.cancel_ms
        o.cancel_at=end;o.release_at=end+self.grace_ms
        self.commands.append(dict(t=now,kind='cancel',id=o.id,lane=o.lane,slot=o.slot,tick=o.tick,size=o.rem,reason=reason))
        self.stats['cancels']+=1
        if o.slot=='base' and reason=='resize':self.stats['base_resize_cancels']+=1

    def plan(self,now,lanes,known_net=0.,reduce_only=False,hard_stop=False):
        self.advance(now)
        if len(lanes)!=4 or not math.isfinite(known_net):raise ValueError('bad policy input')
        if hard_stop:lanes=[(None,0.)]*4
        targets=[]
        for lane,(tick,size) in enumerate(lanes):
            if not math.isfinite(size) or size<0 or (tick is not None and int(tick)!=tick):raise ValueError('bad desired quote')
            if tick is None or size==0:targets.append({});continue
            split=self.layered
            if self.incremental:
                split=any(o.lane==lane and o.slot=='base' and o.tick==tick and o.cancel_at is None and o.size<=self.base for o in self.orders)
            target=components(lane,int(tick),size,self.base,split)
            if not target:self.refused['minimum']+=1
            targets.append(target)
        kept={};switches=[]
        # Cancel first; the quantity still reserves capacity until confirmation+grace.
        for lane,target in enumerate(targets):
            tick,size=lanes[lane];oldtick,oldsize=self.previous[lane]
            candidates=[o for o in self.orders if o.lane==lane and o.cancel_at is None]
            reducing=(lane%2==0 and known_net<0) or (lane%2==1 and known_net>0)
            if reduce_only and not reducing:target={};targets[lane]={}
            for o in sorted(candidates,key=lambda o:o.sent):
                slot=(lane,o.slot)
                desired=target.get(o.slot)
                compatible=(desired is not None and o.tick==tick and slot not in kept and
                            (abs(o.size-desired)<1e-9 or (self.layered and o.size>desired and o.rem<=desired+1e-9)))
                if compatible:kept[slot]=o
                else:
                    reason='pull' if not target else 'reprice' if o.tick!=tick else 'resize'
                    self.cancel(o,now,reason)
            if tick==oldtick and size!=oldsize and oldsize>0 and (lane,'base') in kept:
                switches.append(lane)
        # Aggregate reduce-only room is shared across touch/ladder and all pending cancels.
        for side in (0,1):
            if reduce_only:
                budget=abs(known_net) if (side==0 and known_net<0) or (side==1 and known_net>0) else 0.
                # A previously cancelled ticket can still fill or report a late fill.
                # Do not retain a new reducing ticket against the same inventory.
                budget=max(0.,budget-sum(o.rem for o in self.orders if o.lane%2==side and o.cancel_at is not None))
                active=sorted([o for o in self.orders if o.lane%2==side and o.cancel_at is None],key=lambda o:(o.slot!='extra',-o.sent))
                future=sum(o.rem for o in active)
                for o in active:
                    if future<=budget+1e-9:break
                    self.cancel(o,now,'reduce_only');future-=o.rem;kept.pop((o.lane,o.slot),None)
        self.stats['kept_base_size_switches']+=sum((lane,'base') in kept for lane in switches)
        # Touch before ladder; base before top-up. Each new order passes the same side budget.
        for lane,target in enumerate(targets):
            for slot,size in target.items():
                if (lane,slot) in kept:continue
                side=lane%2
                room=(self.cap-known_net if side==0 else self.cap+known_net)-self.reserved(side)
                if reduce_only:
                    reducing=(side==0 and known_net<0) or (side==1 and known_net>0)
                    room=min(room,(abs(known_net) if reducing else 0.)-self.reserved(side))
                if size>room+1e-9:self.refused['capacity']+=1;continue
                pending=sum(o.lane%2==side and now<o.accepted for o in self.orders)
                if pending>=self.inflight:self.refused['inflight']+=1;continue
                tick=int(lanes[lane][0]);self.next_id+=1
                o=Order(self.next_id,lane,slot,tick,size,size,now,now+self.place_ms)
                self.orders.append(o);kept[(lane,slot)]=o
                self.commands.append(dict(t=now,kind='place',id=o.id,lane=lane,slot=slot,tick=tick,size=size,reason='target'))
                self.stats['places']+=1
        self.previous=list(lanes)
        return self.commands

    def active(self,now):
        return [o for o in self.orders if o.rem>1e-9 and o.accepted<=now and (o.cancel_at is None or now<o.cancel_at)]

    def integrate(self,start,end):
        """Exact no-fill order-time integrals between policy updates, including transitions."""
        if end<start:raise ValueError('negative span')
        age=shares=0.;occupied=0.;base_occupied=0.;oldest_age=0.;reserved_ms=0.
        for o in self.orders:
            reserved_ms+=o.rem*max(0.,min(end,o.release_at if o.release_at is not None else end)-max(start,o.sent))
        for lane in range(4):
            spans=[];bases=[];life=[]
            for o in self.orders:
                if o.lane!=lane:continue
                lo=max(start,o.accepted);hi=min(end,o.cancel_at if o.cancel_at is not None else end)
                if hi<=lo:continue
                age+=o.rem*((hi-o.accepted)**2-(lo-o.accepted)**2)/2
                shares+=o.rem*(hi-lo);spans.append((lo,hi));life.append((lo,hi,o.accepted))
                if o.slot=='base':bases.append((lo,hi))
            def union(v):
                total=0.;right=-math.inf
                for lo,hi in sorted(v):total+=max(0.,hi-max(lo,right));right=max(right,hi)
                return total
            occupied+=union(spans);base_occupied+=union(bases)
            cuts=sorted({x for a,b,_ in life for x in (a,b)})
            for a,b in zip(cuts[:-1],cuts[1:]):
                born=[s for l,h,s in life if l<=a and h>=b]
                if born:
                    s=min(born);oldest_age+=((b-s)**2-(a-s)**2)/2
        return dict(share_ms=shares,age_share_ms2=age,quoted_lane_ms=occupied,base_lane_ms=base_occupied,
                    oldest_lane_age_ms2=oldest_age,reserved_share_ms=reserved_ms)
