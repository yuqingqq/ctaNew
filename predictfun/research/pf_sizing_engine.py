"""Versioned experiment adapter over SHA-pinned v3. No production/frozen edit.

Common execution approximation: net-share gate, lot floor, minimum notional,
and backoff on risk refusal. HCAP clips against that same known-state gate.
SHYST changes only size threshold transitions; quote guards remain unchanged.
"""
import math
import os
from .pf_sizing_study import HERE, context, sha

BASE_SHA='d678e7c3c4e47c6d60ebc031edf8d7fc7317210d5e58ff301d92d7ae0ece5f17'
MARGIN=0.01

def lots(q):
    if not math.isfinite(q) or q<0:raise ValueError('invalid order quantity')
    return math.floor((q+1e-10)*100)/100

def headroom(net,side,opens,cap=150):
    if side not in (0,1) or cap<=0:raise ValueError('invalid cap or side')
    pending=sum(max(0.,o['size']-o.get('kq',0.)) for o in opens if o['side']==side)
    return max(0.,cap-(net if side==0 else -net)-pending)

def size_state(high,v,line,margin=MARGIN):
    if v is None:return False
    return v>=line-margin if high else v>=line+margin

def replace(s,a,b,count=1):
    if s.count(a)!=count:raise ValueError('engine anchor mismatch: '+a[:60])
    return s.replace(a,b,1)

def source():
    p=HERE/'pf_step4_ladder_v3.py'
    if sha(p)!=BASE_SHA:raise ValueError('wrong frozen v3 source')
    s=p.read_text()
    s=replace(s,'    net = 0.0; knet = 0.0; cash = 0.0; kcash = 0.0; fills = []; withheld = collections.Counter()',
      '    size_high = [False, False]\n    net = 0.0; knet = 0.0; cash = 0.0; kcash = 0.0; fills = []; withheld = collections.Counter()')
    old="            return hi if (v is not None and v >= minute_line(par, rel)) else lo"
    new="""            if 'SHYST' in rules:
                before = size_high[side]
                size_high[side] = size_state(before, v, minute_line(par, rel))
                live_ctr['size_state_changes'] += int(before != size_high[side])
                return hi if size_high[side] else lo
            return hi if (v is not None and v >= minute_line(par, rel)) else lo"""
    s=replace(s,old,new)
    # Same event generation formula for thresholds +/- margin, no indiscriminate grid retries.
    old="    for t in sorted({ws + stop * 1000.0, ws + 240_000, ws + 300_000}):"
    new="""    if 'SHYST' in rules:
        for P_ in (PREDVB, PREDVA):
            v_ = P_.get(mid)
            if v_ is not None:
                for delta_ in (-MARGIN, MARGIN):
                    shifted_ = {mid: (v_[0], v_[1] - delta_)}
                    for tt_ in line_flips(mid, ws, ws, ws + 300_000, SZ['par'], shifted_):
                        ev.append((tt_, 2, seq, 'D', None)); seq += 1
    for t in sorted({ws + stop * 1000.0, ws + 240_000, ws + 300_000}):"""
    s=replace(s,old,new)
    # Clip after cancelling old orders: reservations include them until acknowledged.
    old="            wc = worst_case(dict(side=side, pt=p, rem=zsz))"
    new="""            wanted_size = zsz
            room = headroom(knet, side, policy_open(t)) if STUDY_EXEC else float('inf')
            if STUDY_EXEC and 'HCAP' in rules:
                zsz = min(zsz, room)
                live_ctr['capacity_clipped'] += int(zsz < wanted_size - 1e-9)
            if STUDY_EXEC:
                zsz = lots(zsz)
            legal = zsz >= 1 and (p if side == 0 else 100-p) * zsz / 100 >= 1 - 1e-9
            wc = worst_case(dict(side=side, pt=p, rem=zsz))"""
    s=replace(s,old,new,count=2)
    old="""            if LV['losscap'] is not None and realised0 + wc < -LV['losscap']:
                live_ctr['refused'] += 1; intent[lane] = None; continue"""
    new="""            reason = ('netcap' if STUDY_EXEC and zsz > room + 1e-9 else
                      'minimum' if STUDY_EXEC and not legal else
                      'loss_budget' if LV['losscap'] is not None and realised0 + wc < -LV['losscap'] else None)
            if reason is not None:
                live_ctr['refused'] += 1; live_ctr['refused_' + reason] += 1
                if not STUDY_EXEC:
                    intent[lane] = None; continue
                key = (p, wanted_size); old_pd = pend[lane]
                bo = min(4000.,old_pd['bo']*2) if old_pd and old_pd['key']==key else 250.
                pend[lane] = dict(key=key,t_first=old_pd['t_first'] if old_pd and old_pd['key']==key else t,bo=bo)
                heapq.heappush(ev,(t+bo,0,seq,'R',(lane,key)));seq += 1
                continue"""
    s=replace(s,old,new)
    s=replace(s,"                refill(t); key = (p, zsz)",
      "                refill(t); key = (p, wanted_size if STUDY_EXEC else zsz)")
    return s

def main():
    context()
    enabled=os.environ.get('PF_STUDY_EXEC','1')
    if enabled not in ('0','1'):raise ValueError('invalid execution mode')
    exec(compile(source(),str(HERE/'pf_step4_ladder_v3.py'),'exec'),
      dict(__name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'),
           lots=lots,headroom=headroom,size_state=size_state,MARGIN=MARGIN,STUDY_EXEC=enabled=='1'))

if __name__=='__main__':main()
