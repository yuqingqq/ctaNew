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

def cap_cannot_bind(result,realised,budget,halted):
    """Sufficient condition for exact path reuse; never relax a binding cap."""
    if halted or realised<=-budget:return False
    worst=result['min_worst']
    if not math.isfinite(worst):return not result['requests']
    return realised+worst>=-budget

def capacity_cannot_clip(result):
    c=result['live_ctr']
    return (c.get('capacity_checks',0)>0 and c.get('would_clip_capacity',1)==0) or not result['requests']

def replace(s,a,b,count=1):
    if s.count(a)!=count:raise ValueError('engine anchor mismatch: '+a[:60])
    return s.replace(a,b,1)

def source():
    p=HERE/'pf_step4_ladder_v3.py'
    if sha(p)!=BASE_SHA:raise ValueError('wrong frozen v3 source')
    s=p.read_text()
    s=replace(s,"            cancel_wait.setdefault(id(o_), (o_, ack))",
      "            cancel_wait.setdefault(id(o_), (o_, max(ack, t + STUDY_ACK_MS)))")
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
            if STUDY_EXEC:
                live_ctr['capacity_checks'] += 1
                live_ctr['would_clip_capacity'] += int(zsz > room)
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
    s=replace(s,"BASE = {'JOIN': [], 'MICRO': []}","""_raw_run_window = run_window
_cap_cache = {}; _cap_market = None; _cap_reused = 0; _capacity_reused = 0
def run_window(mid, F, kf, rules, policy, mode, size=SIZE, cap_state=None):
    global _cap_market, _cap_reused, _capacity_reused
    if mid != _cap_market:
        _cap_cache.clear(); _cap_market = mid
    budget = live_opts(rules)['losscap']
    key = (frozenset(r for r in rules if not r.startswith('XCAP')), policy, mode, size)
    old = _cap_cache.get(key)
    if CAP_CACHE and budget is None and 'HCAP' in rules:
        parent_key = (key[0] - {'HCAP'}, policy, mode, size)
        parent = _cap_cache.get(parent_key)
        if parent is not None and capacity_cannot_clip(parent):
            import copy
            result = copy.deepcopy(parent)
            if result['live_ctr'].get('capacity_checks',0):result['live_ctr']['capacity_clipped'] = 0
            _cap_cache[key] = result; _capacity_reused += 1
            return result
    if CAP_CACHE and budget is not None and old is not None and cap_cannot_bind(old,cap_state['realised'],budget,cap_state['halted']):
        import copy
        _cap_reused += 1
        return copy.deepcopy(old)
    result = _raw_run_window(mid,F,kf,rules,policy,mode,size,cap_state)
    if budget is None and result is not None:
        _cap_cache[key] = result
    return result

BASE = {'JOIN': [], 'MICRO': []}""")
    s=replace(s,"    _FLIP_CACHE.clear()", "    _FLIP_CACHE.clear()\n    if nw % 48 == 0: print('[study progress]', day, nw, 'windows; cap-equivalent reuses', _cap_reused, flush=True)")
    s += "\nio.write_json(f'{OUT}/study_execution.json',dict(cap_cache=CAP_CACHE,cap_reused=_cap_reused,capacity_reused=_capacity_reused,cancel_ack_ms=STUDY_ACK_MS,adapter_sha=STUDY_SHA))\n"
    return s

def main():
    context()
    enabled=os.environ.get('PF_STUDY_EXEC','1')
    if enabled not in ('0','1'):raise ValueError('invalid execution mode')
    ack=float(os.environ.get('PF_STUDY_CANCEL_ACK_MS','0'))
    if not math.isfinite(ack) or ack<0:raise ValueError('invalid cancel acknowledgement delay')
    exec(compile(source(),str(HERE/'pf_step4_ladder_v3.py'),'exec'),
      dict(__name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'),
           lots=lots,headroom=headroom,size_state=size_state,MARGIN=MARGIN,STUDY_EXEC=enabled=='1',
           cap_cannot_bind=cap_cannot_bind,capacity_cannot_clip=capacity_cannot_clip,
           CAP_CACHE=os.environ.get('PF_CAP_CACHE','0')=='1',STUDY_SHA=sha(__file__),STUDY_ACK_MS=ack))

if __name__=='__main__':main()
