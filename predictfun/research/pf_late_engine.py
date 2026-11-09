"""Late-only reservation and bounded passive concession; pinned engine adapter."""
import math
from . import pf_inflight_engine as prior
from .pf_late_study import HERE, context, sha

PARENT_SHA = '15cd4554677a4a2dfc3ad904db91b80ec9b6a7e9191968a43063e997abf460e2'
parent = prior.parent
MODES = ('late_reserved', 'late_taper1')


def active(mode, rel):
    if mode not in MODES or not math.isfinite(rel):
        raise ValueError('invalid late policy/time')
    return rel >= 60


def target(rel):
    if not math.isfinite(rel):
        raise ValueError('invalid target time')
    return max(0., min(10., 10*(240-rel)/180))


def price(net, side, p, bid, ask, rel, mode):
    enabled = active(mode,rel)
    if side not in (0,1) or not all(math.isfinite(x) for x in [net,bid,ask]):
        raise ValueError('invalid quote state')
    if p is None or not enabled or rel >= 240 or mode != 'late_taper1':
        return p
    if not 0 < bid < ask < 100 or (1 if side == 0 else -1)*net >= 0 or abs(net) <= target(rel):
        return p
    proposed = p + (1 if side == 0 else -1)
    return proposed if 1 <= proposed <= 99 and bid < proposed < ask else p


def plan(net, wants, currents, opens, mode, rel):
    return prior.plan(net,wants,currents,opens,'reduce_reserved') if active(mode,rel) else tuple(wants)


def admission(net, side, size, opens, mode, rel):
    if not active(mode,rel):
        return size
    return 0. if rel>=240 or prior.entry.adding(net,side) else prior.admission(net,side,size,opens,'reduce_reserved')


def assert_admission(net, side, size, opens, mode, rel):
    if active(mode,rel):
        if (rel>=240 or prior.entry.adding(net,side)) and size>0:
            raise ValueError('late submission would increase inventory')
        prior.assert_admission(net,side,size,opens,'reduce_reserved')


def release(ack, now, fill_delay, mode):
    return prior.release_time(ack,now,fill_delay,'reduce_reserved' if mode in MODES else mode)


def visible(ack, now, ws, fill_delay, mode, legacy_ack=None):
    # Stored expiry includes grace even before intervention. Before60, expose exactly original state.
    if mode not in MODES or now-ws >= 60000:
        return ack > now
    if legacy_ack is None or not math.isfinite(legacy_ack):
        raise ValueError('missing original cancellation timestamp')
    return legacy_ack > now


def intent_meta(mode,t,ws,lane,want,net,quotes):
    p=want[0]
    if p is None:
        return None
    if mode is None or t-ws<60000:
        return dict(t=t,price=p,base_price=p,decision_net=net)
    meta = quotes.get(lane)
    if meta is None or meta['t'] != t or meta['price'] != p:
        raise ValueError('missing exact quote origin')
    return dict(meta,decision_net=net)


def order_meta(mode, t, ws, lane, p, net, quotes):
    # A rate-limited retry sends the retained intent later. Keep that intent's own price origin.
    meta=quotes.get(lane)
    if meta is None or meta['t']>t or meta['price']!=p:
        raise ValueError('missing exact intent origin')
    return dict(send_rel=(t-ws)/1000,known_net=net,base_price=meta['base_price'],
                decision_rel=(meta['t']-ws)/1000,decision_net=meta['decision_net'],
                concession_ticks=(1 if lane%2 == 0 else -1)*(p-meta['base_price']))


def fill_meta(mode, mid, index, t, ws, side, p, q, net, order):
    meta = order.get('late_meta')
    if meta is None or meta['concession_ticks'] not in (0.,1.):
        raise ValueError('missing/invalid originating order metadata')
    sg = 1 if side == 0 else -1
    before = net-sg*q
    return dict(variant=mode or 'baseline',mid=int(mid),index=index,t=(t-ws)/1000,
                side=side,pt=p,q=q,net_before=before,net_after=net,
                reducing_q=min(abs(before),q) if before*sg < 0 else 0.,
                concession=q*meta['concession_ticks']/100,**meta)


def source():
    if sha(HERE/'pf_inflight_engine.py') != PARENT_SHA:
        raise ValueError('wrong inflight adapter')
    s = prior.source()
    s = parent.replace(s,"    ENTRY_KIND = 'both_reserved' if 'IBOTH' in rules else",
                       "    ENTRY_KIND = 'late_taper1' if 'LTAPER' in rules else 'late_reserved' if 'LRES' in rules else 'both_reserved' if 'IBOTH' in rules else")
    s = parent.replace(s,'    entry_examples=[]','    entry_examples=[]\n    late_quotes={}\n    late_intents={}')
    s = parent.replace(s,'                    intent[side] = want[side]',
                       '                    intent[side] = want[side]\n                    late_intents[side]=late_intent_meta(ENTRY_KIND,t,ws,side,want[side],knet,late_quotes)')
    s = parent.replace(s,'ENTRY_KIND)\n            live_ctr[\'entry_changed_decisions\']',
                       'ENTRY_KIND,rel)\n            live_ctr[\'entry_changed_decisions\']')
    s = parent.replace(s,'zsz=entry_admission(knet,side,zsz,policy_open(t),ENTRY_KIND)',
                       'zsz=entry_admission(knet,side,zsz,policy_open(t),ENTRY_KIND,(t-ws)/1000)')
    s = parent.replace(s,'entry_assert_admission(knet,side,zsz,policy_open(t),ENTRY_KIND)',
                       'entry_assert_admission(knet,side,zsz,policy_open(t),ENTRY_KIND,(t-ws)/1000)')
    s = parent.replace(s,'            refs += [o_ for o_, _ in cancel_wait.values()]',
                       "            refs += [o_ for o_, ack in cancel_wait.values() if late_visible(ack,now,ws,L_FILL,ENTRY_KIND,o_.get('late_legacy_ack'))]")
    s = parent.replace(s,'            cancel_wait.setdefault(id(o_), (o_, inflight_release(ack, t + STUDY_ACK_MS, L_FILL, ENTRY_KIND)))',
                       "            o_.setdefault('late_legacy_ack',max(ack,t+STUDY_ACK_MS))\n            cancel_wait.setdefault(id(o_), (o_, inflight_release(ack, t + STUDY_ACK_MS, L_FILL, ENTRY_KIND)))")
    old = '                zs[side] = sz_ = min(sz_, abs(knet))\n            if LV[\'capn\'] is not None and not red_'
    new = '''                zs[side] = sz_ = min(sz_, abs(knet))
                if ENTRY_KIND in LATE_MODES:
                    old_p_=p
                    p=late_price(knet,side,p,b1,a1,rel,ENTRY_KIND)
                    q[side]=p
                    late_quotes[side]=dict(t=t,price=p,base_price=old_p_)
                    live_ctr['late_concession_decisions']+=int(p!=old_p_)
            if LV['capn'] is not None and not red_'''
    s = parent.replace(s,old,new)
    s = parent.replace(s,"            inflight[lane].append(rec); heapq.heappush(ev, (t + L_PL, 0, seq, 'P', rec)); seq += 1",
                       "            rec['late_meta']=late_order_meta(ENTRY_KIND,t,ws,lane,p,knet,late_intents)\n            inflight[lane].append(rec); heapq.heappush(ev, (t + L_PL, 0, seq, 'P', rec)); seq += 1")
    s = parent.replace(s,'            if id(rec) in cancel_wait:',
                       "            o['late_meta']=rec['late_meta']\n            if 'late_legacy_ack' in rec: o['late_legacy_ack']=rec['late_legacy_ack']\n            if id(rec) in cancel_wait:")
    s = parent.replace(s,"                    live_ctr['zombie_fills'] += int(is_z)",
                       "                    LATE_FILLS.append(late_fill_meta(ENTRY_KIND,mid,len(fills)-1,t,ws,side,o['pt'],q,net,o))\n                    live_ctr['zombie_fills'] += int(is_z)")
    s += "\npd.DataFrame(LATE_FILLS).to_parquet(f'{OUT}/late_fills.parquet',index=False)\n"
    return s


def main():
    context()
    exec(compile(source(),str(HERE/'pf_step4_ladder_v3.py'),'exec'),dict(
        __name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'),
        lots=parent.lots,headroom=parent.headroom,size_state=parent.size_state,MARGIN=parent.MARGIN,
        cap_cannot_bind=parent.cap_cannot_bind,capacity_cannot_clip=parent.capacity_cannot_clip,
        CAP_CACHE=False,STUDY_EXEC=True,STUDY_ACK_MS=0.,STUDY_SHA=sha(__file__),
        entry_plan=plan,entry_admission=admission,entry_assert_admission=assert_admission,
        entry_adding=prior.entry.adding,ENTRY_AUDIT=[],ENTRY_PARENT_SHA=prior.ENTRY_SHA,
        inflight_release=release,inflight_assert_path=prior.assert_path,
        late_visible=visible,late_price=price,late_intent_meta=intent_meta,late_order_meta=order_meta,late_fill_meta=fill_meta,
        LATE_MODES=MODES,LATE_FILLS=[]))


if __name__ == '__main__':
    main()
