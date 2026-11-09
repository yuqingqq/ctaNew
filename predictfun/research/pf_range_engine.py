"""Allow reversals inside one shared +/-10-share inventory envelope."""
import math
from . import pf_inflight_engine as prior
from .pf_range_study import HERE, context, sha

PARENT_SHA = '15cd4554677a4a2dfc3ad904db91b80ec9b6a7e9191968a43063e997abf460e2'
entry = prior.entry
parent = prior.parent


def plan(net, wants, currents, opens, mode):
    if mode != 'range10':
        return prior.plan(net, wants, currents, opens, mode)
    out=[]; new=[0.,0.]
    for lane,(price,size) in enumerate(wants):
        side=lane%2
        if price is None:
            out.append((price,size));continue
        if not math.isfinite(size) or size<0:
            raise ValueError('invalid desired quantity')
        cur=currents[lane];same=cur is not None and cur['pt']==price
        cap=min(size,max(0.,entry.room(net,side,opens,cur if same else None)-new[side]))
        if same and 1e-9<entry.remainder(cur)<=cap+1e-9:
            out.append((price,cur['size']));continue
        cap=min(size,max(0.,entry.room(net,side,opens)-new[side]))
        q=parent.lots(cap)
        if not entry.legal(price,side,q):
            out.append((None,0.));continue
        out.append((price,q));new[side]+=q
    return tuple(out)


def admission(net,side,size,opens,mode):
    return min(size,entry.room(net,side,opens)) if mode=='range10' else prior.admission(net,side,size,opens,mode)


def assert_admission(net,side,size,opens,mode):
    if mode!='range10':return prior.assert_admission(net,side,size,opens,mode)
    if not math.isfinite(size) or size<0 or size>entry.room(net,side,opens)+1e-7:
        raise ValueError('range submission violation')


def release_time(ack,now,fill_delay,mode):
    return prior.release_time(ack,now,fill_delay,'both_reserved' if mode=='range10' else mode)


def assert_path(net,mode):
    return prior.assert_path(net,'both_reserved' if mode=='range10' else mode)


def source():
    if sha(HERE/'pf_inflight_engine.py')!=PARENT_SHA:raise ValueError('wrong inflight adapter')
    s=prior.source()
    return parent.replace(s,"    ENTRY_KIND = 'both_reserved' if 'IBOTH' in rules else",
        "    ENTRY_KIND = 'range10' if 'IRANGE10' in rules else 'both_reserved' if 'IBOTH' in rules else")


def main():
    context()
    exec(compile(source(),str(HERE/'pf_step4_ladder_v3.py'),'exec'),dict(
        __name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'),
        lots=parent.lots,headroom=parent.headroom,size_state=parent.size_state,MARGIN=parent.MARGIN,
        cap_cannot_bind=parent.cap_cannot_bind,capacity_cannot_clip=parent.capacity_cannot_clip,
        CAP_CACHE=False,STUDY_EXEC=True,STUDY_ACK_MS=0.,STUDY_SHA=sha(__file__),
        entry_plan=plan,entry_admission=admission,entry_assert_admission=assert_admission,
        entry_adding=entry.adding,ENTRY_AUDIT=[],ENTRY_PARENT_SHA=PARENT_SHA,
        inflight_release=release_time,inflight_assert_path=assert_path))


if __name__=='__main__':main()
