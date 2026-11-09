"""Separate closing liquidity from inventory-adding prediction sizes."""
import collections
import math
from . import pf_late_engine as old
from .pf_exit_study import HERE,context,sha

PARENT_SHA='754da32d4b6e4a48c7b65a37ff29e92f2beee07661e8c33402718df2496b1420'
NEW=('exit_keep','exit_close25')
COUNTS={m:collections.Counter() for m in NEW}
parent=old.parent


def mapped(mode):
    return 'late_reserved' if mode in NEW else mode


def desired_size(net,side,p,size,current,opens,mode,rel):
    if mode not in NEW or not 60<=rel<240 or p is None or old.prior.entry.adding(net,side):
        return size
    if current is not None and current['pt']==p:
        room=old.prior.room(net,side,opens,'reduce_reserved',current)
        if 1e-9<old.prior.entry.remainder(current)<=room+1e-9:
            COUNTS[mode]['safe_keep_over_target_quote_decisions']+=int(old.prior.entry.remainder(current)>size+1e-9)
            return current['size']
    target=min(25.,abs(net)) if mode=='exit_close25' else size
    COUNTS[mode]['close25_up_quote_decisions']+=int(target>size+1e-9)
    return target


def plan(net,wants,currents,opens,mode,rel):
    if mode not in NEW:
        return old.plan(net,wants,currents,opens,mode,rel)
    if not old.active(mapped(mode),rel):
        return tuple(wants)
    out=[];new=[0.,0.]
    for lane,(p,size) in enumerate(wants):
        side=lane%2
        if p is None:
            out.append((p,size));continue
        if not math.isfinite(size) or size<0:
            raise ValueError('invalid desired size')
        if old.prior.entry.adding(net,side):
            out.append((None,0.));continue
        cur=currents[lane]
        same=cur is not None and cur['pt']==p
        room=max(0.,old.prior.room(net,side,opens,'reduce_reserved',cur if same else None)-new[side])
        if same and 1e-9<old.prior.entry.remainder(cur)<=room+1e-9:
            COUNTS[mode]['safe_keep_decisions']+=1
            COUNTS[mode]['kept_above_am_target_decisions']+=int(old.prior.entry.remainder(cur)>size+1e-9)
            out.append((p,cur['size']));continue
        available=max(0.,old.prior.room(net,side,opens,'reduce_reserved')-new[side])
        wanted=25. if mode=='exit_close25' else size
        q=parent.lots(min(wanted,available))
        if not old.prior.entry.legal(p,side,q):
            COUNTS[mode]['room_or_minimum_withheld_decisions']+=1
            out.append((None,0.));continue
        COUNTS[mode]['larger_than_am_target_decisions']+=int(q>size+1e-9)
        out.append((p,q));new[side]+=q
    return tuple(out)


def admission(net,side,size,opens,mode,rel):
    return old.admission(net,side,size,opens,mapped(mode),rel)


def assert_admission(net,side,size,opens,mode,rel):
    return old.assert_admission(net,side,size,opens,mapped(mode),rel)


def release(ack,now,fill_delay,mode):
    return old.release(ack,now,fill_delay,mapped(mode))


def visible(ack,now,ws,fill_delay,mode,legacy_ack=None):
    return old.visible(ack,now,ws,fill_delay,mapped(mode),legacy_ack)


def price(net,side,p,bid,ask,rel,mode):
    return old.price(net,side,p,bid,ask,rel,mapped(mode))


def source():
    if sha(HERE/'pf_late_engine.py')!=PARENT_SHA:
        raise ValueError('wrong late adapter')
    s=old.source()
    s=parent.replace(s,"    ENTRY_KIND = 'late_taper1' if 'LTAPER' in rules else",
        "    ENTRY_KIND = 'exit_close25' if 'ECLOSE25' in rules else 'exit_keep' if 'EKEEP' in rules else 'late_taper1' if 'LTAPER' in rules else")
    s=parent.replace(s,'                if ENTRY_KIND in LATE_MODES:\n                    old_p_=p',
        '''                if ENTRY_KIND in EXIT_MODES:
                    zs[side]=sz_=exit_desired_size(knet,side,p,sz_,cur_order(side),policy_open(t),ENTRY_KIND,rel)
                if ENTRY_KIND in LATE_MODES:
                    old_p_=p''')
    s+="\nio.write_json(f'{OUT}/exit_counts.json',EXIT_COUNTS)\n"
    return s


def main():
    context()
    exec(compile(source(),str(HERE/'pf_step4_ladder_v3.py'),'exec'),dict(
        __name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'),
        lots=parent.lots,headroom=parent.headroom,size_state=parent.size_state,MARGIN=parent.MARGIN,
        cap_cannot_bind=parent.cap_cannot_bind,capacity_cannot_clip=parent.capacity_cannot_clip,
        CAP_CACHE=False,STUDY_EXEC=True,STUDY_ACK_MS=0.,STUDY_SHA=sha(__file__),
        entry_plan=plan,entry_admission=admission,entry_assert_admission=assert_admission,
        entry_adding=old.prior.entry.adding,ENTRY_AUDIT=[],ENTRY_PARENT_SHA=old.prior.ENTRY_SHA,
        inflight_release=release,inflight_assert_path=old.prior.assert_path,
        late_visible=visible,late_price=price,late_intent_meta=old.intent_meta,
        late_order_meta=old.order_meta,late_fill_meta=old.fill_meta,
        LATE_MODES=old.MODES+NEW,LATE_FILLS=[],EXIT_COUNTS=COUNTS,EXIT_MODES=NEW,exit_desired_size=desired_size))


if __name__=='__main__':
    main()
