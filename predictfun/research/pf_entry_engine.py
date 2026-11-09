"""Adding-side experiments over the unchanged, pinned execution adapter."""
import math
from . import pf_sizing_engine as parent
from .pf_entry_study import HERE,context,sha

PARENT_SHA='95378036e04f8a0e7778063498fe7d94ce6b70276be01d8d74d992a1a13d101a'


def remainder(o):
    q=o['size']-o.get('kq',0.)
    if not math.isfinite(q) or q < -1e-7:raise ValueError('invalid known remainder')
    return max(0.,q)


def adding(net,side):
    if side not in (0,1) or not math.isfinite(net):raise ValueError('invalid known state')
    return (1 if side==0 else -1)*net>=0


def room(net,side,opens,exclude=None):
    adding(net,side)
    unique={id(o):o for o in opens}
    pending=sum(remainder(o) for o in unique.values() if o is not exclude and o['side']==side)
    return max(0.,10-(net if side==0 else -net)-pending)


def legal(price,side,q):
    return q>=1 and (price if side==0 else 100-price)*q/100>=1-1e-9


def plan(net,wants,currents,opens,mode):
    if mode not in ('uniform10','reserved10'):raise ValueError('invalid entry mode')
    out=[];new=[0.,0.]
    for lane,(price,size) in enumerate(wants):
        side=lane%2
        if price is None or not adding(net,side):out.append((price,size));continue
        if not math.isfinite(size) or size<0:raise ValueError('invalid desired size')
        cur=currents[lane]
        same=cur is not None and cur['pt']==price
        cap=min(size,10.) if mode=='uniform10' else min(size,max(0.,room(net,side,opens,cur if same else None)-new[side]))
        if same and 1e-9<remainder(cur)<=cap+1e-9:
            out.append((price,cur['size']))  # retained order needs no new minimum-notional check
            continue
        cap=min(size,10.) if mode=='uniform10' else min(size,max(0.,room(net,side,opens)-new[side]))
        q=parent.lots(cap)
        if not legal(price,side,q):out.append((None,0.));continue
        out.append((price,q));new[side]+=q
    return tuple(out)


def admission(net,side,size,opens,mode):
    if not adding(net,side):return size
    return min(size,10. if mode=='uniform10' else room(net,side,opens))


def assert_admission(net,side,size,opens,mode):
    if not adding(net,side):return
    cap=10. if mode=='uniform10' else room(net,side,opens)
    if size>cap+1e-7:raise ValueError('entry budget submission violation')


def source():
    if sha(HERE/'pf_sizing_engine.py')!=PARENT_SHA:raise ValueError('wrong parent adapter')
    s=parent.source()
    s=parent.replace(s,"    SZ = parse_size(size); cap = SZ['hi']; LV = live_opts(rules); VSTRICT = 'GS' in rules",
        "    ENTRY_KIND = 'reserved10' if 'ERES10' in rules else 'uniform10' if 'EUN10' in rules else None\n    if ENTRY_KIND and not SPLIT: raise ValueError('entry policy requires split execution')\n    entry_examples=[]\n    SZ = parse_size(size); cap = SZ['hi']; LV = live_opts(rules); VSTRICT = 'GS' in rules")
    s=parent.replace(s,'        return tuple(out_)',"""        if ENTRY_KIND:
            entry_out_=entry_plan(knet,out_,[cur_order(L_) for L_ in LANES],policy_open(t),ENTRY_KIND)
            live_ctr['entry_changed_decisions']+=int(tuple(out_)!=entry_out_)
            live_ctr['entry_withheld_lanes']+=sum(p_ is not None and np_ is None for (p_,_), (np_,_) in zip(out_,entry_out_))
            if tuple(out_)!=entry_out_ and len(entry_examples)<5:
                entry_examples.append(dict(mid=int(mid),rel=float(rel),known_net=float(knet),before=out_,after=entry_out_))
            return entry_out_
        return tuple(out_)""")
    s=parent.replace(s,"            if STUDY_EXEC:\n                zsz = lots(zsz)","""            if ENTRY_KIND:
                zsz=entry_admission(knet,side,zsz,policy_open(t),ENTRY_KIND)
            if STUDY_EXEC:
                zsz = lots(zsz)""")
    s=parent.replace(s,"            requests.append((t, side, 'place', lane))", """            if ENTRY_KIND:
                entry_assert_admission(knet,side,zsz,policy_open(t),ENTRY_KIND)
                live_ctr['entry_adding_submissions']+=int(entry_adding(knet,side))
            requests.append((t, side, 'place', lane))""")
    s=parent.replace(s,'    return dict(pnl=cash + net * y, net_end=net, n_fills=len(fills)',
        "    if ENTRY_KIND: ENTRY_AUDIT.extend(dict(x,variant=ENTRY_KIND) for x in entry_examples)\n    return dict(pnl=cash + net * y, net_end=net, n_fills=len(fills)")
    s+="\nio.write_json(f'{OUT}/entry_audit.json',dict(parent_sha=ENTRY_PARENT_SHA,examples=ENTRY_AUDIT))\n"
    return s


def main():
    context()
    exec(compile(source(),str(HERE/'pf_step4_ladder_v3.py'),'exec'),dict(
        __name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'),
        lots=parent.lots,headroom=parent.headroom,size_state=parent.size_state,MARGIN=parent.MARGIN,
        cap_cannot_bind=parent.cap_cannot_bind,capacity_cannot_clip=parent.capacity_cannot_clip,
        CAP_CACHE=False,STUDY_EXEC=True,STUDY_ACK_MS=0.,STUDY_SHA=sha(__file__),
        entry_plan=plan,entry_admission=admission,entry_assert_admission=assert_admission,entry_adding=adding,
        ENTRY_AUDIT=[],ENTRY_PARENT_SHA=PARENT_SHA))


if __name__=='__main__':main()
