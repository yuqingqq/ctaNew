"""Opt-in passive reduction variants over the previously checked sizing adapter."""
import math
import os
from . import pf_sizing_engine as parent
from .pf_passive_study import HERE, context, sha

PARENT_SHA='95378036e04f8a0e7778063498fe7d94ce6b70276be01d8d74d992a1a13d101a'


def keep_size(net,side,price,current,opens,max_remaining=None):
    if side not in (0,1) or not math.isfinite(net):raise ValueError('invalid known state')
    if current is None or price is None:return None
    if current['side']!=side or current['pt']!=price:return None
    if not ((side==0 and net<0) or (side==1 and net>0)):return None
    rem=current['size']-current.get('kq',0.)
    other=sum(max(0.,o['size']-o.get('kq',0.)) for o in {id(o):o for o in opens}.values()
              if o is not current and o['side']==side)
    room=max(0.,abs(net)-other)
    if 1e-9<rem<=room+1e-9 and (max_remaining is None or rem<=max_remaining+1e-9):
        return current['size']  # engine retains original order identity; remainder may be smaller
    return None


def urgent_price(net,side,price,bid,ask,rel):
    if side not in (0,1) or not all(math.isfinite(x) for x in [net,bid,ask,rel]):raise ValueError('invalid price state')
    if price is None or not 0<bid<ask<100:return price
    if not (180<=rel<240 and abs(net)>=10 and ((side==0 and net<0) or (side==1 and net>0))):return price
    p=price+(1 if side==0 else -1)
    return p if 1<=p<=99 and bid<p<ask else price


def adjust(net,side,price,size,rel,bid,ask,current,opens,rules):
    if price is None or not 60<=rel<240:return price,size
    if not ((side==0 and net<0) or (side==1 and net>0)):return price,size
    if 'RPRICE' in rules:price=urgent_price(net,side,price,bid,ask,rel)
    if 'RSMALL' in rules:size=min(size,5.)
    if 'RKEEP' in rules:
        kept=keep_size(net,side,price,current,opens,5. if 'RSMALL' in rules else None)
        if kept is not None:size=kept
    return price,size


def source():
    if sha(HERE/'pf_sizing_engine.py')!=PARENT_SHA:raise ValueError('wrong parent sizing adapter')
    s=parent.source()
    old="                zs[side] = sz_ = min(sz_, abs(knet))\n            if LV['capn'] is not None and not red_"
    new="""                zs[side] = sz_ = min(sz_, abs(knet))
                if REDUCE_RULES:
                    old_p_,old_z_=p,sz_
                    cur_=cur_order(side)
                    p,sz_=passive_adjust(knet,side,p,sz_,rel,b1,a1,cur_,policy_open(t),rules)
                    q[side]=p;zs[side]=sz_
                    if p!=old_p_:
                        live_ctr['reduce_price_shift_decisions']+=1
                    if sz_!=old_z_:
                        live_ctr['reduce_size_change_decisions']+=1
                    if (p,sz_)!=(old_p_,old_z_) and len(passive_examples)<10:
                        passive_examples.append(dict(mid=int(mid),rel=float(rel),side=side,known_net=float(knet),
                            bid=float(b1),ask=float(a1),old_price=float(old_p_),price=float(p),
                            old_size=float(old_z_),size=float(sz_),current_size=cur_['size'] if cur_ else None,
                            current_known_rem=(cur_['size']-cur_.get('kq',0.)) if cur_ else None))
            if LV['capn'] is not None and not red_"""
    s=parent.replace(s,old,new)
    s=parent.replace(s,"    SZ = parse_size(size); cap = SZ['hi']; LV = live_opts(rules); VSTRICT = 'GS' in rules",
        "    REDUCE_RULES = rules.intersection({'RKEEP','RSMALL','RPRICE'})\n    passive_examples=[]\n    SZ = parse_size(size); cap = SZ['hi']; LV = live_opts(rules); VSTRICT = 'GS' in rules")
    s=parent.replace(s,"    return dict(pnl=cash + net * y, net_end=net, n_fills=len(fills)",
        "    if REDUCE_RULES:\n        PASSIVE_AUDIT.extend(dict(x,variant='+'.join(sorted(REDUCE_RULES))) for x in passive_examples)\n    return dict(pnl=cash + net * y, net_end=net, n_fills=len(fills)")
    s+="\nio.write_json(f'{OUT}/passive_audit.json',dict(parent_sha=PASSIVE_PARENT_SHA,examples=PASSIVE_AUDIT))\n"
    return s


def main():
    context()
    ack=float(os.environ.get('PF_STUDY_CANCEL_ACK_MS','0'))
    if not math.isfinite(ack) or ack<0:raise ValueError('invalid delay')
    if os.environ.get('PF_STUDY_EXEC')!='1':raise ValueError('shared execution required')
    exec(compile(source(),str(HERE/'pf_step4_ladder_v3.py'),'exec'),dict(
        __name__='__main__',__package__=__package__,__file__=str(HERE/'pf_step4_ladder_v3.py'),
        lots=parent.lots,headroom=parent.headroom,size_state=parent.size_state,MARGIN=parent.MARGIN,
        cap_cannot_bind=parent.cap_cannot_bind,capacity_cannot_clip=parent.capacity_cannot_clip,
        CAP_CACHE=False,STUDY_EXEC=True,STUDY_ACK_MS=ack,STUDY_SHA=sha(__file__),
        passive_adjust=adjust,PASSIVE_AUDIT=[],PASSIVE_PARENT_SHA=PARENT_SHA))


if __name__=='__main__':main()
