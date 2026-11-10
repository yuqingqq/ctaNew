"""Opt-in generated research engine; frozen files and live code stay untouched."""
from . import pf_inventory_cost_policy as p
from . import pf_late_engine as old
from .pf_inventory_cost_study import HERE, DATA, sha


def source():
    s = old.source()
    replace = old.parent.replace
    s = replace(s, "    ENTRY_KIND = 'late_taper1' if 'LTAPER' in rules else",
        "    ENTRY_KIND = 'marginal_entry' if 'ERISK' in rules else 'budget_exit' if 'ECOST' in rules else 'soft_exit' if 'ESOFT' in rules else 'late_taper1' if 'LTAPER' in rules else")
    s = replace(s, '    late_quotes={}', "    study_state={'committed_concession':0.}\n    diagnostic=Episodes()\n    late_quotes={}")
    s = replace(s, "    for t in sorted({ws + stop * 1000.0, ws + 240_000, ws + 300_000}):", """    if ENTRY_KIND in ('soft_exit','budget_exit'):
        for P_ in (PREDVB,PREDVA):
            v_=P_.get(mid)
            if v_ is not None:
                shifted_={mid:(v_[0],v_[1]+SOFT_DELTA)}
                for tt_ in line_flips(mid,ws,ws+60000,ws+240000,PMK,shifted_):
                    ev.append((tt_,2,seq,'D',None));seq+=1
    for t in sorted({ws + stop * 1000.0, ws + 240_000, ws + 300_000}):""")
    # ENTRY_KIND and PMK are assigned before event scheduling in the generated source.
    s = replace(s, '    def desired(t):', '    def base_desired(t):')
    s = replace(s, '            ba = ba or pmf_[0]; sa = sa or pmf_[1]', """            pmf_=[soft_flag(ENTRY_KIND,rel,knet,s_,vv_[s_],c_,pmf_[s_]) for s_ in (0,1)]
            ba = ba or pmf_[0]; sa = sa or pmf_[1]""")
    s = replace(s, 'p=late_price(knet,side,p,b1,a1,rel,ENTRY_KIND)',
        'p=cost_price(knet,side,p,b1,a1,rel,ENTRY_KIND,study_state,cur_order(side))')
    s = replace(s, 'entry_plan(knet,out_,[cur_order(L_) for L_ in LANES],policy_open(t),ENTRY_KIND,rel)',
        'cost_plan(knet,out_,[cur_order(L_) for L_ in LANES],risk_open(t),ENTRY_KIND,rel,vv_,minute_line(PMK,rel))')
    s = replace(s, 'zsz=entry_admission(knet,side,zsz,policy_open(t),ENTRY_KIND,(t-ws)/1000)',
        'zsz=cost_admission(knet,side,zsz,risk_open(t),ENTRY_KIND,(t-ws)/1000,model_value(mid,t,PREDVB if side==0 else PREDVA),minute_line(PMK,(t-ws)/1000),p,late_intents.get(lane),study_state)')
    s = replace(s, "            rec['late_meta']=late_order_meta(ENTRY_KIND,t,ws,lane,p,knet,late_intents)",
        "            cost_sent(ENTRY_KIND,side,p,zsz,late_intents.get(lane),study_state)\n            rec['late_meta']=late_order_meta(ENTRY_KIND,t,ws,lane,p,knet,late_intents)")
    anchor = '    def worst_case(extra=None):'
    s = replace(s, anchor, """    def risk_open(t):
        refs=policy_open(t)
        if ENTRY_KIND=='marginal_entry':
            refs += [o_ for o_,until_ in cancel_wait.values() if until_>t]
        return list({id(o_):o_ for o_ in refs}.values())

    def desired(t):
        out=base_desired(t)
        rel=(t-ws)/1000
        if ENTRY_KIND is None and rel>=60:
            f=known_frame(t)
            vals=[model_value(mid,t,PREDVB),model_value(mid,t,PREDVA)]
            sz=[side_size(t,rel,s_,f,0) for s_ in (0,1)] if f is not None else [0.,0.]
            btc=guard_state(t,use1,use2,G1TH,G2KEY) if (use1 or use2) else (False,False)
            diagnostic.observe(int(mid),rel,feasibility(rel,knet,f,vals,minute_line(PMK,rel),sz,btc,policy_open(t),out))
        return out

""" + anchor)
    s = replace(s, '    if ENTRY_KIND: ENTRY_AUDIT.extend',
        "    diagnostic.finish()\n    AUDIT_ROWS.extend(diagnostic.rows)\n    BUDGET_ROWS.append(dict(mid=int(mid),variant=ENTRY_KIND or 'baseline',**study_state))\n    if ENTRY_KIND: ENTRY_AUDIT.extend")
    s += "\npd.DataFrame(AUDIT_ROWS).to_parquet(f'{OUT}/exit_episodes.parquet',index=False)\npd.DataFrame(BUDGET_ROWS).to_parquet(f'{OUT}/cost_budgets.parquet',index=False)\n"
    return s


def main():
    from . import pf_io
    pf_io.OUT = str(DATA)
    parent = old.parent
    exec(compile(source(), str(HERE/'pf_step4_ladder_v3.py'), 'exec'), dict(
        __name__='__main__', __package__=__package__, __file__=str(HERE/'pf_step4_ladder_v3.py'),
        lots=parent.lots, headroom=parent.headroom, size_state=parent.size_state, MARGIN=parent.MARGIN,
        cap_cannot_bind=parent.cap_cannot_bind, capacity_cannot_clip=parent.capacity_cannot_clip,
        CAP_CACHE=False, STUDY_EXEC=True, STUDY_ACK_MS=0., STUDY_SHA=sha(__file__),
        entry_assert_admission=lambda net,side,size,opens,mode,rel:old.assert_admission(net,side,size,opens,p.mapped(mode),rel),
        entry_adding=old.prior.entry.adding, ENTRY_AUDIT=[], ENTRY_PARENT_SHA=old.prior.ENTRY_SHA,
        inflight_release=lambda ack,now,delay,mode:old.release(ack,now,delay,p.mapped(mode)),
        inflight_assert_path=old.prior.assert_path,
        late_visible=lambda ack,now,ws,delay,mode,legacy:old.visible(ack,now,ws,delay,p.mapped(mode),legacy),
        late_intent_meta=old.intent_meta,late_order_meta=old.order_meta,late_fill_meta=old.fill_meta,
        LATE_MODES=old.MODES+p.NEW,LATE_FILLS=[],
        SOFT_DELTA=p.SOFT_DELTA,soft_flag=p.soft_flag,cost_price=p.price,cost_plan=p.plan,
        cost_admission=p.admission,cost_sent=p.sent,feasibility=p.feasibility,Episodes=p.Episodes,
        AUDIT_ROWS=[],BUDGET_ROWS=[]))


if __name__=='__main__':
    main()
