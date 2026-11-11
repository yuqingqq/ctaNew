"""Generate an isolated opt-in engine in memory; frozen v3 is never edited."""
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from . import pf_order_policy_spec as S
from .pf_order_policy_inputs import SafeReader
from .pf_order_policy_hooks import Manager


def replace(s,a,b):
    if s.count(a)!=1:raise ValueError('engine anchor identity: '+a[:90])
    return s.replace(a,b)


def source(queue='M'):
    path=S.HERE/'pf_step4_ladder_v3.py';text=path.read_text()
    expected=json.loads((S.HERE.parent/'freeze/pm_ladder_v3/params.json').read_text())['engine']['sha256']
    if S.sha(path)!=expected:raise ValueError('frozen engine identity')
    start=text.index('OUT = (f');end=text.index('os.makedirs(OUT, exist_ok=True)',start)
    text=text[:start]+'OUT = _DEST\n'+text[end:]
    text=replace(text,'    heapq.heapify(ev)\n', '''    heapq.heapify(ev)
    own = _STUDY.window(mid,ws,rules,F,kf,trades,list(ev),
        lambda tt:guard_state(tt,use1,use2,G1TH,G2KEY),
        lambda tt,ss:model_value(mid,tt,PREDVB if ss==0 else PREDVA),
        lambda tt:minute_line(PMK,(tt-ws)/1000))
    for tt in own.wakeups:
        heapq.heappush(ev,(tt,2,seq,'D',None));seq+=1
''')
    anchor="            pmf_ = [vv_[0] is not None and vv_[0] <= c_, vv_[1] is not None and vv_[1] <= c_]\n"
    text=replace(text,anchor,anchor+'''            for ss in (0,1):
                fl0=[r for r in inflight[ss] if not r['cancelled']]
                current=fl0[-1] if fl0 else known[ss]
                if own.seed_flag(t,ss,current,vv_[ss],c_):pmf_[ss]=False
''')
    text=replace(text,'            zs[side] = sz_ = side_size(t, rel, side, f, p)\n',
        '            zs[side] = sz_ = own.seed_size(t,side,side_size(t, rel, side, f, p))\n')
    anchor='            cur = fl[-1] if fl else pv(lane)\n'
    text=replace(text,anchor,anchor+'''            refs=[o for o in known.values() if o is not None]+[r for fl0 in inflight.values() for r in fl0 if not r['cancelled']]+[o for o,ack in cancel_wait.values() if ack>t]
            refs=list({id(o):o for o in refs}.values())
            p,zsz,hold=own.action(t,lane,p,zsz,cur,bool(fl),knet,refs)
            if (p,zsz)!=(pl[1],pl[2]):intent[lane]=(p,zsz)
            if hold:
                for tt in np.arange(t+50,min(t+1000,ws+240000)+1,50.):
                    heapq.heappush(ev,(tt,2,seq,'D',None));seq+=1
''')
    anchor='                reserve_cancel(cur, t_rm)\n'
    text=replace(text,anchor,anchor+'''                if own.cancelling(cur,t_rm):
                    heapq.heappush(ev,(t_rm,0,seq,'Z',cur));seq+=1
''')
    text=replace(text,'            if p is None:\n                continue\n            wc = worst_case(dict(side=side, pt=p, rem=zsz))\n',
        "            if p is None:\n                continue\n            if not own.allow_new(t,side,knet):intent[lane]=None;continue\n            wc = worst_case(dict(side=side, pt=p, rem=zsz))\n")
    anchor='            rec = dict(side=side, lane=lane, pt=p, size=zsz, t_send=t, t_act=t + L_PL, cancelled=False)\n'
    text=replace(text,anchor,anchor+'            own.sent(rec,knet)\n')
    for anchor in ("                live_ctr['postonly_refused'] += 1\n", "                    live_ctr['postonly_refused_own'] += 1\n                    cancel_wait.pop(id(rec), None)\n"):
        indent='                ' if anchor.startswith('                live_') else '                    '
        text=replace(text,anchor,anchor+indent+'own.rejected(rec)\n')
    anchor="            o['ahead0'] = o['ahead']\n"
    text=replace(text,anchor,anchor+'''            own.activated(o,rec)
            if o.get('_seed'):
                heapq.heappush(ev,(t+1000,2,seq,'D',None));seq+=1
''')
    anchor="        elif kind == 'K':                                   # v2: our fill becomes known\n"
    text=replace(text,anchor,'''        elif kind == 'Z':
            others=[o for zs0 in zombies.values() for o in zs0]+[o for o in orders.values() if o is not None]
            pending=[r for fl0 in inflight.values() for r in fl0]
            own.snapshot(t,pl,others,pending)
'''+anchor)
    anchor="                    fills.append((t, side, o['pt'], q, 'through' if through else 'at', timed, o['size'], is_z, o.get('lane', side), o.get('ahead0', np.nan), t - o['t0']))\n"
    text=replace(text,anchor,anchor+'                    own.fill(o,t,q)\n')
    anchor="    mids = np.array([((r['a1_p'] + r['b1_p']) / 2.0) if (r['b1_p'] >= 0 and r['a1_p'] > r['b1_p']) else np.nan for r in F])\n"
    text=replace(text,anchor,'    own.finish(fills,cash+net*y,net,y)\n'+anchor)
    if queue=='L':
        anchor="                        if d is not None and d < o['ahead']:\n                            o['ahead'] = d\n"
        text=replace(text,anchor,'                        if False: pass  # declared L sensitivity: no displayed-size cap\n')
    return text+'\n_STUDY.close()\n'


GROUPS={'smoke':['reference'],'build':['reference','calm_seed'],
        'eval':['hold_rule','continuation','calm_filter','calm_seed_continuation'],
        'stress':['reference','continuation','calm_seed','calm_seed_continuation']}


def folder(day,group):return S.OUT/'runs'/group/day


def run(day,group):
    S.check()
    if day not in S.DAYS or group not in GROUPS:raise ValueError('undeclared run')
    if group in ('eval','stress') and day not in S.TEST:raise ValueError('not a test day')
    timing='stress' if group=='stress' else 'X';c=S.TIMINGS[timing];dest=folder(day,group)
    if dest.exists():raise ValueError('refuse overwrite')
    for k in list(os.environ):
        if k.startswith('PF_'):del os.environ[k]
    os.environ.update(PF_FORWARD='20261104',PF_RL_DDEC='2',PF_RL_LPLACE=str(c['place']),PF_RL_LCANCEL=str(c['cancel']),
        PF_RL_LPENDCANCEL='95',PF_RL_LFILL=str(c['notice']),PF_TIMING='e2',PF_TXTIME='ws',PF_TXKEY='tk',
        PF_TXSHIFT=str(c['shift']),PF_QACT='max',PF_PREDS='preds_t60k',PF_ARMS=';'.join(S.arm(m) for m in GROUPS[group]),
        PF_MAX_WINDOWS='8' if group=='smoke' else '0')
    model=None
    if group in ('eval','stress'):
        import lightgbm as lgb
        model=lgb.Booster(model_file=str(S.OUT/'models'/f'{day}_queue.txt'))
    manager=Manager(day,timing,dest,'build' if group=='build' else 'eval',model)
    code=source(c['queue']);sys.argv=[str(S.HERE/'pf_step4_ladder_v3.py'),day,'tokyoreal','custom']
    original=pd.read_parquet;pd.read_parquet=SafeReader(day,original)
    try:exec(compile(code,str(S.HERE/'pf_step4_ladder_v3.py'),'exec'),dict(__name__='__main__',__package__=__package__,
        __file__=str(S.HERE/'pf_step4_ladder_v3.py'),_DEST=str(dest),_STUDY=manager))
    finally:pd.read_parquet=original
    from .pf_order_policy_verify import check_run
    check_run(day,group)
    S.write(dest/'source.json',dict(generated_sha=hashlib.sha256(code.encode()).hexdigest(),source_sha=S.sha(__file__),timing=c,day=day,group=group))
    print('VERIFIED_RUN',day,group,flush=True)


if __name__=='__main__':run(*sys.argv[1:])
