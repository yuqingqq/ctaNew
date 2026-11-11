"""Opt-in generated engine, full portfolio execution and protected input boundaries."""
import hashlib
import os
import sys
import pandas as pd
from . import pf_default_pull_spec as S
from .pf_order_policy_engine import source as original_source,replace
from .pf_order_policy_inputs import SafeReader
from .pf_default_pull_hooks import Manager

def source(queue='M'):
    s=original_source(queue)
    s=replace(s,'        return tuple(out_)\n','        return own.targets(t,out_,known,inflight,policy_open(t),knet)\n')
    s=replace(s,'            want = desired(t)\n', '''            want = desired(t)
            for at,target in own.probes:
                heapq.heappush(ev,(at,0,seq,'Y',target));seq+=1
            own.probes.clear()
''')
    s=replace(s,"        elif kind == 'Z':\n", "        elif kind in ('Y','Z'):\n")
    return s

def run(day,group):
    S.check()
    if day not in S.DAYS or group not in S.GROUPS:raise ValueError('undeclared run')
    if group in ('eval','stress') and day not in S.TEST:raise ValueError('not a test day')
    timing='stress' if group=='stress' else 'X';c=S.TIMINGS[timing];dest=S.folder(day,group)
    if dest.exists():raise ValueError('refuse overwrite')
    for k in list(os.environ):
        if k.startswith('PF_'):del os.environ[k]
    os.environ.update(PF_FORWARD='20261104',PF_RL_DDEC='2',PF_RL_LPLACE=str(c['place']),PF_RL_LCANCEL=str(c['cancel']),
        PF_RL_LPENDCANCEL='95',PF_RL_LFILL=str(c['notice']),PF_TIMING='e2',PF_TXTIME='ws',PF_TXKEY='tk',
        PF_TXSHIFT=str(c['shift']),PF_QACT='max',PF_PREDS='preds_t60k',PF_ARMS=';'.join(S.arm(m) for m in S.GROUPS[group]),
        PF_MAX_WINDOWS='8' if group=='smoke' else '0')
    models={}
    if group in ('eval','stress'):
        import lightgbm as lgb
        for k in ('market','queue'):models[k]=lgb.Booster(model_file=str(S.OUT/'models'/f'{day}_{k}.txt'))
    manager=Manager(day,timing,dest,'build' if group in ('smoke','build') else 'eval',models=models)
    code=source(c['queue']);sys.argv=[str(S.HERE/'pf_step4_ladder_v3.py'),day,'tokyoreal','custom']
    original=pd.read_parquet;pd.read_parquet=SafeReader(day,original)
    try:exec(compile(code,str(S.HERE/'pf_step4_ladder_v3.py'),'exec'),dict(__name__='__main__',__package__=__package__,
        __file__=str(S.HERE/'pf_step4_ladder_v3.py'),_DEST=str(dest),_STUDY=manager))
    finally:pd.read_parquet=original
    from .pf_default_pull_verify import check_run
    check_run(day,group)
    S.write(dest/'source.json',dict(generated_sha=hashlib.sha256(code.encode()).hexdigest(),timing=c,day=day,group=group))
    print('VERIFIED_RUN',day,group,flush=True)

if __name__=='__main__':run(*sys.argv[1:])
