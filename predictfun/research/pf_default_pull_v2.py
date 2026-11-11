"""Supersede failed v1 smoke: reserve cancelled exposure through late-fill notice."""
import sys
from pathlib import Path

ROOT=Path('/home/yuqing/ctaNew')
OUT=ROOT/'data/predict_fun/research/default_pull_20261111_v2'
HERE=Path(__file__).resolve().parent

def configure():
    from . import pf_default_pull_spec as S
    from . import pf_default_pull_hooks as H
    S.OUT=OUT;S.REVIEW=HERE.parent/'reviews/default_pull_20261111_v2'
    original=H.Hook
    class ReservedHook(original):
        def __init__(self,*args,**kw):
            super().__init__(*args,**kw);self.retired={}
        def cancelling(self,cur,t_rm):
            if self.mode!='reference':self.retired[cur['_gid']]=t_rm+self.c['notice']
            return super().cancelling(cur,t_rm)
        def targets(self,t,original,known,inflight,refs,net):
            extra=[]
            for gid,until in list(self.retired.items()):
                if t>=until or self.records[gid]['status']=='postonly_refused':
                    self.retired.pop(gid);continue
                # Pending create is replaced by its accepted object; kq follows the actual notice.
                extra.append(self.objects[gid])
            refs=list({id(o):o for o in list(refs)+extra}.values())
            return super().targets(t,original,known,inflight,refs,net)
    H.Hook=ReservedHook
    return S

def declaration():
    import datetime as dt
    import json
    import subprocess
    from . import pf_default_pull_spec as S
    old=S.OUT/'declaration.json';d=json.loads(old.read_text());old_sha=S.sha(old)
    if OUT.exists():raise ValueError('v2 output exists')
    d.update(version=2,as_of=dt.datetime.now(dt.timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
        supersedes=dict(declaration_sha=old_sha,path=str(old),status='FAILED_SMOKE_INVENTORY_INVARIANT'),
        correction='v1 custom +/-10 reservation design released cancellation capacity at effectiveness before the last possible fill notification. Retain cancelled generations through effective cancel + configured fill-notice delay, use notified kq only, transfer pending identity on activation. Reference stays passive/exact. No model, threshold, input or evaluation change; new output folder preserves all failed v1 artifacts.',
        additional_check='Every saved feature decision compared directly with its actual cached timestamp/float32 features/250ms history after each replay. Positive identity and known-bad changed-value/clock controls. Full inventory invariant and exact reference parity must pass before fitting.')
    for p in (HERE/'pf_default_pull_v2.py',HERE/'run_default_pull_v2.sh'):
        d['source_sha'][str(p.relative_to(HERE.parents[1]))]=S.sha(p)
    S.write(OUT/'declaration.json',d);S.write(HERE.parent/'reviews/default_pull_20261111_v2/declaration.json',d)
    print('V2_DECLARED',flush=True)

def controls():
    import unittest
    from .tests import test_default_pull as T
    h,o=T.hook();known={i:None for i in range(4)};fl={i:[] for i in known}
    h.cancelling(o,6522.)
    if h.targets(6523,[],known,fl,[],5)[0]!=(None,0):raise ValueError('late-fill reservation released early')
    if h.targets(6622,[],known,fl,[],5)[0]!=(49,5):raise ValueError('reservation not released after uncertainty')
    # Known bad (empty retired registry) must visibly allow the dangerous extra order.
    h,o=T.hook();h.cancelling(o,6522.);h.retired.clear()
    if h.targets(6523,[],known,fl,[],5)[0]!=(49,5):raise ValueError('late-fill falsifier did not fire')
    # Pending-to-active identity tracks the accepted object's notifications.
    h,o=T.hook();h.cancelling(o,6522.);new=dict(o,kq=5);h.objects[o['_gid']]=new
    if h.targets(6523,[],known,fl,[],5)[0]!=(49,5):raise ValueError('accepted notification identity not transferred')
    suite=unittest.defaultTestLoader.loadTestsFromModule(T)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise ValueError('controls failed')
    print('V2_RESERVATION_CONTROLS_PASS',flush=True)

def feature_check(day,group,S):
    import datetime as dt
    import json
    import numpy as np
    import pandas as pd
    from .pf_model_spec import SETS
    from .pf_own_order_grid_check import same
    from .pf_io import mem_check
    same([1,np.nan],[1,np.nan])
    try:same([1],[2])
    except ValueError:pass
    else:raise ValueError('feature falsifier')
    out=S.folder(day,group);checks=json.loads((out/'checks.json').read_text())
    start=pd.Timestamp(day,tz='UTC').timestamp()*1000
    path=ROOT/f'data/predict_fun/research/step5/{day}/rows_rl_d2_pk.parquet'
    r=pd.read_parquet(path,columns=['mid','tau']+SETS['ALL'],filters=[('tau','>=',start),('tau','<',start+86400000)])
    by={m:g for m,g in r.groupby('mid',sort=False)};rows=[]
    for mode in S.GROUPS[group]:
        p=out/'generations'/f'{mode}.parquet';g=pd.read_parquet(p);a=g[g.order_age_ms.notna()];count=0
        for mid,aa in a.groupby('mid',sort=False):
            raw=by[mid];tt=raw.tau.to_numpy();d=aa.decision.to_numpy();j=np.searchsorted(tt,d,side='right')-1
            if (j<0).any() or (d-tt[j]>100).any() or (tt[j]>d).any():raise ValueError('feature event clock mismatch')
            same(aa.order_age_ms.to_numpy(),d-aa.t_act.to_numpy())
            for key in SETS['ALL']:same(aa[key].to_numpy(),raw[key].to_numpy(dtype=np.float32)[j])
            k=np.searchsorted(tt,tt[j]-250,side='right')-1;ok=(k>=0)&(tt[j]-tt[np.maximum(0,k)]<=350)
            for key,dest in [('bn_imb','d_bnimb250'),('pf_imb1','d_pfimb250')]:
                x=raw[key].to_numpy(dtype=np.float32);wanted=np.where(ok,x[j]-x[np.maximum(k,0)],np.nan)
                same(aa[dest].to_numpy(),wanted)
            count+=len(aa)
        rows.append(dict(mode=mode,decisions=count,generation_sha=S.sha(p)))
    mem_check('default pull feature identity')
    S.write(out/'feature_check.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',checks_sha=S.sha(out/'checks.json'),
        cache_sha=S.sha(path),rows=rows,controls='Exact identity passes; changed-feature refusal fires; no future/stale feature time accepted.'))
    print('FEATURE_IDENTITIES_PASS',day,group,sum(x['decisions'] for x in rows),flush=True)

def main():
    if sys.argv[1]=='declare':declaration();return
    S=configure();S.check()
    command=sys.argv[1]
    if command=='controls':controls()
    elif command=='run':
        import gc
        from .pf_default_pull_engine import run
        day,group=sys.argv[2:];run(day,group);gc.collect();feature_check(day,group,S)
    elif command=='fit':
        from .pf_default_pull_fit import run
        run()
    elif command=='report':
        from .pf_default_pull_report import run
        run(sys.argv[2])
    else:raise ValueError('unknown v2 command')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1] in ('progress','slot'):
        # Keep bookkeeping stdlib-only and outside the numeric slot.
        import importlib.util
        spec=importlib.util.spec_from_file_location('progress',HERE/'pf_default_pull_progress.py')
        P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P);P.OUT=OUT
        if sys.argv[1]=='slot':
            import json
            b=P.busy();print(json.dumps(b));sys.exit(bool(b))
        P.update(*sys.argv[2:])
    else:main()
