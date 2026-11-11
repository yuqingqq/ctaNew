"""Causal warm-up correction; preserve original frozen source and run artifacts."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from . import pf_order_policy_spec as S
from .pf_order_policy_inputs import Window,FEATURES


class CausalWindow(Window):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if len(self.rt):
            old=np.searchsorted(self.rt,self.rt-250,side='right')-1
            valid=(old>=0)&(self.rt-self.rt[np.maximum(old,0)]<=350)
            self.db[~valid]=np.nan;self.dp[~valid]=np.nan


def controls():
    from .tests.test_order_policy import fixture,order
    from .pf_model_spec import SETS
    inputs,f,t=fixture();inputs.by[1][1][-1,SETS['ALL'].index('bn_imb')]=.7
    a=CausalWindow(inputs,1,f,t,0.);bad=Window(inputs,1,f,t,0.)
    x=a.features(100,order(t0=50),0,0,1,1);xb=bad.features(100,order(t0=50),0,0,1,1)
    inputs.by[1][1][-1,SETS['ALL'].index('bn_imb')]=-.7
    b=CausalWindow(inputs,1,f,t,0.);bad2=Window(inputs,1,f,t,0.)
    y=b.features(100,order(t0=50),0,0,1,1);yb=bad2.features(100,order(t0=50),0,0,1,1)
    np.testing.assert_allclose([x[k] for k in FEATURES],[y[k] for k in FEATURES],equal_nan=True)
    if xb['d_bnimb250']==yb['d_bnimb250']:raise ValueError('known-bad future-read falsifier failed')
    np.testing.assert_array_equal(a.calms,bad.calms)
    if not np.isnan(x['d_bnimb250']) or not np.isnan(x['d_pfimb250']) or a.db[130]!=bad.db[130]:raise ValueError('warm-up correction scope')
    print('WARMUP_CONTROLS_PASS',flush=True)


def audit():
    S.check();controls();records=[]
    for group,days in [('build',S.DAYS),('eval',S.TEST)]:
        for day in days:
            folder=S.OUT/'runs'/group/day;c=json.loads((folder/'checks.json').read_text())
            for v in c['variants']:
                p=folder/'generations'/f"{v['mode']}.parquet"
                if S.sha(p)!=v['generation_sha']:raise ValueError('generation identity')
                a=pd.read_parquet(p);early=a.t_rel.lt(.25)&a.t_rel.ge(0);finite=a[['d_bnimb250','d_pfimb250']].notna().any(axis=1)
                records.append(dict(group=group,day=day,mode=v['mode'],generations=len(a),
                    early_decisions=int(early.sum()),affected=int((early&finite).sum()),
                    affected_ids=a.loc[early&finite,'id'].tolist(),input_sha=v['generation_sha']))
    r=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),version=3,status='AUDIT_ONLY',source_sha=S.sha(__file__),
        issue='Original Window used x[-1] for imbalance deltas during its first250ms. Calm already masked that history, but continuation features did not. Corrected deltas are unknown until past history is available. Original source/artifacts retained.',
        controls='Causal future-mutation positive and original implementation known-bad falsifier pass; calm unchanged; mature history unchanged.',rows=records)
    S.write(S.OUT/'warmup_v3_audit.json',r);S.write(S.REVIEW/'warmup_v3_audit.json',r)
    print(json.dumps({g:sum(x['affected'] for x in records if x['group']==g) for g in ('build','eval')}),flush=True)


if __name__=='__main__':audit()
