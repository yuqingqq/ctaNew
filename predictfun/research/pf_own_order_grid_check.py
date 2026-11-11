"""Verify every saved primary feature decision against its actual timestamped cache row."""
import datetime as dt
import json
import numpy as np
import pandas as pd
from . import pf_order_policy_spec as S


def same(a,b):
    if not np.allclose(a,b,rtol=0,atol=0,equal_nan=True):raise ValueError('causal cached feature identity')


def run():
    S.check();same([1.,np.nan],[1.,np.nan])
    try:same([1.],[2.])
    except ValueError:pass
    else:raise ValueError('known-bad feature accepted')
    audit=json.loads((S.OUT/'warmup_v3_audit.json').read_text());checks=[]
    for day in S.DAYS:
        start=pd.Timestamp(day,tz='UTC').timestamp()*1000
        path=S.ROOT/f'data/predict_fun/research/step5/{day}/rows_rl_d2_pk.parquet'
        raw=pd.read_parquet(path,columns=['mid','tau','t_rel','bn_imb','pf_imb1'],filters=[('tau','>=',start),('tau','<',start+86400000)])
        by={m:r for m,r in raw.groupby('mid',sort=False)}
        for record in [r for r in audit['rows'] if r['day']==day]:
            p=S.OUT/'runs'/record['group']/day/'generations'/f"{record['mode']}.parquet"
            if S.sha(p)!=record['input_sha']:raise ValueError('generation changed')
            g=pd.read_parquet(p);a=g[g.order_age_ms.notna()];n=0;min_history=float('inf')
            for mid,rows in a.groupby('mid',sort=False):
                r=by[mid];t=r.tau.to_numpy();ws=rows.ws.iloc[0]
                if t[0]!=ws or not np.all(np.diff(t)==50):raise ValueError('cached grid is not complete from window start')
                j=np.searchsorted(t,rows.decision.to_numpy(),side='right')-1;k=np.searchsorted(t,t[j]-250,side='right')-1
                if np.any(k<0):raise ValueError('actual action used unavailable history')
                for field in ('t_rel','bn_imb','pf_imb1'):same(rows[field].to_numpy(),r[field].to_numpy()[j])
                for source,target in [('bn_imb','d_bnimb250'),('pf_imb1','d_pfimb250')]:
                    v=r[source].to_numpy(dtype=np.float32);same(rows[target].to_numpy(),v[j]-v[k])
                n+=len(rows);min_history=min(min_history,float((t[j]-t[0]).min()))
            checks.append(dict(day=day,group=record['group'],mode=record['mode'],decisions=n,
                minimum_history_ms=min_history if n else None,cache_sha=S.sha(path),generation_sha=record['input_sha']))
    total=sum(x['decisions'] for x in checks)
    d=json.loads((S.OUT/'causal_v3_declaration.json').read_text())
    if total!=d['feature_decisions_checked']:raise ValueError('audit population identity')
    result=dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),status='PASS',source_sha=S.sha(__file__),
        comparison_sha=S.sha(S.OUT/'comparison_v3.json'),decisions=total,checks=checks,
        control='Positive identity and known-bad changed-feature refusal pass. Every original primary derivative equals its actual causal250ms cache difference; grids complete from window start.')
    S.write(S.OUT/'grid_check.json',result);S.write(S.REVIEW/'grid_check.json',result)
    print('GRID_VERIFIED',total,'decisions',flush=True)


if __name__=='__main__':run()
