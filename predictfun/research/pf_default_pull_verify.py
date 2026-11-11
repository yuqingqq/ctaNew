"""Validate actual saved portfolios and action populations before reporting."""
import datetime as dt
import numpy as np
import pandas as pd
from . import pf_default_pull_spec as S
from .pf_order_policy_verify import reconcile,exact

def actions(g):
    if g.id.duplicated().any():raise ValueError('duplicated generation')
    a=g[g.status.eq('eligible')]
    if (a.decision<a.t_act+a.probe_age-1e-7).any():raise ValueError('probe before assigned age')
    if (a.cancel_effective<a.decision).any():raise ValueError('cancel clock')
    if (a.label_q>a.postcancel_rem+1e-7).any():raise ValueError('keep overfill')
    if (a.reset_q>a.initial_size+1e-7).any():raise ValueError('reset overfill')
    if not np.allclose(a.label,a.keep_value-a.reset_value,atol=1e-9,rtol=0):raise ValueError('label decomposition')
    if ((a.d_bnimb250.notna()|a.d_pfimb250.notna()) & a.t_rel.lt(.25)).any():raise ValueError('warmup lookahead')

def check_run(day,group):
    out=S.folder(day,group);ind=pd.read_parquet(out/'independent_windows.parquet');checked=[]
    for mode in S.GROUPS[group]:
        w,f,q=(pd.read_parquet(out/k/S.stem(mode)) for k in ('win','fills','requests'))
        g=pd.read_parquet(out/'generations'/f'{mode}.parquet')
        reconcile(w,f,q,ind[ind['mode']==mode],g,8 if group=='smoke' else 288)
        actions(g)
        if mode!='reference' and (ind.loc[ind['mode']==mode,'max_inventory']>10+1e-6).any():raise ValueError('inventory range exceeded')
        checked.append(dict(mode=mode,windows=len(w),generations=len(g),fills=len(f),statuses=g.status.value_counts().to_dict(),
            hashes={k:S.sha(out/k/S.stem(mode)) for k in ('win','fills','requests')},generation_sha=S.sha(out/'generations'/f'{mode}.parquet')))
    if group in ('smoke','build'):
        ref=S.ROOT/f'data/predict_fun/research/step4/ladder_tokyoreal_custom_e2_preds_t60k_b14rX/{day}'
        mids=pd.read_parquet(out/'win'/S.stem('reference')).mid
        for kind in ('win','fills','requests'):
            a=pd.read_parquet(out/kind/S.stem('reference'));b=pd.read_parquet(ref/kind/S.stem('reference'))
            exact(a,b[b.mid.isin(mids)])
    S.write(out/'checks.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),day=day,group=group,
        variants=checked,independent_sha=S.sha(out/'independent_windows.parquet'),reference_exact=group in ('smoke','build'),status='PASS'))

if __name__=='__main__':
    import sys
    check_run(*sys.argv[1:])
