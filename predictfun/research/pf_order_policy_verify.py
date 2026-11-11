"""Artifact identity, engine parity and independent portfolio arithmetic."""
import datetime as dt
import math
import numpy as np
import pandas as pd
from . import pf_order_policy_spec as S


def reconcile(w,f,q,ind,g,expected):
    if len(w)!=expected or w.mid.duplicated().any() or set(w.mid)!=set(ind.mid):raise ValueError('window population')
    if not f.mid.isin(w.mid).all() or not q.mid.isin(w.mid).all():raise ValueError('orphan event')
    if g.id.duplicated().any():raise ValueError('duplicate generation identity')
    if len(g)!=int(q.kind.eq('place').sum()):raise ValueError('generation/place count')
    if not np.isclose(g.entry_q.sum(),f.q.sum(),atol=1e-6,rtol=0):raise ValueError('generation share identity')
    if not np.isclose(g.entry_settle.sum()/100,w.pnl.sum(),atol=1e-6,rtol=0):raise ValueError('generation settlement identity')
    groups={m:z for m,z in f.groupby('mid',sort=False)};ii=ind.set_index('mid')
    for row in w.itertuples():
        z=groups.get(row.mid,f.iloc[:0]);signed=[];cash=[];net=last=area=0.
        for x in z.sort_values(['t','side'],kind='stable').itertuples():
            if x.t<last or x.q<=0:raise ValueError('invalid fill chronology')
            area+=(x.t-last)*net**2;last=x.t;sg=1 if x.side==0 else -1
            signed.append(sg*x.q);cash.append(-sg*x.q*x.pt/100);net=math.fsum(signed)
            if abs(net)<1e-9:net=0.
        area+=max(0.,240.1-last)*net**2
        for actual,wanted,label in [(net,row.net_end,'net'),(math.fsum(z.settle),row.pnl,'settle'),
            (math.fsum(cash),ii.loc[row.mid,'cash'],'cash'),(area,ii.loc[row.mid,'inventory_square_s'],'inventory area')]:
            if not np.isclose(actual,wanted,atol=1e-5,rtol=1e-10):raise ValueError('independent '+label)
    good=g[(g.status=='eligible')&g.label.notna()]
    if (good.label_q>good.postcancel_rem+1e-7).any() or (good.label_q<0).any():raise ValueError('shadow overfill')
    if ((good.cancel_effective<good.decision)|(good.remaining>5+1e-7)).any():raise ValueError('invalid continuation population')


def exact(a,b):
    pd.testing.assert_frame_equal(a.reset_index(drop=True),b.reset_index(drop=True),check_exact=True)


def check_run(day,group):
    from .pf_order_policy_engine import GROUPS,folder
    out=folder(day,group);ind=pd.read_parquet(out/'independent_windows.parquet');checked=[]
    for mode in GROUPS[group]:
        w,f,q=(pd.read_parquet(out/k/S.stem(mode)) for k in ('win','fills','requests'))
        g=pd.read_parquet(out/'generations'/f'{mode}.parquet')
        reconcile(w,f,q,ind[ind['mode']==mode],g,8 if group=='smoke' else 288)
        checked.append(dict(mode=mode,windows=len(w),generations=len(g),fills=len(f),statuses=g.status.value_counts().to_dict(),
                            hashes={k:S.sha(out/k/S.stem(mode)) for k in ('win','fills','requests')},generation_sha=S.sha(out/'generations'/f'{mode}.parquet')))
    if group in ('smoke','build'):
        ref=S.ROOT/f'data/predict_fun/research/step4/ladder_tokyoreal_custom_e2_preds_t60k_b14rX/{day}'
        mids=pd.read_parquet(out/'win'/S.stem('reference')).mid
        for kind in ('win','fills','requests'):
            a=pd.read_parquet(out/kind/S.stem('reference'));b=pd.read_parquet(ref/kind/S.stem('reference'))
            exact(a,b[b.mid.isin(mids)])
    S.write(out/'checks.json',dict(as_of=dt.datetime.now(dt.timezone.utc).isoformat(),day=day,group=group,
        variants=checked,reference_exact=group in ('smoke','build'),status='PASS'))


if __name__=='__main__':
    import sys
    check_run(*sys.argv[1:])
