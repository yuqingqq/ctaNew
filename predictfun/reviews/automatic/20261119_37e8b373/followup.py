"""Bounded impact and attribution follow-up on the two saved consumed days."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/yuqing/ctaNew')
SCOPE = json.loads((HERE/'scope.json').read_text())
sys.path.insert(0, str(Path(SCOPE['isolated_checkout'])/'predictfun/research'))
import numpy as np
import pandas as pd
import pf_pipeline_policy as P

DATA = ROOT/'data/predict_fun/research'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def run(day):
    bench=json.loads((DATA/'pipeline_bench_v2_20261119/bench.json').read_text())
    f=DATA/'pipeline_bench_v2_20261119'/f'pred_{day}.parquet'
    assert sha(f)==bench['outputs'][f.name]
    pred=pd.read_parquet(f); r=pred[pred.kind=='retain'].copy(); post=pred[pred.kind=='post'].copy()
    old=P.triggers_of(pred).droplevel('day')
    impossible=(r.J_Eq==0)&(r.J_Eqm!=0)&r.J_ok
    assert impossible.any()
    strict=r.copy(); strict.loc[impossible,'J_ok']=False
    new=P.triggers_of(strict).droplevel('day')
    ids=old.index.union(new.index)
    changed=ids[old.reindex(ids).fillna(99).values != new.reindex(ids).fillna(99).values]
    bad=r[impossible].copy(); bad['first_pull']=bad.oid.map(old).fillna(99)
    reached=bad[bad.j<=bad.first_pull]
    parity=P.contract_parity(bad,0.,n=2*len(bad))
    # Exact wrong-action counterexample was already reproduced by verify.py;
    # these are the original saved rows, not manufactured predictions.
    kp=DATA/'keep_pull_20261117'/day
    kr=json.loads((kp/'receipt.json').read_text())
    assert sha(kp/'orders.parquet')==kr['outputs']['orders.parquet'] and sha(kp/'fills.parquet')==kr['outputs']['fills.parquet']
    O=pd.read_parquet(kp/'orders.parquet').set_index('oid'); E=O[O.status=='eligible']
    fl=pd.read_parquet(kp/'fills.parquet'); fl=fl[fl.clock==50.].copy()
    fl['value']=fl.q*np.where(fl.oid.map(E.side)==0,1.,-1.)*(fl.oid.map(E.mark)-fl.oid.map(E.p))
    ej=pd.Series(P.A.endj_of(E.t_full.values,E.g_struct.values),index=E.index)
    def values(hit):
        ts=pd.Series(E.index.map(hit).fillna(99).values,index=E.index)
        act=(ts>=P.KP.J0)&(ts<=P.KP.NSTEP)&(ts<ej)
        rm=fl.oid.map(act).values & (fl.t.values>=P.KP.landing(fl.oid.map(ts).values))
        vals=pd.Series(np.where(rm,0.,fl.value),index=fl.index).groupby(fl.oid).sum().reindex(E.index,fill_value=0.)
        return vals,ts,act
    v,ts,act=values(old); corrected,_,_=values(new)
    pa,_,_=values(pd.Series(P.KP.J0,index=E.index))
    source=json.loads((DATA/'pipeline_policy_v2_20261119'/f'policy_{day}.json').read_text())['values']['MODEL-J'][P.B_NAME]
    assert abs(v.sum()-source['KEEP']-source['score'])<1e-6
    # Distinguish no RETAIN prediction / opportunity from choosing never to pull.
    cat=np.where(~E.index.isin(r.oid),'no_RETAIN_prediction',np.where(act & (ts==P.KP.J0),'first_step',np.where(act,'later_pull','never_model_pull_with_predictions')))
    tab=pd.DataFrame(dict(category=cat,model=v,pullall=pa,delta=v-pa)).groupby('category').agg(orders=('delta','size'),model_c=('model','sum'),pullall_c=('pullall','sum'),delta_c=('delta','sum'))
    assert abs(tab.delta_c.sum()-(source['score']-source['PULLALL']))<1e-6
    parts={}
    for side,g in post.groupby('side'):
        g=g[g.J_ok & (g.J_V>0)]
        edge=np.where(g.side==0,1.,-1.)*(g.pf_mid-g.p)
        missing=~g.oid.isin(E.index)
        assert (g.loc[missing,'oid'].map(O.status)=='refused_post_only').all()
        assert np.allclose(g.value_m50,edge*g.q_m50+g.qm_m50,rtol=0,atol=1e-6)
        assert np.allclose(g.J_V,edge*g.J_Eq+g.J_Eqm,rtol=0,atol=1e-6)
        parts[str(side)]=dict(n=len(g), predicted_value_mean=float(g.J_V.mean()), actual_value_mean=float(g.value_m50.mean()),
                             predicted_quantity=float(g.J_Eq.sum()),actual_quantity=float(g.q_m50.sum()),
                             edge_prediction_error_c=float((edge*(g.J_Eq-g.q_m50)).sum()),
                             movement_prediction_error_c=float((g.J_Eqm-g.qm_m50).sum()),
                             with_MODEL_J_c=float(g.oid.map(v).fillna(0.).sum()),with_PULL_ALL_c=float(g.oid.map(pa).fillna(0.).sum()))
    result=dict(prediction_asof=bench['asof'],bad_moment_rows=len(bad),bad_moment_orders=int(bad.oid.nunique()),
                bad_moment_abs_max_c=float(bad.J_Eqm.abs().max()),bad_moment_positive=int((bad.J_Eqm>0).sum()),bad_moment_negative=int((bad.J_Eqm<0).sum()),
                bad_moment_rows_reached=len(reached),parity_on_all_bad_rows=list(parity),changed_first_trigger_orders=len(changed),
                original_model_value_c=float(v.sum()),reject_zero_moment_model_value_c=float(corrected.sum()),
                action_value_impact_c=float((corrected-v).sum()),attribution=tab.to_dict('index'),positive_POST_components=parts)
    print(json.dumps(dict(day=day,**result)),flush=True)
    return result


def main():
    out=dict(target_commit=SCOPE['target_commit'],asof=datetime.now(timezone.utc).isoformat(),days={})
    for day in ('20261026','20261027'):out['days'][day]=run(day)
    out['completed_asof']=datetime.now(timezone.utc).isoformat()
    with (HERE/'followup.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')


if __name__=='__main__':main()
