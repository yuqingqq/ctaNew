"""Descriptive decomposition at the already-declared -30 c severe-loss threshold.

Bounded saved-artifact check, not a fitted risk filter or a strategy result.
Read only committed artifacts at the reviewed SHA. No outcomes are removed
from the total, and no threshold is selected using this calculation.
"""
from pathlib import Path
import gzip, hashlib, io, json, subprocess
import numpy as np
import pandas as pd
ROOT=Path('/home/yuqing/ctaNew')
OUT=ROOT/'predictfun/reviews/automatic/20261121_tail_question_530eac3b'
SHA='530eac3b230659f1d90c630aac9ecd0f80f139e5'
def blob(p):return subprocess.check_output(['git','-C',str(ROOT),'show',SHA+':'+p])
def decompose(v,w):
    v,w=np.asarray(v,float),np.asarray(w,float); severe=v < -30.; W=w.sum()
    total=float(np.dot(w,v)); bad=float(np.dot(w[severe],v[severe])); rest=float(np.dot(w[~severe],v[~severe]))
    assert np.isclose(total,bad+rest,rtol=0,atol=1e-8)
    return dict(n=len(v),weight_sum=float(W),positive=int((v>0).sum()),negative=int((v<0).sum()),zero=int((v==0).sum()),
                severe_n=int(severe.sum()),severe_weight=float(w[severe].sum()),weighted_value_sum_c=total,
                severe_weighted_value_sum_c=bad,remaining_weighted_value_sum_c=rest,
                J_c=total/W,severe_contribution_c_per_opportunity=bad/W,remaining_contribution_c_per_opportunity=rest/W,
                remaining_subset_mean_c=rest/w[~severe].sum() if (~severe).any() else None)
# Positive and known-bad controls for the exact severe boundary/accounting.
h=decompose([10.,-30.,-31.,0.],[2.,1.,3.,1.])
assert h['severe_n']==1 and h['weighted_value_sum_c']==-103. and h['remaining_weighted_value_sum_c']==-10.
assert not np.isclose(h['weighted_value_sum_c'],h['severe_weighted_value_sum_c']+h['remaining_weighted_value_sum_c']+1.)
base='predictfun/reviews/movement_shrink_20261121/outputs/'
b=blob(base+'eval_rows.csv.gz');a=blob(base+'actions.csv')
R=pd.read_csv(io.BytesIO(gzip.decompress(b)),float_precision='round_trip')
A=pd.read_csv(io.BytesIO(a),float_precision='round_trip',dtype={'side':str})
records=[]
for stage,g in R.groupby('head',sort=True):
    assert set(g.day)=={20261027}
    for side,t in [('all',g)]+[(str(s),g[g.side==s]) for s in (0,1)]:
        d=decompose(t.value,t.weight);r=A[(A['head']==stage)&(A.side==side)&(A.rule=='all')].iloc[0]
        assert d['n']==int(r.n) and d['severe_n']==int(r.severe_n)
        assert np.isclose(d['J_c'],r.J,rtol=1e-12,atol=1e-9)
        assert np.isclose(d['severe_weighted_value_sum_c'],r.severe_sum,rtol=1e-12,atol=1e-9)
        records.append(dict(day='20261027',stage=stage,side=side,**d))
result=dict(source_commit=SHA,asof_receipt='2026-11-21T12:43Z',severe_threshold_c=-30.,threshold_predeclared=True,
            sources={base+'eval_rows.csv.gz':hashlib.sha256(b).hexdigest(),base+'actions.csv':hashlib.sha256(a).hexdigest()},
            descriptive_only=True,rule_fitted=False,formal_heavy_tail_distribution_established=False,controls_passed=True,rows=records)
(OUT/'decomposition.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,indent=2,sort_keys=True))
