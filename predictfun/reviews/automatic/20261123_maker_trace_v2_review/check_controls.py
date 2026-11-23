"""Pinned-source synthetic controls and receipt identity, no development edits."""
import os
os.environ['MT_RUN'] = 'v2'
import sys, json, hashlib, importlib.util, time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
PIN=Path('/tmp/pf-mt-review-0e21193f')
ROOT=Path('/home/yuqing/ctaNew')
OUT=Path(__file__).resolve().parent
RS=PIN/'predictfun/research'
REC=PIN/'predictfun/reviews/maker_trace_20261122'
sys.path.insert(0,str(RS))
sys.path.insert(0,str(REC/'tools'))
import pf_mt_common, pf_mt_lifecycle, pf_mt_lifecycle_selftest
import pf_mt_decision, pf_mt_decision_selftest, closes_selftest
mods=(pf_mt_common,pf_mt_lifecycle,pf_mt_lifecycle_selftest,pf_mt_decision,pf_mt_decision_selftest)
assert all(str(x.__file__).startswith(str(PIN)) for x in mods)
def invoke(label,fn):
    t=time.process_time()
    try: fn()
    except SystemExit as e:
        assert e.code in (0,None),(label,e.code)
    return dict(label=label,cpu_s=time.process_time()-t)
checks=[]
checks.append(invoke('lifecycle_selftest',lambda:pf_mt_lifecycle_selftest.main(pf_mt_lifecycle)))
checks.append(invoke('closes_selftest',closes_selftest.main))
sp=importlib.util.spec_from_file_location('identity_control_pinned',REC/'tools/decision_identifiability_control.py')
mi=importlib.util.module_from_spec(sp)
sp.loader.exec_module(mi)
assert mi.M is pf_mt_decision and mi.LC is pf_mt_lifecycle
checks.append(invoke('identifiability_and_decision_selftest',mi.main))

# Positive/negative inventory distinction: buying 10 then selling 30 increases abs position from 10 to 20.
F=dict(d=np.zeros(2,dtype=int),a=np.zeros(2,dtype=int),role=np.ones(2,dtype=int),sid=np.array([1,2]),
       settle=np.array([100,200]),s=np.array([1,-1]),q=np.array([10.,30.]),self=np.zeros(2,dtype=bool))
iv=pf_mt_decision.inventory(F,np.zeros(2,dtype=int),np.ones(2,dtype=bool),('20261024',))
row=iv[(iv.scope=='rank1')&(iv.day=='all')&(iv.pos_bucket=='all')].iloc[0]
assert row.n_reduce==1 and abs(10-30)>abs(10)

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
cr=json.loads((REC/'v2/closes/closes_receipt.json').read_text())
hashes={}
for name,v in cr['outputs'].items():
    candidates=[REC/'v2/closes'/name,ROOT/'predictfun/reviews/maker_trace_20261122/v2/closes'/name]
    p=next(p for p in candidates if p.exists())
    hashes[name]=sha(p)==v['sha256']
assert all(hashes.values())
cp=ROOT/'predictfun/reviews/maker_trace_20261122/v2/closes/closes.parquet'
schema=pq.read_schema(cp)
print('closes_schema',schema.names)
v1=ROOT/'data/predict_fun/research/maker_trace_v1_20261122/econ'
v2=ROOT/'data/predict_fun/research/maker_trace_v2_20261123/econ'
unchanged={n:sha(v1/n)==sha(v2/n) for n in ('fills_valued.parquet','maker_market.parquet','identities.json')}
assert all(unchanged.values())
report=dict(target='0e21193f2d403d08bb5172aaf727a6a93110e8a9',checks=checks,output_hashes=hashes,
            economic_inputs_unchanged=unchanged,overshoot_synthetic=dict(position_before=10,position_after=-20,
            reported_reduce=int(row.n_reduce),actually_reduces=False),source_files={x.__name__:str(x.__file__) for x in mods})
(OUT/'controls_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
