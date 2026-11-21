"""Bounded receipt controls and computed interpretation; no fits or real simulation."""
import copy
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path
import verify as V
sys.path.insert(0,str(V.RC))
import r3_checks as C1
import r3_checks_v2 as C2
C1.RES=V.PIN/'predictfun/research';C1.REPO=V.PIN
out={};assert not C1.check_G(V.A.OUT,out) and len(out['G']['bad'])==2
old=copy.deepcopy(out);out={};assert C2.check_G2(V.A.OUT,out)
cases={};original=C2.AR.OUT
with tempfile.TemporaryDirectory(dir=V.OUT) as td:
    td=Path(td)
    try:
        for case in ('clean','required_callback_missing','binding_failed','cap_failed'):
            dst=td/case/'parts';shutil.copytree(original/'parts',dst)
            rp=dst/'part_000.json';fp=dst/'part_000.jsonl.gz';rr=V.S.gz_rows(fp.read_bytes())
            w=next(x for x in rr if x['status']=='ok');x=next(x for x in w['reached'] if x['status']=='ok')
            if case=='required_callback_missing':x['binding_checked']=0;x['key']['decision_ms']=1000
            if case=='binding_failed':x['binding_bad']=1
            if case=='cap_failed':w['episodes']['RICH']['cap_ok']=False
            fp.write_bytes(V.S.gz_bytes(rr));rec=V.read(rp);rec['sha256']=V.sha(fp);rp.write_text(json.dumps(rec))
            C2.AR.OUT=dst.parent;cases[case]=C2.check_G2(V.A.OUT,{})
            assert cases[case]==(case=='clean')
    finally:C2.AR.OUT=original
V.AE=V.E
fz=V.read(V.A.OUT/'fit.json');train={k:v['insample_wmse']/v['insample_wmse_zero'] for k,v in fz['models'].items() if k.startswith('RICH')}
ch=V.pd.read_csv(V.R.OUT/'churn.csv');ch=ch[(ch.policy=='RICH')&(ch.action=='PULL')]
exp=V.pd.read_csv(V.R.OUT/'exposure.csv');exp=exp[(exp.policy=='RICH')&exp.key.str.startswith('absent|model|c_gen=')]
tr=V.pd.read_csv(V.A.OUT/'eval/coverage_train.csv');tr=tr[(tr['head']=='post')&(tr.dim=='c_gen')]
res=dict(old_G=old,new_G=out,G2_falsifiers=cases,train_mse_ratios=train,
         rich_first_pull=dict(n=len(ch),post_mean=float(ch.value_post.mean()),review_mean=float(ch.value_review.mean()),dt=sorted(ch.dt.unique().tolist())),
         generation3=dict(replay_share=float(exp[exp.key=='absent|model|c_gen=>=3'].n.sum()/exp.n.sum()),training_share=float(tr[tr.value=='>=3'].rows.sum()/tr.rows.sum())))
# Independent count/rank/hash falsifier for the first saved label partition.
plan=V.read(V.E.EVAL_DIR/'plan.json');part=plan['partitions'][0];rp,fp,qp=V.S.part_paths(V.E.EVAL_DIR,0);rec=V.read(rp);raw=(fp.read_bytes(),qp.read_bytes());rr,qq=map(V.S.gz_rows,raw)
assert not V.S.validate(plan,V.sha(V.E.EVAL_DIR/'plan.json'),part,rr,qq,rec,raw)
bad=copy.deepcopy(rr);next(x for x in bad if x['status']=='ok')['result']['Y']+=1
b=V.S.gz_bytes(bad);r=copy.deepcopy(rec);r['sha256']=V.hashlib.sha256(b).hexdigest()
assert V.S.validate(plan,V.sha(V.E.EVAL_DIR/'plan.json'),part,bad,qq,r,(b,raw[1]))
res['changed_label_refused']=True
(V.OUT/'controls.json').write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print(json.dumps(res,sort_keys=True))
