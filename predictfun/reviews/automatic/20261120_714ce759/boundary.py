"""Small final-tick engine/validator-interface control, no real tape replay."""
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
sys.path.insert(0, '/tmp/pf-review-714ce759/predictfun/research')
import pf_r2_label_sample as S

BE, L = S.BE, S.L
OUT = Path('/home/yuqing/ctaNew/predictfun/reviews/automatic/20261120_714ce759')
A = Path('/home/yuqing/ctaNew/data/predict_fun/research/r2_label_sample_v2_20261120/attempt4_pilot_A')
plan = json.loads((A/'plan.json').read_text())
near = collections.Counter()
for p in plan['partitions']:
    pools = S.gz_rows(S.part_paths(A, p['part'])[2].read_bytes())
    for pool in pools:
        for k in pool['keys']:
            if k['kind']=='keep' and k['decision_ms']==4950:
                near[(k['day'],k['side'])] += 1

def factory(B):
    t=L.ArmTeacher({},None); t.bind(B,None)
    # Deterministic policy stand-in; actual ArmTeacher.fn binding check and stats.
    t.decide=lambda v,b:dict(side=v['side'],state=v['state'],status='synthetic',action='WAIT' if v['state']=='absent' else 'KEEP')
    return t

path=BE.tape([BE.frame(0,0,50,52,bsz=0,asz=0)],[[4990.,0,50.,5.]],51.)
results=[]
for k,side in ((98,0),(99,0),(99,1)):
    ctx=BE.fork_state(path,BE.POST_KEEP,k,side)
    res=L.pair_bound(ctx,factory,'keep')
    stats={n:a['teacher'] for n,a in res['arms'].items()}
    assert res['status']=='ok' and res['reconciled']
    gate=all(t['binding_checked']>0 and t['binding_bad']==0 for t in stats.values())
    results.append(dict(k=k,side=side,Y=res['Y'],binding_by_arm={n:t['binding_checked'] for n,t in stats.items()},
                        binding_failures=sum(t['binding_bad'] for t in stats.values()),current_binding_gate_pass=gate))
assert results[0]['current_binding_gate_pass'] and results[2]['current_binding_gate_pass']
assert results[1]['Y']==5 and not results[1]['current_binding_gate_pass'] and results[1]['binding_failures']==0
obj=dict(target='714ce75978ab8dfec29c1fc41fd0dd800b982ad8',asof=datetime.now(timezone.utc).isoformat(),
         source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),synthetic_final_tick=results,
         saved_pool_final_tick_keep={'|'.join(map(str,k)):v for k,v in near.items()})
(OUT/'boundary.json').write_text(json.dumps(obj,indent=2)+'\n')
print(json.dumps(obj))
