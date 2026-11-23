"""Saved-label semantic check only; no panel rebuild, policy fitting or new days."""
import hashlib
import json
from pathlib import Path
import pandas as pd

ROOT=Path('/home/yuqing/ctaNew')
PIN=Path('/tmp/pf-takelogic-review-cccac366')
BASE=Path('predictfun/reviews/maker_trace_20261122')
HERE=Path(__file__).resolve().parent
rec=json.loads((PIN/BASE/'v2/chains/take_logic/receipt.json').read_text())
path=ROOT/BASE/'v2/chains/take/takes.csv'
key=str(path.relative_to(ROOT))
assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['inputs'][key]
t=pd.read_csv(path,float_precision='round_trip')
e={'input_sha256':rec['inputs'][key], 'n':len(t)}
cols=['cls_rec','cls_sim','cls_cert_0']
e['classes']={k:t[k].value_counts().to_dict() for k in cols}
u=t[t.take_class=='timing_sensitive'].copy()
reduce={'reduce_partial','full_close','reversal_reducing'}
add={'add','from_flat','overshoot'}
e['ambiguous_fine_class_n']=len(u)
e['all_available_classes_reduce']=int((u[cols].isin(reduce)|u[cols].isna()).all(axis=1).sum())
e['at_least_one_class_reduce']=int(u[cols].isin(reduce).any(axis=1).sum())
e['all_available_classes_add']=int((u[cols].isin(add)|u[cols].isna()).all(axis=1).sum())
e['at_least_one_class_add']=int(u[cols].isin(add).any(axis=1).sum())
e['class_combinations']=u.fillna({c:'absent' for c in cols}).groupby(cols).size().reset_index(name='n').to_dict('records')
e['old_overshoots']=t[t.take_class=='overshoot'][['day','mid','rank','sid','dq','pos_rec','pos_cert_0','cls_rec','cls_sim','cls_cert_0']].to_dict('records')
h=pd.read_csv(PIN/BASE/'v2/chains/take_logic/hazard_cells.csv',float_precision='round_trip')
e['conditional_events']=h[(h.scope=='pooled')&(h.factor=='inv_ledger_conditional')&(h.level=='yes')].to_dict('records')
p=pd.read_csv(PIN/BASE/'v2/chains/take_logic/patterns.csv',float_precision='round_trip')
e['discordant_take_counts']=p[(p.scope=='pooled')&p.pattern.str.startswith('status:')].to_dict('records')
c=pd.read_csv(PIN/BASE/'v2/chains/correction_v1/census_graded.csv',dtype={'day':str})
e['census_base_and_overlaps']=c[(c.scope=='pooled')&(c.day=='all')&(c.evidence_grade=='observed_execution')].to_dict('records')
(HERE/'label_evidence.json').write_text(json.dumps(e,indent=2,default=lambda o:int(o))+'\n')
print(json.dumps({k:v for k,v in e.items() if k not in ('class_combinations','census_base_and_overlaps','old_overshoots')},indent=2))
