"""Measure the scope of review counterexamples on saved small tables only."""
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root=Path(sys.argv[1]).resolve(); out=Path(__file__).resolve().parent
target='541aeca9ad9cefddd6b6346146072ea05692e1ed'
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==target
sys.path.insert(0,str(root/'predictfun/research'))
import pf_keep_pull_exec_windows as X
import pf_timing_case_audit as A
np,pd=X.np,X.pd
data=Path('/home/yuqing/ctaNew/data/predict_fun/research')
kj=json.loads((data/'keep_pull_exec_windows_known_v6_20261118/known.json').read_text())
K=pd.read_parquet(data/'keep_pull_exec_windows_known_v6_20261118/known_cases.parquet')
KE=pd.read_parquet(data/'keep_pull_exec_windows_known_v6_20261118/known_case_rows.parquet')
book,_=X.book_bounds(K)
inputs={}
for nm in ('legs_timed2.parquet','legs_tk.parquet','pend_inc.parquet'):
 p=X.NOV8/nm; h=hashlib.sha256(p.read_bytes()).hexdigest()
 assert kj['inputs'][str(p)]==h; inputs[str(p)]=h
L8=pd.read_parquet(X.NOV8/'legs_timed2.parquet',columns=X.CKEY+['how','t_ms','t_e2'])
L8['tk']=pd.read_parquet(X.NOV8/'legs_tk.parquet',columns=['tk']).tk.values
L8['group']=np.where(L8.tk.values>=0,L8.tk.values,-1-np.arange(len(L8)))
status_lower=pd.Series(X.rest_range(L8.exec_s.values)[0]).groupby(L8.group.values).max()
ec=KE.copy(); ec['S3_lo_status']=status_lower.reindex(ec.group).values
assert np.array_equal(np.maximum(ec.S3_lo_status.values,ec.S3_lo_book.values),ec.S3_lo.values)
actual,_=X.window_checks_dom(K,KE,book,kj['R']['K7']['lo'],scen=('S3',))
restored,cols=X.window_checks_dom(K,ec,book,kj['R']['K7']['lo'],scen=('S3',))
assert np.all(book[np.isfinite(book)]>=kj['R']['K7']['lo'])
assert restored==actual

P=pd.read_parquet(X.NOV8/'pend_inc.parquet',columns=['mid','t_ms','ptick','inc'])
audit=json.loads((root/'predictfun/reviews/timing_case_audit_20261118/audit_v2.json').read_text())
raw=[json.loads(l) for l in (root/'predictfun/reviews/timing_case_audit_20261118/raw_records_v2.jsonl').read_text().splitlines()]
exact=[]
for x in audit['legs']:
 ev,lo,binding,domain,u=A.event_binding(x['group'],L8,P)
 assert lo==x['S3_lo'] and domain==x['A3_domain'] and u==x['A3_u']
 elems=[(z['recv_ns'],z['page_line_sha256'],z['element']) for z in raw if z['leg']==x['leg'] and z['source']=='research_rest']
 MR=A.maker_rows(elems)
 keys=['mid','exec_s','side','ptick','shares','price_executed','taker_index','taker_quote']
 ids=set()
 for _,r in binding.iterrows():
  good=np.logical_and.reduce([MR[c].values==r[c] for c in keys])
  assert good.any(); ids |= set(zip(MR.tx[good],MR.sid[good]))
 assert ids==set(map(tuple,x['A2_own_ids']))
 exact.append(dict(leg=x['leg'],endpoint_exact=True,identity_exact=True,clock_bound_above_critical=x['A4']['B']>=x['critical_ms']))

# An incorrect whole second still passes the content identity check on epoch seconds.
plant=list(A._planted_leg())
corrupt=copy.deepcopy(plant); corrupt[5]['exec_s']+=1.
bad=A.leg_eval(0,*corrupt)[0]
assert bad['A2']=='SAME-SETTLEMENT' and bad['A5']==A.VERDICT_OK
far=copy.deepcopy(plant); far[5]['exec_s']+=100_000.
good_refusal=A.leg_eval(0,*far)[0]
assert good_refusal['A2']=='ABSENT' and good_refusal['A5']!=A.VERDICT_OK

# Audit the saved economic evidence-tier arithmetic; no full economics rerun.
tiers,ok,hs=A.a6_ledger()
pd.testing.assert_frame_equal(tiers.reset_index(drop=True),pd.DataFrame(audit['A6']).reset_index(drop=True))
assert ok
v=tiers[(tiers.cancel=='CP') & (tiers.cls=='CANCEL_FIRST') & (tiers.tier=='EVIDENCE')]
assert int(v.fills.sum())==0
R=dict(target=target,asof=datetime.now(timezone.utc).isoformat(),consumed_days=['20261108','20261023'],
       untouched_validation_read=False,full_tape_replayed=False,inputs=inputs,
       exact_three_cases=exact,
       restored_REST_lower_bounds=dict(candidate_rows=len(KE),rows_with_changed_status_bound=int((KE.S3_lo_status!=ec.S3_lo_status).sum()),
                                      original=actual,restored=restored,scope='fixed saved candidate set; no candidate search or upper-bound remapping'),
       identity_tolerance=dict(shift_seconds=1.,A2=bad['A2'],A5=bad['A5'],large_wrong_second_refused=True),
       evidence_tiers=dict(reproduced=True,rows=len(tiers),cancel_first_evidence_backed=0,inputs=hs))
(out/'followup.json').write_text(json.dumps(R,indent=2)+'\n')
print(json.dumps({k:v for k,v in R.items() if k not in ['inputs','evidence_tiers']},indent=2))
print('Actual three case identities/endpoints are exact; restored REST bound leaves saved S3 count unchanged; A6 reproduces.')
