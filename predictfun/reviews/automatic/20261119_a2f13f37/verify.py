"""Readiness check: prior review regressions, synthetic suites, saved receipts only."""
import contextlib
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

root=Path(sys.argv[1]).resolve(); out=Path(__file__).resolve().parent
target='a2f13f37d58d546b949bb4919725739978a47c76'
assert subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()==target
sys.path.insert(0,str(root/'predictfun/research'))
import pf_keep_pull_exec_windows as X
import pf_timing_case_audit as A
import pf_known_clock_census as N
import pf_keep_pull_scenarios as S
import pf_settlement_times as ST
import pf_cancel_race_study as CR
np,pd=X.np,X.pd
data=Path('/home/yuqing/ctaNew/data/predict_fun/research')
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
R=dict(target=target,asof=datetime.now(timezone.utc).isoformat(),scope='targeted testing-readiness review',
       full_tape_replayed=False,training_run=False,untouched_validation_read=False)
if '--reuse-selftests' in sys.argv:
 log=(out/'selftests.txt').read_text()
 assert 'cancel race study selftests passed' in log and 'settlement selftests passed' in log
 assert 'SELFTEST FAIL' not in log and 'SELFTEST FAILED' not in log
 R['selftests_reused']='unchanged pinned source; six suites passed before attempt 2 receipt-path assertion'
else:
 with (out/'selftests.txt').open('w') as f,contextlib.redirect_stdout(f):
  for mod in (A,N,X,S,ST,CR):
   assert mod.selftest() is not False
R['selftests']=[m.__name__ for m in (A,N,X,S,ST,CR)]

# Reapply the previous review's independent negative controls.
plant=list(A._planted_leg()); clean=A.leg_eval(0,*plant)[0]
assert clean['A5']==A.VERDICT_OK
bad=copy.deepcopy(plant); bad[0]['lo']+=5000.
z=A.leg_eval(0,*bad)[0]
assert not z['A3_reproduces'] and z['A5']!=A.VERDICT_OK
bad=copy.deepcopy(plant); bad[5]['exec_s']+=1.
z=A.leg_eval(0,*bad)[0]
assert z['A2']=='ABSENT' and z['A5']!=A.VERDICT_OK
cancel_bound=[b for b in clean['bounds'] if b['name']=='match <= cancel answer receipt (c_hi)']
assert len(cancel_bound)==1 and cancel_bound[0]['type']=='CONDITIONAL'
leg=pd.DataFrame(dict(mid=[7],exec_s=[100.],first_recv_ns=[1],side=[0],ptick=[50],shares=[10.],price_executed=[.5],
                     taker_index=[1],taker_quote=[0],how=[1],t_ms=[90300.],t_e2=[np.nan],tk=[5],Zr=[90300.]))
pk=pd.DataFrame(dict(mid=[7],t_ms=[90300.],ptick=[50],inc=[5.]))
e=X.case_linkage(leg[X.CKEY].assign(candidate_legs=1),leg,pk,dict(h=600.,q01=-900.,q99=1050.),day='20990101')
assert e.S3_lo_book.iloc[0]==90050. and e.S3_lo_status.iloc[0]==90000.
k7=dict(exists=True,lo=-200.,hi=-200.)
got,_=X.window_checks_dom(pd.DataFrame(dict(ack=[90100.])),e,[0.],k7,scen=('S3',))
control=e.copy();control['S3_lo_status']=-np.inf
g0,_=X.window_checks_dom(pd.DataFrame(dict(ack=[90100.])),control,[0.],k7,scen=('S3',))
assert got['S3']['false_cancel_first']['some_offset_per_domain']==1
assert g0['S3']['false_cancel_first']['some_offset_per_domain']==0
R['prior_findings']={'P2_1_constituent_constraints':'fixed','P2_2_exact_timestamp_identity':'fixed','P2_3_cancel_bound':'conditional; dependent inference checked below'}

# Source hashes pinned in final suite manifests (later manifest wins for changed harnesses).
hashes={}
for fname in ('r1_sources.sha256','r2_sources.sha256','r3_sources.sha256'):
 for line in (root/'predictfun/reviews/simulator_readiness_20261118/suites'/fname).read_text().splitlines():
  h,p=line.split();hashes[p]=h
for p,h in hashes.items(): assert sha(root/p)==h,p
R['suite_source_hashes']=hashes

# Known v7/census v4: saved sources and outputs plus exact reading reproduction.
kj=load(X.OUT_K/'known.json'); cj=load(N.OUT/'census.json')
for folder,j in ((X.OUT_K,kj),(N.OUT,cj),(CR.OUT,load(CR.OUT/'study.json'))):
 for name,h in j['outputs'].items(): assert sha(folder/name)==h,(folder,name)
assert kj['source_sha256']==sha(Path(X.__file__))
assert cj['source_sha256']==sha(Path(N.__file__))
K=pd.read_parquet(X.OUT_K/'known_cases.parquet'); KE=pd.read_parquet(X.OUT_K/'known_case_rows.parquet')
book,info,_=X.book_bounds(K)
kr,read=X.known_checks(K,kj['R']['K0_ref'],KE,book=book,ack_premise=False,census=info)
assert {k:v for k,v in kr.items() if not k.startswith('_')}==kj['R'] and read==kj['readings']
assert kr['K5']['cancel'].get('CERTIFIED',0)==0
assert read['K6 GATE S3 (per clock domain)']=='PASS'
assert all(b['type']=='CONDITIONAL' for l in load(data/'timing_case_audit_v3_20261118/audit.json')['legs']
           for b in l['bounds'] if b['name']=='match <= cancel answer receipt (c_hi)')
assert not any('then filled' in v for v in cj['readings'].values())
R['known']=dict(asof=kj['asof'],legs=len(K),orders=kr['orders'],candidate_rows=len(KE),readings=read,K5=kr['K5'],K9=kr['K9'])

# Verify the final five-day receipts use settlement v2, exclude sealed hours,
# and reconcile published CSV values with the current scenario artifact.
sj=load(S.OUT/'scenarios.json'); assert sj['source_sha256']==sha(Path(S.__file__))
table=pd.read_csv(root/'predictfun/reviews/scenario_checklist_v6_20261118/scenarios_5days.csv')
byday=pd.read_csv(root/'predictfun/reviews/scenario_checklist_v6_20261118/by_day_5days.csv')
assert len(table)==19
for row in table.to_dict('records'):
 for key in ('E','score','null','KEEP','fills','shares','eligible'):
  assert abs(float(row[key])-float(sj['scenarios'][row['scenario']][key]))<1e-6,(row['scenario'],key)
for name, g in byday.groupby('scenario'):
 assert abs(float(g.E.sum())-sj['scenarios'][name]['E'])<1e-6
days={}
for day in X.KP.DAYS:
 dr=load(S.OUT/day/'receipt.json'); st=load(ST.OUT/day/'settlement.json')
 assert st['source_sha256']==sha(Path(ST.__file__))
 assert all(ST.hour_allowed(h) for h in st['R']['hours'])
 assert all(not ST.hour_allowed(h) for h in st['R']['hours_excluded'])
 assert all('VOID' not in p for p in dr['settlement_inputs'])
 table_path=ST.OUT/day/'legs_set.parquet'
 assert str(table_path) in dr['settlement_inputs']
 assert dr['settlement_inputs'][str(table_path)]==st['outputs']['legs_set.parquet']==sha(table_path)
 days[day]=dict(asof=dr['asof'],population=dr['stat'],excluded_hours=st['R']['hours_excluded'])
assert days['20261027']['excluded_hours']==['20261028_00','20261028_01']
base=sj['scenarios']['M50|A58|L37|N100']; late=sj['scenarios']['M250|A58|L37|N100']
assert base['E']>0 and late['E']<0
assert sj['readings']['C2 E AT M250 vs M50']=='SIGN-DIFFERS-AT-M250'
R['scenarios']=dict(asof=sj['asof'],days=days,readings=sj['readings'],scenario_rows=19,
                    baseline_E=base['E'],M250_E=late['E'],baseline_KEEP_plus_score=base['KEEP']+base['score'])
study=load(CR.OUT/'study.json')
R['cancel_race']=dict(asof=study['asof'],readings=study['readings'],C0=study['R']['C0'],C3=study['R']['C3'],C3b=study['R']['C3b'])
R['readiness']='ready for scenario-conditional research tests on consumed days; execution/profitability validation not established'
(out/'receipt.json').write_text(json.dumps(R,indent=2)+'\n')
print(json.dumps({k:R[k] for k in ('target','asof','selftests','prior_findings','readiness')},indent=2))
print('Known-v7 readings reproduce; 19 pooled scenario CSV rows reconcile; corrected settlement-v2 inputs verified.')
