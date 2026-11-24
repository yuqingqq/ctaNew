"""Action-first packet plumbing; no condition-based selection."""
import csv,gzip,hashlib,json,subprocess,sys,time,resource
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2];V=R/'data/predict_fun/research/maker_trace_v2_20261123';C=R/'predictfun/reviews/maker_trace_20261122/v2/chains';OLD=R/'predictfun/reviews/maker_action_chains_20261124';T=R/'predictfun/reviews/maker_timing_estimates_20261124';D=R/'data/predict_fun/research/maker_action_population_20261124'
sys.path.insert(0,str(R/'predictfun/research'))
import pf_mt_signed_evidence_v3 as S
def sha(p):return S.sha(p)
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def write(p,rr):
 if not rr:raise ValueError('empty table without schema')
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr)
def flag(x):return x is True or x=='True'
def guard(start,cap=6):
 assert time.process_time()-start<cap,'CPU stage cap';assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<1200*1024,'RSS cap'
def bound():
 m=json.loads((H/'manifest.json').read_text())
 for p,h in m['files'].items():assert sha(R/p)==h,p
 for p in m['sources']:assert subprocess.check_output(['git','-C',str(R),'show','HEAD:'+p])==(R/p).read_bytes(),'unfrozen '+p
 return m
def responses():
 out={}
 for root in ('maker_terminal_chain_20261124','maker_signed_six_20261124'):
  for p in sorted((R/'data/predict_fun/research'/root).glob('batch_*.json')):
   for r in json.loads(p.read_text()):
    v=r.get('result');assert isinstance(v,dict),'cached RPC error'
    kind='receipt' if 'transactionHash' in v else 'input';tx=v.get('transactionHash',v.get('hash','')).lower();assert tx,'missing tx'
    key=(tx,kind)
    if key in out:
     for f in (('blockHash','blockNumber','logs','status') if kind=='receipt' else ('input','blockHash','blockNumber')):assert out[key][f]==v[f],'cache conflict'
    else:out[key]=v
 return out
def relation(lo,hi,nlo,nhi,valid):
 if not valid:return 'unavailable'
 if nlo>hi:return 'sequential_after_under_bounds'
 if nhi<lo:return 'successor_before_ending_possible_overlap'
 return 'ordering_ambiguous'
def budget(original,making,complete,roles):
 if not complete:return 'incomplete_recorded_history'
 if set(roles)!={'maker'}:return 'mixed_or_taker_role'
 assert 0<making<=original,'budget'
 return 'exhausted' if original==making else 'positive_remainder_not_a_pull'
