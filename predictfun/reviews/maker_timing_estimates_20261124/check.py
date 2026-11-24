"""Independent interval/gate recount and known-bad result refusals."""
import copy
import json
import math
from collections import Counter
import build as B

def truth(v):return str(v).lower() in ('true','1')
def eq(a,b,msg):
 if a in ('',None) and b in ('',None):return
 B.need(math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-8),msg)

def points(r,lo,hi,enabled):
 eq(r['effect_lo_ms'],lo,'lower changed');eq(r['effect_hi_ms'],hi,'upper changed')
 eq(r['midpoint_ms'],(lo+hi)/2 if enabled else None,'unsupported point')
 for d in [0,250,1000]:
  eq(r[f'submit_lo_D{d}_ms'],lo-d if enabled else None,'submission lower')
  eq(r[f'submit_hi_D{d}_ms'],hi if enabled else None,'submission upper')

def check_posts(out,original,grades):
 B.need(len(out)==len(original),'post membership')
 idx={(r['day'],int(r['mid']),r['maker'],r['order_hash']):r for r in out}
 B.need(set(idx)==set(original),'post identity')
 for k,p in original.items():
  r=idx[k];g=grades[k];lo=int(g['post_lo_union_ms']);hi=int(g['post_hi_ms'])
  enabled=g['grade'] in ['G2_bound_constrained','G4_scenario_only'] and 0<=lo<=hi
  B.need(truth(r['estimate_enabled'])==enabled,'post gate')
  B.need(not truth(r['midpoint_is_observed_command']),'private post claim')
  points(r,lo,hi,enabled)
  for f in B.POST_CONTEXT:B.need(r[f]==p[f],'post context')

def check_pulls(out,raw,grades,signed):
 idx={(r['day'],int(r['mid']),r['maker'],r['order_hash'],r['view']):r for r in out}
 B.need(len(idx)==len(raw),'ending membership')
 for x in raw:
  k=B.key(x);r=idx[(*k,x['view'])];g=grades[k];s=signed.get(k)
  lo=int(x['removal_lo_ms']);hi=int(x['removal_hi_ms']);status=s['status'] if s else 'unknown_signed_budget'
  cap=float(x['x_size_ub']) if x['x_size_ub'] else math.nan
  conflict=bool(s and math.isfinite(cap) and float(s['signed_quote_shares'])>cap+1e-6)
  invalid=x['window_status']!='ok' or g['grade'] in ('G0_unbounded_or_invalid','G1_conflict','G3_constrained_status_flagged')
  flags=' '.join([x['status'],g['status_packet'],g['census_order_flags']]).lower()
  invalid=invalid or any(v in flags for v in ('invalid','conflict','capacity_bounds_void','unplaceable'))
  usable=not conflict and not invalid and 0<=lo<=hi
  expiry='unknown'
  if s and float(s['expiration_s'])>0 and 0<=lo<=hi:
   exp=float(s['expiration_s'])*1000
   expiry='after_window' if exp>hi else ('before_window' if exp<lo else 'inside_window')
  if status=='exhausted_by_observed_fills':label='exhausted_no_terminal_pull'
  elif conflict:label='signed_size_capacity_conflict'
  elif not usable:label='timing_estimate_refused'
  elif status=='unresolved_remainder':label='conditional_withdrawal_before_expiry' if expiry=='after_window' else 'conditional_withdrawal_or_expiry'
  else:label='unsigned_exhaustion_or_withdrawal'
  pull=usable and status=='unresolved_remainder'
  B.need(r['ending_class']==label and r['expiry_relation']==expiry,'cause/expiry gate')
  B.need(truth(r['pull_estimate_enabled'])==pull,'false pull')
  B.need(not truth(r['private_pull_observed']),'private cancel claim')
  points(r,lo,hi,usable);eq(r['pull_midpoint_ms'],(lo+hi)/2 if pull else None,'pull point')

def run():
 B.bound();out=B.HERE/'outputs';summary=json.loads((out/'summary.json').read_text())
 for name,sha in summary['hashes'].items():B.need(B.digest(out/name)==sha,'output hash')
 pr=B.lookup(B.rows(B.INPUTS['post']));gr=B.lookup(B.rows(B.INPUTS['grade']));sr=B.rows(B.INPUTS['signed']);sd=B.lookup(sr);rr=B.rows(B.INPUTS['pull'])
 posts=B.rows(out/'post_estimates.csv.gz');pulls=B.rows(out/'ending_estimates.csv.gz')
 check_posts(posts,pr,gr);check_pulls(pulls,rr,gr,sd)
 # Successor gap uses exactly the independently checked next order and retained end window.
 pdct={(r['day'],int(r['mid']),r['maker'],r['order_hash']):r for r in posts}
 for r in pulls:
  n=pdct.get((r['day'],int(r['mid']),r['maker'],r['successor_hash']))
  possible=n and truth(n['estimate_enabled']) and truth(r['window_usable'])
  if possible:
   lo=(float(n['effect_lo_ms'])-float(r['effect_hi_ms']))/1000;hi=(float(n['effect_hi_ms'])-float(r['effect_lo_ms']))/1000
   eq(r['gap_lo_s'],lo,'successor lower');eq(r['gap_hi_s'],hi,'successor upper')
   label='ordered_after_under_windows' if lo>0 else ('ordered_before_under_windows' if hi<0 else 'overlap_or_order_tied')
   B.need(r['successor_link']==label,'successor order')
  else:B.need(r['gap_lo_s']=='' and r['gap_hi_s']=='','unidentified successor point')
  B.need(not truth(r['successor_is_proven_replacement']),'replacement claim')
 # Complete deterministic output reproduction, including unions and all summaries.
 ps=[B.post_row(p,gr[k]) for k,p in pr.items()]
 psd={(r['day'],r['mid'],r['maker'],r['order_hash']):r for r in ps}
 ends=[B.pull_row(r,gr[B.key(r)],sd.get(B.key(r)),psd) for r in rr]
 unions=B.rows(out/'union_windows.csv.gz');groups={}
 for r in rr:groups.setdefault(B.key(r),[]).append(r)
 B.need(len(unions)==len(groups),'union membership')
 for r in unions:
  k=(r['day'],int(r['mid']),r['maker'],r['order_hash']);status,lo,hi=B.union_windows(groups[k])
  B.need(r['union_status']==status,'union status');eq(r['union_lo_ms'],lo,'union lower');eq(r['union_hi_ms'],hi,'union upper')
 sums,pg,cx=B.summarize(ps,ends,unions,sr)
 for k,v in sums.items():B.need(summary[k]==v,'summary '+k)
 # CSV round-trip numeric types checked using normalized writer serialization.
 import csv,io
 def normalized(rs):
  f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs);return list(csv.DictReader(io.StringIO(f.getvalue())))
 B.need(B.rows(out/'posting_coverage.csv')==normalized(pg),'posting coverage')
 B.need(B.rows(out/'terminal_context.csv')==normalized(cx),'terminal context')
 falsifiers={}
 tests=[]
 p=copy.deepcopy(posts);p[0]['effect_lo_ms']=float(p[0]['effect_lo_ms'])+1;tests.append(('shifted_window',lambda:check_posts(p,pr,gr)))
 tests.append(('omitted_order',lambda:check_posts(posts[1:],pr,gr)))
 e=copy.deepcopy(pulls);e[0]['private_pull_observed']=True;tests.append(('private_pull_claim',lambda:check_pulls(e,rr,gr,sd)))
 e2=copy.deepcopy(pulls);i=next(i for i,r in enumerate(e2) if r['ending_class']=='exhausted_no_terminal_pull');e2[i]['pull_midpoint_ms']=1
 tests.append(('exhausted_as_pull',lambda:check_pulls(e2,rr,gr,sd)))
 e3=copy.deepcopy(pulls);i=next(i for i,r in enumerate(e3) if r['signed_status']=='unresolved_remainder');e3[i]['expiry_relation']='fake'
 tests.append(('expiry_corruption',lambda:check_pulls(e3,rr,gr,sd)))
 for name,fn in tests:
  try:fn()
  except ValueError as exc:falsifiers[name]=str(exc)
  else:raise AssertionError('falsifier escaped '+name)
 result=dict(pass_all=True,post_orders=len(posts),ending_view_rows=len(pulls),unique_endings=len(unions),falsifiers=falsifiers)
 B.dump(B.HERE/'checks.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':run()
