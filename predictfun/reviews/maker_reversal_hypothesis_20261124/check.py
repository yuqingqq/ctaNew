"""Independent source-to-output arithmetic and corruptions; no action certificate."""
import copy, math, time
from collections import Counter
import build as B
import numpy as np
import pandas as pd

def stamp(r):return B.key(r)+(int(r['delay_ms']),r['mode'],r['control'])
def independent_status(c,g,delay,mode,control):
 if c['group']=='unresolved':return 'unresolved_ending',[]
 lo,hi,ws=c['lo'],c['hi'],c['ws'];first=math.ceil((lo-delay-5000-ws)/1000);last=math.floor((hi-delay-ws)/1000)
 if first<5 or last>=300 or any(not g['valid'][i] for i in range(first,last+1)):return 'context_or_boundary_unknown',[]
 if mode=='post_compatible' and not math.isfinite(c['post_hi']):return 'post_bound_unknown',[]
 spans=[];hits=[]
 for k,on in enumerate(g['events']):
  t=ws+1000*k
  if on and (mode=='timing_only' or t>=c['post_hi']) and t+delay<=hi and t+delay+5000>=lo:
   hits.append(t);spans.append((t+delay,t+delay+5000))
 if not spans:return 'no_match',[]
 endpoints=sorted(set([lo,hi]+[min(hi,max(lo,t)) for s in spans for t in s]));probes=endpoints+[(a+b)/2 for a,b in zip(endpoints,endpoints[1:])]
 full=all(any(a<=t<=b for a,b in spans) for t in probes)
 return ('all_times_compatible' if full else 'possible_only'),hits

def verify_cases(rr,cases,grids):
 cd={B.key(c):c for c in cases};expected={(k+(d,m,c)) for k in cd for d in (0,1000) for m in ('timing_only','post_compatible') for c in ('correct_side','wrong_side')}
 B.need(len(rr)==704 and {stamp(x) for x in rr}==expected,'case membership')
 for r in rr:
  c=cd[B.key(r)];side=c['side'] if r['control']=='correct_side' else ('ask' if c['side']=='bid' else 'bid');g=grids[(c['day'],c['mid'],side)]
  B.need(r['group']==c['group'] and r['post_grade']==c['post_grade'],'ending identity')
  B.need(not B.truth(r['private_pull_observed']),'private action claim')
  B.need(float(r['lo_ms'])==c['lo'] and float(r['hi_ms'])==c['hi'],'window changed')
  st,hits=independent_status(c,g,int(r['delay_ms']),r['mode'],r['control'])
  B.need(r['status']==st,'alignment status')
  B.need(r['reversal_times_ms']=='|'.join(map(str,hits)),'anchor clock')
  B.need(B.truth(r['evaluable'])==(st in ('no_match','possible_only','all_times_compatible')),'denominator')
  B.need(B.truth(r['matched'])==(st in ('possible_only','all_times_compatible')),'match')

def verify_events(ev,grids):
 expected=set()
 for (d,m,s),g in grids.items():
  a=g['r']*(1 if s=='bid' else -1)
  for k in range(5,300):
   if all(math.isfinite(x) for x in a[k-5:k+1]) and a[k-5]>=1 and a[k]<=-1 and a[k-1]>-1:expected.add((d,m,s,g['ws']+k*1000))
 actual={(r['day'],int(r['mid']),r['side'],int(r['t_ms'])) for r in ev}
 B.need(len(ev)==len(expected) and actual==expected,'signal census')
 for r in ev:
  g=grids[(r['day'],int(r['mid']),r['side'])];k=int(r['k'])
  B.need(all(float(r[f])==float(g['r'][i]) for f,i in [('r5_prior',k-5),('r5_previous',k-1),('r5_now',k)]),'signal features')

def main():
 start=time.process_time();B.resource.setrlimit(B.resource.RLIMIT_CPU,(18,20));B.bound();out=B.HERE/'outputs';result=B.json.loads((out/'result.json').read_text())
 for p,h in result['hashes'].items():B.need(B.digest(out/p)==h,'output hash '+p)
 grids,cases,cs,ev=B.load();rr=B.rows(out/'case_alignment.csv');verify_cases(rr,cases,grids);verify_events(B.rows(out/'reversals.csv'),grids)
 # All aggregates independently reduce the saved case records.
 summary=B.rows(out/'summary.csv')
 for s in summary:
  part=[r for r in rr if (r['group'],r['delay_ms'],r['mode'],r['control'])==(s['group'],s['delay_ms'],s['mode'],s['control']) and (s['day']=='all' or r['day']==s['day'])]
  n=sum(B.truth(x['evaluable']) for x in part);hit=sum(B.truth(x['matched']) for x in part)
  values=dict(orders=len(part),evaluable=n,matched=hit,all_times=sum(x['status']=='all_times_compatible' for x in part),possible_only=sum(x['status']=='possible_only' for x in part),no_match=sum(x['status']=='no_match' for x in part),unknown=len(part)-n)
  B.need(all(int(s[k])==v for k,v in values.items()),'summary counts')
  B.need((s['rate']=='' if n==0 else abs(float(s['rate'])-hit/n)<1e-15),'summary rate')
 fresh=B.placebo(cases,grids);saved=B.rows(out/'draws.csv.gz');B.need(len(saved)==len(fresh),'draw membership')
 for a,b in zip(saved,fresh):
  for k,v in b.items():B.need((a[k]=='' if v is None else str(v)==a[k]),'draw reproducibility')
 ag=B.aggregates(B.evaluate(cases,grids));read=B.readings(ag,fresh)
 for k,v in read.items():B.need(result[k]==v,'computed reading')
 # Independent bounds on null counts and membership, avoiding missing denominators or duplicates.
 B.need(len({(x['draw'],x['delay_ms'],x['day']) for x in saved})==len(saved),'duplicate null')
 for x in saved:B.need(0<=int(x['matched'])<=int(x['evaluable'])<=40,'null counts')
 refused={}
 def bad(name,f):
  try:f()
  except ValueError as e:refused[name]=str(e);return
  raise AssertionError('falsifier escaped: '+name)
 bad('missing_case',lambda:verify_cases(rr[:-1],cases,grids))
 for name,field,value in [('private_pull','private_pull_observed','True'),('clock','reversal_times_ms','999'),('window','hi_ms','0'),('false_match','matched','False')]:
  z=copy.deepcopy(rr);i=0
  if name=='false_match':value=str(not B.truth(z[i]['matched']))
  z[i][field]=value
  bad(name,lambda z=z:verify_cases(z,cases,grids))
 z=copy.deepcopy(rr);i=next(i for i,x in enumerate(z) if x['group']=='exhausted');z[i]['group']='conditional_withdrawal';bad('exhausted_as_withdrawal',lambda:verify_cases(z,cases,grids))
 z=B.rows(out/'reversals.csv')
 if z:z[0]['r5_now']='999'
 else:z=[dict(day='20261024',mid=0,side='bid',t_ms=0,k=0,r5_prior=0,r5_previous=0,r5_now=999)]
 bad('signal_corruption',lambda:verify_events(z,grids))
 B.guard();B.dump(B.HERE/'checks.json',dict(pass_all=True,cases=len(cases),alignment_rows=len(rr),signal_events=len(ev),draw_rows=len(saved),falsifiers=refused,cpu_seconds=time.process_time()-start));print((B.HERE/'checks.json').read_text())
if __name__=='__main__':main()
