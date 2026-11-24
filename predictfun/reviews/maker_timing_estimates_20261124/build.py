"""Incremental interval estimates from accepted windows and signed terminal terms."""
import csv
import gzip
import hashlib
import io
import json
import math
import resource
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
OLD=ROOT/'predictfun/reviews/maker_trace_20261122/v2/chains'
INPUTS={
 'post':OLD/'post_fill/orders.csv', 'grade':OLD/'correction_v1/post_regrade_orders.csv',
 'pull':OLD/'pull/pulls.csv', 'post_receipt':OLD/'post_fill/receipt.json',
 'grade_receipt':OLD/'correction_v1/receipt_post.json', 'pull_receipt':OLD/'pull/receipt.json',
 'signed':ROOT/'predictfun/reviews/maker_terminal_chain_20261124/orders.csv',
 'signed_receipt':ROOT/'predictfun/reviews/maker_terminal_chain_20261124/summary.json',
}
CONTEXT=('btc_against_1s_union','btc_against_5s_union','inv_built_5s_union','behind_touch_union','own_fill_5s_union','book_away_5s_union')
POST_CONTEXT=('spread_cat','btc_1s_set','btc_5s_set','btc_30s_set','inv_set','inv_certain_cat','touch_set','phase')

def need(ok,msg):
 if not ok: raise ValueError(msg)

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):
 opener=gzip.open if str(p).endswith('.gz') else open
 with opener(p,'rt',newline='') as f:return list(csv.DictReader(f))
def num(v):
 try:return float(v)
 except (ValueError,TypeError):return math.nan
def integer(v):return int(float(v))
def key(r):return (str(r['day']),integer(r.get('mid',r.get('market'))),r.get('addr',r.get('maker','')).lower(),r.get('mk_hash',r.get('order_hash')))
def finite_window(lo,hi):return math.isfinite(lo) and math.isfinite(hi) and 0<=lo<=hi
def lookup(rs):
 d={key(r):r for r in rs};need(len(d)==len(rs),'duplicate exact identity');return d
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def write_csv(p,rs):
 if not rs:return
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
 if str(p).endswith('.gz'):
  with p.open('wb') as f:
   with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0) as g:g.write(buf.getvalue().encode())
 else:p.write_text(buf.getvalue())

def plan():
 files=list(INPUTS.values())+list(HERE.glob('*.py'))+[HERE/'contract.md']
 need(not (HERE/'manifest.json').exists(),'write once manifest')
 dump(HERE/'manifest.json',{'files':{str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))}})
 print('Bound existing sources; no timing outcome inspected.')

def bound():
 for name,sha in json.loads((HERE/'manifest.json').read_text())['files'].items():need(digest(ROOT/name)==sha,'source hash '+name)
 for name,rec in [('post','post_receipt'),('grade','grade_receipt'),('pull','pull_receipt')]:
  receipt=json.loads(INPUTS[rec].read_text());expected=receipt['outputs'][INPUTS[name].name]['sha256']
  need(digest(INPUTS[name])==expected,'accepted receipt '+name)
 need(digest(INPUTS['signed'])==json.loads(INPUTS['signed_receipt'].read_text())['orders_sha256'],'signed receipt')

def point_fields(lo,hi,enabled):
 d={'effect_lo_ms':int(lo) if math.isfinite(lo) else None,'effect_hi_ms':int(hi) if math.isfinite(hi) else None,
    'midpoint_ms':(lo+hi)/2 if enabled else None,'width_s':(hi-lo)/1000 if finite_window(lo,hi) else None}
 for delay in (0,250,1000):
  d[f'submit_lo_D{delay}_ms']=int(lo-delay) if enabled else None
  d[f'submit_hi_D{delay}_ms']=int(hi) if enabled else None
 return d

def post_row(p,g):
 need(key(p)==key(g),'post key')
 for f in ('post_lo_union_ms','post_hi_ms'):need(integer(p[f])==integer(g[f]),'post window disagreement')
 lo,hi=num(g['post_lo_union_ms']),num(g['post_hi_ms']);grade=g['grade']
 enabled=grade in ('G2_bound_constrained','G4_scenario_only') and finite_window(lo,hi)
 label='bounded_conditional' if enabled and grade=='G2_bound_constrained' else ('scenario_only' if enabled else 'estimate_refused')
 r=dict(day=p['day'],mid=integer(p['mid']),rank=integer(p['rank']),maker=p['addr'].lower(),order_hash=p['mk_hash'],side=p['y_side'],tick=integer(p['tick']),
   grade=grade,estimate_class=label,estimate_enabled=enabled,ws_ms=integer(p['ws_ms']),first_record_ms=integer(p['first_hi_ms']),
   midpoint_is_observed_command=False,**point_fields(lo,hi,enabled))
 r['relative_midpoint_s']=(r['midpoint_ms']-r['ws_ms'])/1000 if enabled else None
 r['record_lead_from_midpoint_s']=(r['first_record_ms']-r['midpoint_ms'])/1000 if enabled else None
 for c in POST_CONTEXT:r[c]=p[c]
 return r

def end_class(r,g,s):
 lo,hi=num(r['removal_lo_ms']),num(r['removal_hi_ms'])
 cause='unknown_signed_budget';expiry='unknown';conflict=False
 if s:
  cause=s['status']
  exp=num(s['expiration_s'])*1000
  if exp>0 and finite_window(lo,hi):expiry='after_window' if exp>hi else ('inside_window' if exp>=lo else 'before_window')
  ub=num(r['x_size_ub']);conflict=math.isfinite(ub) and num(s['signed_quote_shares'])>ub+1e-6
 flags=(r['status']+' '+g['status_packet']+' '+g['census_order_flags']).lower()
 invalid=(r['window_status']!='ok' or any(z in flags for z in ('invalid','conflict','capacity_bounds_void','unplaceable')) or g['grade'] in ('G0_unbounded_or_invalid','G1_conflict','G3_constrained_status_flagged'))
 usable=finite_window(lo,hi) and not invalid and not conflict
 if cause=='exhausted_by_observed_fills':label='exhausted_no_terminal_pull'
 elif conflict:label='signed_size_capacity_conflict'
 elif not usable:label='timing_estimate_refused'
 elif cause=='unresolved_remainder':label='conditional_withdrawal_before_expiry' if expiry=='after_window' else 'conditional_withdrawal_or_expiry'
 else:label='unsigned_exhaustion_or_withdrawal'
 pull=usable and label.startswith('conditional_withdrawal')
 return dict(ending_class=label,signed_status=cause,expiry_relation=expiry,size_capacity_conflict=conflict,window_usable=usable,
             pull_estimate_enabled=pull,private_pull_observed=False)

def successor_gap(endlo,endhi,nextrow):
 if not nextrow:return 'no_usable_successor',None,None
 if not nextrow['estimate_enabled'] or not finite_window(endlo,endhi):return 'next_post_or_end_refused',None,None
 lo=nextrow['effect_lo_ms']-endhi;hi=nextrow['effect_hi_ms']-endlo
 return ('ordered_after_under_windows' if lo>0 else ('ordered_before_under_windows' if hi<0 else 'overlap_or_order_tied')),lo/1000,hi/1000

def pull_row(r,g,s,posts):
 k=key(r);c=end_class(r,g,s);lo,hi=num(r['removal_lo_ms']),num(r['removal_hi_ms'])
 out=dict(day=r['day'],mid=k[1],rank=integer(g['rank']),maker=k[2],order_hash=k[3],view=r['view'],side=r['y_side'],tick=integer(r['y_ptick']),
   original_status=r['status'],window_status=r['window_status'],removal_method=r['removal_method'],post_grade=g['grade'],**c,
   **point_fields(lo,hi,c['window_usable']))
 out['pull_midpoint_ms']=out['midpoint_ms'] if c['pull_estimate_enabled'] else None
 out['midpoint_kind']='withdrawal_or_invalidation_scenario' if c['pull_estimate_enabled'] else 'disappearance_only'
 out['signed_quote_shares']=s['signed_quote_shares'] if s else None
 out['observed_shares']=s['observed_shares'] if s else r['cum_shares']
 out['expiration_s']=s['expiration_s'] if s else None
 h=r['successor_hash'];nxt=posts.get((k[0],k[1],k[2],h)) if h else None
 label,gl,gh=successor_gap(lo,hi,nxt if c['window_usable'] else None)
 out.update(successor_hash=h,successor_kind=r['successor'],successor_link=label,gap_lo_s=gl,gap_hi_s=gh,successor_is_proven_replacement=False)
 for f in CONTEXT:out[f]=r[f]
 return out

def union_windows(raw):
 v={r['view']:r for r in raw}
 if not {'S_FIFO','S_NOPRIO'}<=set(v):return 'missing_view',None,None
 bounds=[num(v['S_FIFO']['removal_hi_ms']),num(v['S_NOPRIO']['removal_hi_ms']),num(v['S_FIFO']['x_rhi_S_FIFO_LO'])]
 lows=[num(r['removal_lo_ms']) for r in v.values()]
 if not all(finite_window(lo,hi) for lo in lows for hi in bounds):return 'unbounded_or_inconsistent_view',None,None
 return 'three_view_union',int(min(lows)),int(max(bounds))

def summarize(posts,pulls,unions,signed):
 def counts(rs,f):return dict(sorted(Counter(r[f] for r in rs).items()))
 def width(rs):
  a=sorted(r['width_s'] for r in rs if r['width_s'] is not None)
  return dict(n=len(a),le_1s=sum(x<=1 for x in a),le_10s=sum(x<=10 for x in a),median_s=(a[(len(a)-1)//2]+a[len(a)//2])/2 if a else None)
 postgroups=[]
 for day in ['all']+sorted({r['day'] for r in posts}):
  for rank in [0,1,2,3]:
   part=[r for r in posts if (day=='all' or r['day']==day) and (rank==0 or r['rank']==rank)]
   for grade in sorted({r['grade'] for r in posts}):
    z=[r for r in part if r['grade']==grade]
    postgroups.append(dict(day=day,rank=rank,grade=grade,orders=len(z),enabled=sum(r['estimate_enabled'] for r in z),**width(z)))
 byview={}
 for view in sorted({r['view'] for r in pulls}):
  rr=[r for r in pulls if r['view']==view]
  byview[view]=dict(rows=len(rr),classes=counts(rr,'ending_class'),conditional_withdrawal_width=width([r for r in rr if r['pull_estimate_enabled']]),
    successors=counts(rr,'successor_link'),expiry=counts([r for r in rr if r['signed_status']=='unresolved_remainder'],'expiry_relation'))
 contexts=[]
 for view in byview:
  for label in sorted({r['ending_class'] for r in pulls if r['view']==view}):
   z=[r for r in pulls if r['view']==view and r['ending_class']==label]
   for f in CONTEXT:
    for value,n in counts(z,f).items():contexts.append(dict(view=view,ending_class=label,feature=f,classification=value,n=n,denominator=len(z)))
 return dict(post_orders=len(posts),post_grades=counts(posts,'grade'),post_estimates=counts(posts,'estimate_class'),
     pull_views=byview,signed_orders=len(signed),signed_statuses=counts(signed,'status'),
     union_statuses=counts(unions,'union_status'),signed_union_statuses=counts([r for r in unions if r['signed_status']!='unknown_signed_budget'],'union_status')) ,postgroups,contexts

def run():
 start=time.process_time();bound();out=HERE/'outputs';need(not out.exists(),'write once outputs')
 pr=rows(INPUTS['post']);gr=lookup(rows(INPUTS['grade']));sr=rows(INPUTS['signed']);sd=lookup(sr);rr=rows(INPUTS['pull'])
 need(len(pr)==4149 and len(gr)==4149 and set(map(key,pr))==set(gr),'post population')
 need(len(sr)==88 and set(sd)<=set(gr),'signed population')
 need(len(rr)==6640 and len({(key(r),r['view']) for r in rr})==len(rr),'pull population')
 posts=[post_row(p,gr[key(p)]) for p in pr];pdct={(r['day'],r['mid'],r['maker'],r['order_hash']):r for r in posts}
 pulls=[pull_row(r,gr[key(r)],sd.get(key(r)),pdct) for r in rr]
 grouped=defaultdict(list)
 for r in rr:grouped[key(r)].append(r)
 unions=[]
 for k,rs in sorted(grouped.items()):
  st,lo,hi=union_windows(rs);s=sd.get(k)
  unions.append(dict(day=k[0],mid=k[1],maker=k[2],order_hash=k[3],union_status=st,union_lo_ms=lo,union_hi_ms=hi,
   signed_status=s['status'] if s else 'unknown_signed_budget',views='|'.join(sorted(r['view'] for r in rs))))
 summary,pg,cx=summarize(posts,pulls,unions,sr)
 need(time.process_time()-start<20,'CPU guard');need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<1450,'memory guard')
 out.mkdir()
 for name,data in [('post_estimates.csv.gz',posts),('ending_estimates.csv.gz',pulls),('union_windows.csv.gz',unions),('posting_coverage.csv',pg),('terminal_context.csv',cx)]:write_csv(out/name,data)
 summary.update(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),cpu_seconds=time.process_time()-start,hashes={p.name:digest(p) for p in out.iterdir()})
 dump(out/'summary.json',summary);print(json.dumps(summary,indent=2))

if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
