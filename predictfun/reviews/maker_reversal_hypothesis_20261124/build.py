"""One frozen reversal/withdrawal timing hypothesis; no fitted rule or private-action claim."""
import csv, gzip, hashlib, io, json, math, resource, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
OLD=ROOT/'predictfun/reviews/maker_timing_estimates_20261124'
PANEL=ROOT/'predictfun/reviews/maker_trace_20261122/v2/chains/posting_opportunity/rows.parquet'
FILES={'panel':PANEL,'panel_receipt':PANEL.parent/'receipt.json','signed':OLD/'outputs/signed_chain_estimates.csv',
 'post':OLD/'outputs/post_estimates.csv.gz','union':OLD/'outputs/union_windows.csv.gz','summary':OLD/'outputs/summary.json',
 'readout':OLD/'readout.json','checks':OLD/'checks.json'}
SEED=20261124; ND=500

def need(v,s):
 if not v:raise ValueError(s)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def rows(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt',newline='') as f:return list(csv.DictReader(f))
def number(x):
 try:return float(x)
 except (TypeError,ValueError):return math.nan
def truth(x):return x is True or x=='True'
def key(r):return str(r['day']),int(r['mid']),r['maker'].lower(),r['order_hash']
def write(p,rs,fields=None):
 need(bool(rs) or fields is not None,'missing empty-table schema '+str(p));buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rs[0]) if rs else fields,lineterminator='\n');w.writeheader();w.writerows(rs)
 if p.suffix=='.gz':
  with p.open('wb') as f:
   with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(buf.getvalue().encode())
 else:p.write_text(buf.getvalue())
def guard():need(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024<1450,'memory cap')
def plan():
 need(not (HERE/'manifest.json').exists(),'write once')
 paths=list(FILES.values())+list(HERE.glob('*.py'))+[HERE/'contract.md']
 dump(HERE/'manifest.json',{'version':'reversal_withdrawal_v1','inputs':{k:str(v.relative_to(ROOT)) for k,v in FILES.items()},'files':{str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}})
 print('Input/source identities bound; no signal/end alignment inspected.')
def bound():
 for p,h in json.loads((HERE/'manifest.json').read_text())['files'].items():need(digest(ROOT/p)==h,'source hash '+p)
 summ=json.loads(FILES['summary'].read_text())
 for p in (FILES['post'],FILES['union']):need(digest(p)==summ['hashes'][p.name],'timing receipt')
 need(digest(FILES['signed'])==json.loads(FILES['readout'].read_text())['rendered_sha256'],'signed rendering')
 rec=json.loads(FILES['panel_receipt'].read_text())
 need(digest(PANEL)==rec['outputs']['rows.parquet']['sha256'],'panel receipt')
 need(json.loads(FILES['checks'].read_text())['pass_all'],'prior checks')
def signal(r,side):
 a=np.asarray(r,float)*(1 if side=='bid' else -1);ok=np.zeros(len(a),bool);events=np.zeros(len(a),bool)
 for k in range(5,len(a)):
  ok[k]=np.isfinite(a[k-5:k+1]).all()
  events[k]=ok[k] and a[k-5]>=1 and a[k]<=-1 and a[k-1]>-1
 return events,ok

def classify(lo,hi,ws,event_seconds,valid,delay,post_hi=None):
 # Need complete context at all possible anchors; integer grid makes earliest ceil and latest floor exact.
 if not all(math.isfinite(v) for v in (lo,hi,ws)) or lo>hi:return 'invalid_window',[]
 first=math.ceil((lo-delay-5000-ws)/1000);last=math.floor((hi-delay-ws)/1000)
 if first<5 or last>=len(valid) or first>last or not bool(np.all(valid[first:last+1])):return 'context_or_boundary_unknown',[]
 if post_hi is not None and not math.isfinite(post_hi):return 'post_bound_unknown',[]
 hits=[];intervals=[]
 for sec in event_seconds:
  t=int(ws+int(sec)*1000)
  if post_hi is not None and t<post_hi:continue
  l=t+delay;h=l+5000
  if h>=lo and l<=hi:hits.append(t);intervals.append((l,h))
 if not hits:return 'no_match',[]
 cursor=lo
 for l,h in sorted(intervals):
  if l>cursor:return 'possible_only',hits
  cursor=max(cursor,h)
  if cursor>=hi:return 'all_times_compatible',hits
 return 'possible_only',hits

def shift_events(events,rng):
 out=np.zeros_like(events)
 for a,b in [(5,60),(60,120),(120,180),(180,240),(240,300)]:
  out[a:b]=np.roll(events[a:b],int(rng.integers(0,b-a)))
 return out

def load():
 cols=['day','mid','side','k','t_ms','btc_r_5s_bp_D0','btc_r_5s_bp_D1000']
 p=pd.read_parquet(PANEL,columns=cols);need(len(p)==11400,'panel membership');need(not p.duplicated(['day','mid','side','k']).any(),'panel duplicates')
 grids={};census=[];events=[]
 for (d,m),g in p.groupby(['day','mid'],sort=True):
  a=g[g.side=='bid'].sort_values('k');b=g[g.side=='ask'].sort_values('k')
  need(len(a)==300 and len(b)==300 and np.array_equal(a.k,np.arange(300)) and np.array_equal(b.k,np.arange(300)),'complete seconds')
  ts=a.t_ms.to_numpy(np.int64);r=a.btc_r_5s_bp_D0.to_numpy(float)
  need(np.array_equal(ts,ts[0]+np.arange(300)*1000),'grid clock')
  need(np.array_equal(r,b.btc_r_5s_bp_D0.to_numpy(float),equal_nan=True),'side shared BTC')
  need(np.array_equal(r[:-1],a.btc_r_5s_bp_D1000.to_numpy(float)[1:],equal_nan=True),'one-second asof parity')
  for side in ('bid','ask'):
   e,v=signal(r,side);grids[(str(d),int(m),side)]={'ws':int(ts[0]),'r':r,'events':e,'valid':v}
   census.append(dict(day=str(d),mid=int(m),side=side,seconds=300,valid_transition_seconds=int(v.sum()),reversals=int(e.sum()),unknown_seconds=int((~v).sum())))
   for k in np.flatnonzero(e):events.append(dict(day=str(d),mid=int(m),side=side,t_ms=int(ts[k]),k=int(k),r5_prior=float(r[k-5]),r5_previous=float(r[k-1]),r5_now=float(r[k])))
 signed=rows(FILES['signed']);post={key(x):x for x in rows(FILES['post'])};un={key(x):x for x in rows(FILES['union'])};need(len(signed)==88 and len(set(map(key,signed)))==88,'signed membership')
 cases=[]
 for x in signed:
  k=key(x);q=post[k];u=un[k]
  group='conditional_withdrawal' if truth(x['conditional_withdrawal_all_views']) else ('exhausted' if x['signed_ending']=='exhausted_by_observed_fills' else 'unresolved')
  g=grids[(k[0],k[1],q['side'])];need(int(q['ws_ms'])==g['ws'],'market join')
  cases.append(dict(day=k[0],mid=k[1],maker=k[2],order_hash=k[3],rank=int(x['rank']),side=q['side'],group=group,post_grade=q['grade'],
   post_hi=number(q['effect_hi_ms']) if truth(q['estimate_enabled']) else math.nan,lo=number(u['union_lo_ms']),hi=number(u['union_hi_ms']),ws=g['ws']))
 need(Counter(x['group'] for x in cases)=={'conditional_withdrawal':40,'exhausted':41,'unresolved':7},'ending cohorts')
 return grids,cases,census,events

def evaluate(cases,grids,shifted=None):
 out=[]
 for c in cases:
  for delay in (0,1000):
   for mode in ('timing_only','post_compatible'):
    for control in ('correct_side','wrong_side'):
     side=c['side'] if control=='correct_side' else ('ask' if c['side']=='bid' else 'bid');k=(c['day'],c['mid'],side);g=grids[k]
     es=np.flatnonzero(shifted[k] if shifted is not None else g['events'])
     st,hits=('unresolved_ending',[]) if c['group']=='unresolved' else classify(c['lo'],c['hi'],c['ws'],es,g['valid'],delay,None if mode=='timing_only' else c['post_hi'])
     out.append(dict(day=c['day'],mid=c['mid'],maker=c['maker'],order_hash=c['order_hash'],rank=c['rank'],side=c['side'],group=c['group'],post_grade=c['post_grade'],
       delay_ms=delay,mode=mode,control=control,status=st,evaluable=st in ('no_match','possible_only','all_times_compatible'),matched=st in ('possible_only','all_times_compatible'),
       lo_ms=int(c['lo']) if math.isfinite(c['lo']) else None,hi_ms=int(c['hi']) if math.isfinite(c['hi']) else None,post_hi_ms=int(c['post_hi']) if math.isfinite(c['post_hi']) else None,
       reversal_times_ms='|'.join(map(str,hits)),private_pull_observed=False))
 return out

def aggregates(rr):
 result=[]
 for group in ('conditional_withdrawal','exhausted','unresolved'):
  for delay in (0,1000):
   for mode in ('timing_only','post_compatible'):
    for control in ('correct_side','wrong_side'):
     part=[r for r in rr if (r['group'],r['delay_ms'],r['mode'],r['control'])==(group,delay,mode,control)]
     for day in ['all']+sorted({r['day'] for r in part}):
      a=[r for r in part if day=='all' or r['day']==day];n=sum(r['evaluable'] for r in a);hits=sum(r['matched'] for r in a)
      result.append(dict(group=group,delay_ms=delay,mode=mode,control=control,day=day,orders=len(a),evaluable=n,matched=hits,rate=hits/n if n else None,
       all_times=sum(r['status']=='all_times_compatible' for r in a),possible_only=sum(r['status']=='possible_only' for r in a),no_match=sum(r['status']=='no_match' for r in a),unknown=len(a)-n))
 return result

def placebo(cases,grids):
 rng=np.random.default_rng(SEED);draws=[]
 # Only 40 primary orders are used in the placebo; no extrapolation from terminal samples to live hazards.
 cc=[c for c in cases if c['group']=='conditional_withdrawal'];days=['all']+sorted({c['day'] for c in cc})
 for draw in range(ND):
  shifted={k:shift_events(grids[k]['events'],rng) for k in sorted(grids)}
  for delay in (0,1000):
   hits=Counter();den=Counter()
   for c in cc:
    k=(c['day'],c['mid'],c['side']);g=grids[k];st,_=classify(c['lo'],c['hi'],c['ws'],np.flatnonzero(shifted[k]),g['valid'],delay,c['post_hi'])
    if st not in ('no_match','possible_only','all_times_compatible'):continue
    for d in ('all',c['day']):den[d]+=1;hits[d]+=st!='no_match'
   for day in days:draws.append(dict(draw=draw,delay_ms=delay,day=day,evaluable=den[day],matched=hits[day],rate=hits[day]/den[day] if den[day] else None))
 return draws

def readings(agg,draws):
 lookup={(r['group'],r['delay_ms'],r['mode'],r['control'],r['day']):r for r in agg};out=[]
 for delay in (0,1000):
  pref=('conditional_withdrawal',delay,'post_compatible','correct_side');obs=lookup[(*pref,'all')];wrong=lookup[('conditional_withdrawal',delay,'post_compatible','wrong_side','all')]
  a=[x['rate'] for x in draws if x['delay_ms']==delay and x['day']=='all' and x['rate'] is not None];n=obs['evaluable']
  need(n==wrong['evaluable'],'wrong-side common denominator')
  positive=0;day_support=0
  for day in ('20261024','20261025','20261026','20261027'):
   row=lookup.get((*pref,day));sample=[x['rate'] for x in draws if x['delay_ms']==delay and x['day']==day and x['rate'] is not None]
   if row and row['evaluable']>=3:day_support+=1
   if row and row['rate'] is not None and sample and row['rate']>float(np.mean(sample)):positive+=1
  sufficient=n>=20 and day_support>=3
  q=float(np.quantile(a,.95)) if a else None
  enriched=sufficient and a and obs['rate']>q and obs['rate']>wrong['rate'] and positive>=3
  out.append(dict(delay_ms=delay,n=n,observed=obs['rate'],wrong_side=wrong['rate'],null_mean=float(np.mean(a)) if a else None,null_p05=float(np.quantile(a,.05)) if a else None,
    null_p95=q,upper_tail_rank=(1+sum(x>=obs['rate'] for x in a))/(1+len(a)) if a else None,draws=len(a),days_supported=day_support,days_above_null_mean=positive,
    sufficient=bool(sufficient),timing_enriched=bool(enriched)))
 label='timing_alignment_enriched' if all(x['timing_enriched'] for x in out) else ('insufficient_coverage' if not all(x['sufficient'] for x in out) else 'hypothesis_not_demonstrated')
 return dict(reading=label,profiles=out,private_action_identified=False,population_cancel_probability_estimated=False)

def run():
 start=time.process_time();resource.setrlimit(resource.RLIMIT_CPU,(18,20));bound();need(not (HERE/'outputs').exists(),'write once outputs')
 grids,cases,cs,ev=load();rr=evaluate(cases,grids);ag=aggregates(rr);draws=placebo(cases,grids);reading=readings(ag,draws);guard()
 out=HERE/'outputs';out.mkdir()
 for name,data in [('signal_census.csv',cs),('reversals.csv',ev),('case_alignment.csv',rr),('summary.csv',ag),('draws.csv.gz',draws)]:write(out/name,data,['day','mid','side','t_ms','k','r5_prior','r5_previous','r5_now'] if name=='reversals.csv' else None)
 result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'as_of_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'population':{'signed_orders':88,'market_side_seconds':11400},**reading,'cpu_seconds':time.process_time()-start,'hashes':{p.name:digest(p) for p in sorted(out.iterdir())}}
 dump(out/'result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
