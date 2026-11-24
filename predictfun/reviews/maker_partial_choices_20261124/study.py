"""Action identities first, then predeclared matched observed-action comparisons."""
import csv,gzip,json,hashlib,math,sys,subprocess,time,collections,importlib.util
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[2];C=R/'predictfun/reviews/maker_trace_20261122/v2/chains';A=R/'predictfun/reviews/maker_action_population_20261124';T=R/'predictfun/reviews/maker_timing_estimates_20261124'
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def write(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 with (gzip.open if str(p).endswith('.gz') else open)(p,'wt') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def num(v):
 try:return float(v)
 except (ValueError,TypeError):return math.nan
def bcat(lo,hi):
 if not math.isfinite(lo+hi) or lo>hi:return 'unknown'
 return 'against' if hi<=-1 else ('with' if lo>=1 else ('flat' if lo>-1 and hi<1 else 'mixed'))
def icat(lo,hi):
 if not math.isfinite(lo+hi) or lo>hi:return 'unknown'
 return 'adding' if lo>1e-6 else ('offsetting' if hi< -1e-6 else ('flat' if lo>=-1e-6 and hi<=1e-6 else 'mixed'))
def band(lo,hi,edges,labels):
 if not math.isfinite(lo+hi) or lo>hi:return 'unknown'
 f=lambda x:next((labels[i] for i,e in enumerate(edges) if x<=e),labels[-1])
 return f(lo) if f(lo)==f(hi) else 'mixed'
def bounds(p):
 a=num(p['effect_lo_ms']);b=num(p['effect_hi_ms']);return p['grade'] in ('G2_bound_constrained','G4_scenario_only') and p['estimate_enabled']=='True' and 0<=a<=b
def derive():
 ps={r['mk_hash']:r for r in read(C/'post_fill/orders.csv')};ns={r['order_hash']:r for r in read(T/'outputs/post_estimates.csv.gz')};incoming=collections.defaultdict(list);links=[]
 for h,p in sorted(ps.items()):
  n=ps.get(p['next_hash']);status='no_link' if not p['next_hash'] else ('missing_successor' if n is None else 'linked')
  if n:
   assert tuple(p[k] for k in ('day','mid','addr','y_side'))==tuple(n[k] for k in ('day','mid','addr','y_side')),'identity';incoming[n['mk_hash']].append(p)
  links.append(dict(order_hash=h,successor_hash=p['next_hash'],link_status=status))
 out=[]
 for nh,parents in sorted(incoming.items()):
  p=max(parents,key=lambda r:(int(r['first_hi_ms']),r['mk_hash']));n=ps[nh];a=ns[p['mk_hash']];z=ns[nh];sg=1 if n['y_side']=='bid' else -1;d=sg*(int(n['tick'])-int(p['tick']));status='ok'
  if not bounds(a) or not bounds(z):status='refused_post_window'
  elif num(z['effect_lo_ms'])<=num(a['effect_hi_ms']):status='post_order_ambiguous'
  elif not int(n['ws_ms'])<=num(z['effect_lo_ms'])<=num(z['effect_hi_ms'])<int(n['we_ms']):status='outside_live_window'
  btc=bcat(num(n['a_5s_min']),num(n['a_5s_max']));sensitive=set(n['d_sensitive_fields'].split('|'));q0=sg*num(n['pos_min']);q1=sg*num(n['pos_max']);lo,hi=sorted((q0,q1));inv=icat(lo,hi)
  if int(n['n_btc_missing'])>0:btc='unknown'
  if 'btc_5s' in sensitive:btc='clock_sensitive'
  if 'inv' in sensitive:inv='clock_sensitive'
  # No unknown point is permitted to turn a range into certainty.
  if n['inv_set']!=inv and inv not in ('mixed','unknown','clock_sensitive'):inv='unknown'
  amin=0 if lo<=0<=hi else min(abs(lo),abs(hi));amax=max(abs(lo),abs(hi));ib=band(amin,amax,[50,200,500],['<=50','50_200','200_500','>500']);phase=band(num(n['tau_min_s']),num(n['tau_max_s']),[30,60],['last30','last60','earlier'])
  dist=[num(p['dist_min']),num(p['dist_max']),num(n['dist_min']),num(n['dist_max'])];complete=int(p['n_unknown_instants'])==0 and int(n['n_unknown_instants'])==0 and all(math.isfinite(x) for x in dist) and dist[0]<=dist[1] and dist[2]<=dist[3];olo=dist[0]-dist[3] if complete else '';ohi=dist[1]-dist[2] if complete else '';offset='unknown' if not complete else ('retreat' if ohi<0 else ('advance' if olo>0 else ('unchanged' if olo==ohi==0 else 'ambiguous')))
  out.append(dict(successor_hash=nh,predecessor_hash=p['mk_hash'],incoming_links=len(parents),day=n['day'],mid=n['mid'],rank=n['rank'],side=n['y_side'],old_tick=int(p['tick']),new_tick=int(n['tick']),signed_price_change=d,price_action='less' if d<0 else ('more' if d>0 else 'same'),status=status,old_touch=p['touch_set'] if int(p['n_unknown_instants'])==0 else 'unknown',btc=btc,inventory=inv,inventory_abs_band=ib,phase=phase,offset_change_lo=olo,offset_change_hi=ohi,offset_action=offset,old_queue_class=p['queue_class'],old_queue_ahead_lo=p['ahead_lo_fifo'],old_queue_ahead_hi=p['ahead_hi_fifo'],old_post_width_ms=p['window_width_ms'],new_post_width_ms=n['window_width_ms'],private_pull_identified=False))
 return ps,out,links
def group(rows,hyp,outcome):
 gs=collections.defaultdict(lambda:[[],[]]);used=[]
 for r in rows:
  if r['status']!='ok' or r['old_touch']=='unknown' or r['btc'] not in ('against','with','flat') or r['inventory'] not in ('adding','offsetting','flat'):continue
  if outcome=='offset_retreat' and r['offset_action']=='unknown':continue
  if hyp=='btc_against':
   if r['btc'] not in ('against','flat'):continue
   exposed=r['btc']=='against';other=r['inventory']
  else:
   if r['inventory'] not in ('adding','offsetting'):continue
   exposed=r['inventory']=='adding';other=r['btc']
  y=int(r['offset_action']=='retreat') if outcome=='offset_retreat' else int(r['price_action']==outcome)
  key=(r['day'],r['mid'],r['rank'],r['side'],r['old_touch'],r['phase'],r['inventory_abs_band'],other);gs[key][int(not exposed)].append(y)
 return {k:v for k,v in gs.items() if all(v)}
def stats(rows):
 result=[];draw=[];rng=np.random.default_rng(20261124)
 for hyp in ('btc_against','inventory_adding'):
  for target in ('less','same','more','offset_retreat'):
   g=group(rows,hyp,target);total=0.;obs=0.;null=np.zeros(500);byday=collections.defaultdict(lambda:[0.,0.]);bymkt=collections.defaultdict(float);n1=n0=0
   for k,(a,b) in sorted(g.items()):
    n=len(a);m=len(b);N=n+m;Y=sum(a)+sum(b);w=n*m/N;diff=sum(a)/n-sum(b)/m;x=rng.hypergeometric(Y,N-Y,n,size=500);null+=w*(x/n-(Y-x)/m);obs+=w*diff;total+=w;byday[k[0]][0]+=w*diff;byday[k[0]][1]+=w;bymkt[k[:2]]+=w*diff;n1+=n;n0+=m
   rd=obs/total if total else 0.;null=null/total if total else null;p=(1+int((np.abs(null)>=abs(rd)-1e-15).sum()))/501 if total else 1.;days={d:v[0]/v[1] for d,v in sorted(byday.items())};positive=sum(v>0 for v in days.values());negative=sum(v<0 for v in days.values());support=n1>=20 and n0>=20 and len(bymkt)>=5;consistent=positive>=3 if rd>0 else negative>=3 if rd<0 else False;pattern=support and p<=.05/8 and consistent;den=sum(abs(v) for v in bymkt.values());concentration=max(map(abs,bymkt.values()),default=0)/den if den else 0.
   result.append(dict(hypothesis=hyp,outcome=target,exposed=n1,reference=n0,common_strata=len(g),common_markets=len(bymkt),matched_difference_pp=100*rd,raw_two_sided_p=p,day_differences_pp=json.dumps({d:100*v for d,v in days.items()},sort_keys=True),positive_days=positive,negative_days=negative,max_market_absolute_share=concentration,support='supported_descriptive' if support else 'insufficient',exploratory_pattern=pattern))
   for j,x in enumerate(null):draw.append(dict(hypothesis=hyp,outcome=target,draw=j,difference_pp=100*float(x)))
 return result,draw
def run():
 for p,h in json.loads((H/'manifest.json').read_text())['files'].items():assert sha(R/p)==h,p
 ps,rows,links=derive();assert not (H/'actions.csv.gz').exists();write(H/'actions.csv.gz',rows);write(H/'links.csv',links);scores,draw=stats(rows);write(H/'matched.csv',scores);write(H/'draws.csv',draw)
 partial=[];pop=read(A/'outputs/orders.csv');contexts={(r['order_hash'],r['stage'],r['lag_ms']):r for r in read(A/'outputs/contexts.csv')};ordered={r['order_hash']:r for r in read(A/'outputs/ordered_chains.csv')}
 for r in pop:
  if r['budget_status']!='positive_remainder_not_a_pull':continue
  n=ps.get(r['successor_hash']);p=ps[r['order_hash']];sg=1 if p['y_side']=='bid' else -1;d=sg*(int(n['tick'])-int(p['tick'])) if n else None;cx=contexts[(r['order_hash'],'successor','0')];lag=contexts[(r['order_hash'],'successor','1000')];o=ordered.get(r['order_hash'],{})
  partial.append(dict(order_hash=r['order_hash'],day=r['day'],mid=r['mid'],rank=r['rank'],side=r['side'],old_tick=r['tick'],new_tick=n['tick'] if n else '',original_shares=r['original_shares'],filled_shares=r['recorded_shares'],remaining_budget_wei=r['remaining_upper_wei'],ending=r['end_kind'],successor_relation=r['successor_relation'],price_action='unknown' if d is None else ('less' if d<0 else ('more' if d>0 else 'same')),gap_min_s=o.get('gap_min_s',''),gap_max_s=o.get('gap_max_s',''),btc_D0=cx['btc_side'],btc_lag=lag['btc_side'],inventory_D0=cx['inventory_side'],inventory_lag=lag['inventory_side'],signed_next_size=r['successor_original_shares'],private_pull_identified=False))
 assert len(partial)==56;write(H/'partial_cases.csv',partial);summary=dict(as_of_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),orders=len(ps),unique_successor_actions=len(rows),physical_status=dict(collections.Counter(r['status'] for r in rows)),price_actions=dict(collections.Counter(r['price_action'] for r in rows)),ordered_valid_actions=sum(r['status']=='ok' for r in rows),offset_actions=dict(collections.Counter(r['offset_action'] for r in rows if r['status']=='ok')),partial_cases=len(partial),partial_sequence=dict(collections.Counter(r['successor_relation'] for r in partial)),partial_ordered_prices=dict(collections.Counter(r['price_action'] for r in partial if r['successor_relation']=='sequential_after_under_bounds')),matched_tests=len(scores),exploratory_patterns=sum(r['exploratory_pattern'] for r in scores));dump(H/'result.json',summary);print(json.dumps(summary,indent=2));print(json.dumps(scores,indent=2))
def freeze():
 files=[H/'study.py',H/'check.py',H/'contract.md',C/'post_fill/orders.csv',C/'correction_v1/post_regrade_orders.csv',T/'outputs/post_estimates.csv.gz',A/'outputs/orders.csv',A/'outputs/contexts.csv',A/'outputs/ordered_chains.csv',A/'checks.json'];assert not (H/'manifest.json').exists();spec=importlib.util.spec_from_file_location('cpu',R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);task=sum(x['cpu_s'] for x in m.ledger())/3600+m.EXTRA_H;assert task+.0014+.05<=2 and m.LINE_BEFORE+task+.0014+.05<=5.3
 dump(H/'manifest.json',dict(files={str(p.relative_to(R)):sha(p) for p in files},task_before=task,packet_cap_coreh=.0014,candidate_models=0,hypothesis_outcomes=8));print('frozen inputs, resource gate passed')
def tests():
 assert bcat(-3,-1)=='against' and bcat(-.5,.5)=='flat' and bcat(-2,2)=='mixed';assert icat(-4,-2)=='offsetting' and icat(-4,2)=='mixed';assert band(31,40,[30,60],['last30','last60','earlier'])=='last60';assert band(29,31,[30,60],['last30','last60','earlier'])=='mixed';assert bcat(float('nan'),1)=='unknown';print('8 condition/boundary/unknown controls pass')
 fixture=[dict(status='ok',old_touch='at',btc='against' if e else 'flat',inventory='adding',inventory_abs_band='<=50',phase='earlier',price_action='less' if e else 'more',offset_action='retreat' if e else 'advance',day=str(i%4),mid=str(i),rank='1',side='bid') for i in range(8) for e in (0,1) for _ in range(8)]
 scores,_=stats(fixture);assert scores[0]['exploratory_pattern'] and scores[0]['matched_difference_pp']==100
 for row in fixture:row['price_action']='same';row['offset_action']='unchanged'
 scores,_=stats(fixture);assert not any(r['exploratory_pattern'] for r in scores);print('matched association positive and no-association controls pass')
if __name__=='__main__':
 if '--freeze' in sys.argv:tests();freeze()
 elif '--test' in sys.argv:tests()
 else:run()
