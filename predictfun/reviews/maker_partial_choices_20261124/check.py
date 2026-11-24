"""Independent scalar checks at the saved source artifacts; no study derivation reused."""
import collections,csv,gzip,hashlib,json,math,copy
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[2]
def read(p):
 with (gzip.open if str(p).endswith('.gz') else open)(p,'rt') as f:return list(csv.DictReader(f))
def number(x):
 try:return float(x)
 except (ValueError,TypeError):return math.nan
def category(lo,hi,kind):
 if not math.isfinite(lo+hi) or lo>hi:return 'unknown'
 if kind=='btc':return 'against' if hi<=-1 else 'with' if lo>=1 else 'flat' if -1<lo<=hi<1 else 'mixed'
 return 'adding' if lo>1e-6 else 'offsetting' if hi< -1e-6 else 'flat' if -1e-6<=lo<=hi<=1e-6 else 'mixed'
def binning(lo,hi,edges,labels):
 if not math.isfinite(lo+hi) or lo>hi:return 'unknown'
 ix=lambda v:sum(v>e for e in edges)
 return labels[ix(lo)] if ix(lo)==ix(hi) else 'mixed'
def eq(a,b):
 if isinstance(b,(int,float)):assert math.isclose(number(a),b,rel_tol=1e-12,abs_tol=1e-12),(a,b)
 else:assert str(a)==str(b),(a,b)
def validate(actions,scores,draws,partial,links):
 root=R/'predictfun/reviews';P=read(root/'maker_trace_20261122/v2/chains/post_fill/orders.csv');ps={r['mk_hash']:r for r in P};ns={r['order_hash']:r for r in read(root/'maker_timing_estimates_20261124/outputs/post_estimates.csv.gz')};inc=collections.defaultdict(list)
 assert len(P)==len(ps)==len(links);assert {r['order_hash'] for r in links}==set(ps)
 for r in links:
  p=ps[r['order_hash']];eq(r['successor_hash'],p['next_hash']);status='no_link' if not p['next_hash'] else 'linked' if p['next_hash'] in ps else 'missing_successor';eq(r['link_status'],status)
 for p in P:
  if p['next_hash'] in ps:inc[p['next_hash']].append(p)
 assert len(actions)==len(inc) and {r['successor_hash'] for r in actions}==set(inc)
 for r in actions:
  child=ps[r['successor_hash']];parent=sorted(inc[r['successor_hash']],key=lambda x:(int(x['first_hi_ms']),x['mk_hash']))[-1];eq(r['predecessor_hash'],parent['mk_hash']);eq(r['incoming_links'],len(inc[r['successor_hash']]))
  for k in ('day','mid','rank'):eq(r[k],child[k]);eq(parent[k],child[k])
  eq(parent['addr'],child['addr']);eq(parent['y_side'],child['y_side']);eq(r['side'],child['y_side']);eq(r['private_pull_identified'],'False')
  sign=1 if child['y_side']=='bid' else -1;delta=sign*(int(child['tick'])-int(parent['tick']));eq(r['old_tick'],int(parent['tick']));eq(r['new_tick'],int(child['tick']));eq(r['signed_price_change'],delta);eq(r['price_action'],'less' if delta<0 else 'more' if delta>0 else 'same')
  a,b=ns[parent['mk_hash']],ns[child['mk_hash']];valid=lambda x:x['grade'] in ['G2_bound_constrained','G4_scenario_only'] and x['estimate_enabled']=='True' and 0<=number(x['effect_lo_ms'])<=number(x['effect_hi_ms'])
  status='refused_post_window' if not valid(a) or not valid(b) else 'post_order_ambiguous' if number(b['effect_lo_ms'])<=number(a['effect_hi_ms']) else 'outside_live_window' if not int(child['ws_ms'])<=number(b['effect_lo_ms'])<=number(b['effect_hi_ms'])<int(child['we_ms']) else 'ok';eq(r['status'],status)
  sensitive=child['d_sensitive_fields'].split('|');btc=category(number(child['a_5s_min']),number(child['a_5s_max']),'btc')
  if int(child['n_btc_missing']):btc='unknown'
  if 'btc_5s' in sensitive:btc='clock_sensitive'
  lo,hi=sorted([sign*number(child['pos_min']),sign*number(child['pos_max'])]);inv=category(lo,hi,'inv')
  if 'inv' in sensitive:inv='clock_sensitive'
  if inv not in ['mixed','unknown','clock_sensitive'] and child['inv_set']!=inv:inv='unknown'
  eq(r['btc'],btc);eq(r['inventory'],inv);eq(r['old_touch'],parent['touch_set'] if int(parent['n_unknown_instants'])==0 else 'unknown')
  eq(r['phase'],binning(number(child['tau_min_s']),number(child['tau_max_s']),[30,60],['last30','last60','earlier']));eq(r['inventory_abs_band'],binning(0 if lo<=0<=hi else min(abs(lo),abs(hi)),max(abs(lo),abs(hi)),[50,200,500],['<=50','50_200','200_500','>500']))
  ds=[number(x[k]) for x in [parent,child] for k in ['dist_min','dist_max']];complete=all(math.isfinite(d) for d in ds) and ds[0]<=ds[1] and ds[2]<=ds[3] and int(parent['n_unknown_instants'])==int(child['n_unknown_instants'])==0
  if complete:
   x,y=ds[0]-ds[3],ds[1]-ds[2];eq(r['offset_change_lo'],x);eq(r['offset_change_hi'],y);eq(r['offset_action'],'retreat' if y<0 else 'advance' if x>0 else 'unchanged' if x==y==0 else 'ambiguous')
  else:eq(r['offset_action'],'unknown');eq(r['offset_change_lo'],'');eq(r['offset_change_hi'],'')
  for k,s in [('old_queue_class','queue_class'),('old_queue_ahead_lo','ahead_lo_fifo'),('old_queue_ahead_hi','ahead_hi_fifo'),('old_post_width_ms','window_width_ms')]:eq(r[k],parent[s])
  eq(r['new_post_width_ms'],child['window_width_ms'])
 assert len(scores)==8 and len(draws)==4000
 rng=np.random.default_rng(20261124)
 for score in scores:
  hyp,target=score['hypothesis'],score['outcome'];groups=collections.defaultdict(list)
  for r in actions:
   if r['status']!='ok' or r['old_touch']=='unknown' or r['btc'] not in ['against','with','flat'] or r['inventory'] not in ['adding','offsetting','flat']:continue
   if target=='offset_retreat' and r['offset_action']=='unknown':continue
   key=tuple(r[k] for k in ['day','mid','rank','side','old_touch','phase','inventory_abs_band'])
   if hyp=='btc_against':
    if r['btc'] not in ['against','flat']:continue
    exposed=r['btc']=='against';key+=(r['inventory'],)
   else:
    if r['inventory'] not in ['adding','offsetting']:continue
    exposed=r['inventory']=='adding';key+=(r['btc'],)
   y=int(r['offset_action']=='retreat') if target=='offset_retreat' else int(r['price_action']==target);groups[key].append((exposed,y))
  num=den=0.;sims=np.zeros(500);days=collections.defaultdict(lambda:[0.,0.]);markets=collections.defaultdict(float);n1=n0=ng=0
  for key,items in sorted(groups.items()):
   a=[y for e,y in items if e];b=[y for e,y in items if not e]
   if not a or not b:continue
   n,m=len(a),len(b);N=n+m;Y=sum(a)+sum(b);w=1/(1/n+1/m);value=w*(sum(a)/n-sum(b)/m);den+=w;num+=value;days[key[0]][0]+=value;days[key[0]][1]+=w;markets[key[:2]]+=value;ng+=1;n1+=n;n0+=m
   selected=rng.hypergeometric(Y,N-Y,n,size=500)
   for j,x in enumerate(selected):sims[j]+=w*(x/n-(Y-x)/m)
  rd=num/den if den else 0.;sims=sims/den if den else sims;p=(1+sum(abs(v)>=abs(rd)-1e-15 for v in sims))/501 if den else 1.;dp={d:100*v[0]/v[1] for d,v in days.items()};positive=sum(v>0 for v in dp.values());negative=sum(v<0 for v in dp.values());support=n1>=20 and n0>=20 and len(markets)>=5;ok=support and p<=.05/8 and (positive>=3 if rd>0 else negative>=3 if rd<0 else False)
  for k,v in dict(exposed=n1,reference=n0,common_strata=ng,common_markets=len(markets),matched_difference_pp=rd*100,raw_two_sided_p=p,positive_days=positive,negative_days=negative,support='supported_descriptive' if support else 'insufficient',exploratory_pattern=bool(ok),max_market_absolute_share=max(map(abs,markets.values()),default=0)/sum(map(abs,markets.values())) if any(markets.values()) else 0.).items():eq(score[k],v)
  actual=json.loads(score['day_differences_pp']);assert set(actual)==set(dp)
  for k,v in dp.items():eq(actual[k],v)
  ds=[r for r in draws if (r['hypothesis'],r['outcome'])==(hyp,target)];assert len(ds)==500
  for j,r in enumerate(ds):eq(r['draw'],j);eq(r['difference_pp'],100*sims[j])
 pop={r['order_hash']:r for r in read(root/'maker_action_population_20261124/outputs/orders.csv') if r['budget_status']=='positive_remainder_not_a_pull'};ctx={(r['order_hash'],r['stage'],r['lag_ms']):r for r in read(root/'maker_action_population_20261124/outputs/contexts.csv')};ordered={r['order_hash']:r for r in read(root/'maker_action_population_20261124/outputs/ordered_chains.csv')};assert len(partial)==len(pop)==56 and {r['order_hash'] for r in partial}==set(pop)
 for r in partial:
  p=pop[r['order_hash']];n=ps.get(p['successor_hash']);old=ps[r['order_hash']];delta=(1 if old['y_side']=='bid' else -1)*(int(n['tick'])-int(old['tick'])) if n else None
  mapping={'day':'day','mid':'mid','rank':'rank','side':'side','old_tick':'tick','original_shares':'original_shares','filled_shares':'recorded_shares','remaining_budget_wei':'remaining_upper_wei','ending':'end_kind','successor_relation':'successor_relation','signed_next_size':'successor_original_shares'}
  for k,v in mapping.items():eq(r[k],p[v])
  eq(r['price_action'],'unknown' if delta is None else 'less' if delta<0 else 'more' if delta>0 else 'same');eq(r['new_tick'],n['tick'] if n else '');eq(r['private_pull_identified'],'False')
  for suffix,lag in [('D0','0'),('lag','1000')]:
   c=ctx[(r['order_hash'],'successor',lag)];eq(r['btc_'+suffix],c['btc_side']);eq(r['inventory_'+suffix],c['inventory_side'])
  for k in ['gap_min_s','gap_max_s']:eq(r[k],ordered.get(r['order_hash'],{}).get(k,''))
 return True
def run():
 for path,digest in json.loads((H/'manifest.json').read_text())['files'].items():assert hashlib.sha256((R/path).read_bytes()).hexdigest()==digest,path
 a=read(H/'actions.csv.gz');s=read(H/'matched.csv');d=read(H/'draws.csv');p=read(H/'partial_cases.csv');l=read(H/'links.csv');assert validate(a,s,d,p,l)
 cases=[('price',0,'new_tick','101'),('pull',0,'private_pull_identified','True'),('btc',0,'btc','invented'),('offset',0,'offset_action','invented'),('matched',1,'matched_difference_pp','999'),('null',2,'difference_pp','999'),('budget',3,'remaining_budget_wei','-1'),('membership',4,'successor_hash','invented')];caught=[]
 # Validate at the exact named family; source and membership checks already passed.
 for name,which,key,value in cases:
  arrays=[a,s,d,p,l];old=arrays[which][0][key];arrays[which][0][key]=value
  try:validate(*arrays)
  except (AssertionError,KeyError):caught.append(name)
  else:raise AssertionError('escaped '+name)
  finally:arrays[which][0][key]=old
 result=json.loads((H/'result.json').read_text());eq(result['orders'],len(l));eq(result['unique_successor_actions'],len(a));eq(result['ordered_valid_actions'],sum(r['status']=='ok' for r in a));eq(result['partial_cases'],len(p));eq(result['matched_tests'],len(s));eq(result['exploratory_patterns'],sum(r['exploratory_pattern']=='True' for r in s))
 for name,values in [('physical_status',[r['status'] for r in a]),('price_actions',[r['price_action'] for r in a]),('offset_actions',[r['offset_action'] for r in a if r['status']=='ok']),('partial_sequence',[r['successor_relation'] for r in p]),('partial_ordered_prices',[r['price_action'] for r in p if r['successor_relation']=='sequential_after_under_bounds'])]:assert result[name]==dict(collections.Counter(values))
 (H/'checks.json').write_text(json.dumps(dict(positive=True,falsifiers=caught,all_true=True,limitations='Recounts accepted source artifacts, not raw tape or hidden cancellations.'),indent=2)+'\n');print('independent source/action/rate/null/partial checks pass; 8 corruptions refused')
if __name__=='__main__':run()
