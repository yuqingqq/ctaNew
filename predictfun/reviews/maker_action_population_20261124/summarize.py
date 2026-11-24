"""Post-run descriptive presentation; no new hypothesis, fit or selected sample."""
import collections,csv,json,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[2]
def read(p):return list(csv.DictReader(p.open()))
def flag(v):return v=='True'
rr=read(H/'outputs/orders.csv');cx=read(H/'outputs/contexts.csv');ix={(x['order_hash'],x['stage'],x['lag_ms']):x for x in cx};census=json.loads((H/'census.json').read_text());checks=json.loads((H/'checks.json').read_text());assert checks['pass_all'];ordered=[r for r in rr if r['successor_relation']=='sequential_after_under_bounds'];groups=[];conditions=[]
for scope in ('all','old88','new'):
 ss=[r for r in rr if scope=='all' or flag(r['old88'])==(scope=='old88')];groups.append(dict(scope=scope,n=len(ss),budget=dict(collections.Counter(r['budget_status'] for r in ss)),successors=dict(collections.Counter(r['successor_relation'] for r in ss)),ordered_kinds=dict(collections.Counter(r['end_kind']+' → '+r['successor_kind'] for r in ss if r['successor_relation']=='sequential_after_under_bounds'))))
 for stage in ('post','ending','successor'):
  for lag in ('0','1000'):
   for action in ('all','ordered_refill','ordered_reprice'):
    pop=ss if action=='all' else [r for r in ss if r['successor_relation']=='sequential_after_under_bounds' and r['successor_kind']==action.split('_')[1]]
    rows=[ix[(r['order_hash'],stage,lag)] for r in pop];conditions.append(dict(scope=scope,stage=stage,lag_ms=int(lag),action=action,orders=len(rows),covered=sum(r['coverage']=='covered' for r in rows),btc=dict(collections.Counter(r['btc_side'] for r in rows)),inventory=dict(collections.Counter(r['inventory_side'] for r in rows)),spread=dict(collections.Counter(r['spread'] for r in rows)),depth=dict(collections.Counter(r['own_depth'] for r in rows)),time_left=dict(collections.Counter(r['time_left'] for r in rows))))
table=[]
for r in ordered:
 a,b=ix[(r['order_hash'],'successor','0')],ix[(r['order_hash'],'successor','1000')];row={k:r[k] for k in ('order_hash','old88','day','mid','rank','side','original_shares','recorded_shares','end_kind','successor_kind','successor_hash','successor_original_shares')};row.update(ending_lo_s=(int(r['end_lo_ms'])-int(r['ws_ms']))/1000,ending_hi_s=(int(r['end_hi_ms'])-int(r['ws_ms']))/1000,next_post_lo_s=(int(r['successor_post_lo_ms'])-int(r['ws_ms']))/1000,next_post_hi_s=(int(r['successor_post_hi_ms'])-int(r['ws_ms']))/1000,gap_min_s=(int(r['successor_post_lo_ms'])-int(r['end_hi_ms']))/1000,gap_max_s=(int(r['successor_post_hi_ms'])-int(r['end_lo_ms']))/1000,btc_D0=a['btc_side'],btc_D1000=b['btc_side'],inventory_D0=a['inventory_side'],inventory_D1000=b['inventory_side'],spread_D0=a['spread'],depth_D0=a['own_depth'],trigger_identified=False);table.append(row)
with (H/'outputs/ordered_chains.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(table[0]),lineterminator='\n');w.writeheader();w.writerows(table)
poststats=[]
for stage in ('post','successor'):
 population=rr if stage=='post' else ordered
 for field in ('btc_side','inventory_side','spread','own_depth'):
  counts=collections.Counter()
  for r in population:
   a,b=ix[(r['order_hash'],stage,'0')][field],ix[(r['order_hash'],stage,'1000')][field]
   lab='clock_sensitive' if a!=b else ('unknown' if a=='unknown' else ('timing_ambiguous' if '|' in a else a));counts[lab]+=1
  poststats.append(dict(stage=stage,field=field,n=len(population),both_clock_counts=dict(counts)))
summary=dict(groups=groups,conditions=conditions,both_clocks=poststats,ordered_days=dict(collections.Counter(r['day'] for r in ordered)),ordered_markets=len({(r['day'],r['mid']) for r in ordered}),new_ordered_markets=len({(r['day'],r['mid']) for r in ordered if not flag(r['old88'])}),next_test_gate='no_observed_nonposting_controls',private_cancel_trigger_identified=False)
(H/'readout.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(dict(groups=groups,both_clocks=poststats,ordered_days=summary['ordered_days'],ordered_markets=summary['ordered_markets']),indent=2))
