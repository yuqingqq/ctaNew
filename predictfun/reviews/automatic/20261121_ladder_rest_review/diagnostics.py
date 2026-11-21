"""Saved-artifact diagnostics only; run via codex run_light. No tape or replay."""
from pathlib import Path
import gzip,json,math,hashlib,datetime
import pandas as pd
import numpy as np
R=Path('/home/yuqing/ctaNew')
D=R/'data/predict_fun/research/ladder_rest_v1_20261121'
P=Path('/tmp/pf-lr-review-869fa85f/predictfun/reviews/ladder_rest_20261121')
O=Path(__file__).resolve().parent
ev=json.loads((O/'evidence.json').read_text())
man=json.loads((P/'manifest.json').read_text())
parent_path=Path('/tmp/pf-lr-review-869fa85f')/man['parent']['rel']
assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==man['parent']['sha256']
parent=json.loads(parent_path.read_text())
candidates={(f['day'],int(f['m'])) for s in parent['sets'].values() for f in s['forks']}
cells={};registry_ok={};opens={}
for day in man['days']:
 p=R/'data/predict_fun/research/step1'/day/'registry.parquet'
 registry_ok[day]=hashlib.sha256(p.read_bytes()).hexdigest()==man['registry'][day]['registry_sha256']
 assert registry_ok[day]
 reg=pd.read_parquet(p,columns=['mid','ws','decimal_precision'])
 for x in reg.itertuples():
  if (day,int(x.mid)) in candidates:
   hour=datetime.datetime.fromtimestamp(x.ws,datetime.timezone.utc).hour
   cells.setdefault((day,hour),[]).append(int(x.mid));opens[(day,int(x.mid))]=(x.ws*1000.,int(x.decimal_precision))
assert sum(map(len,cells.values()))==len(candidates)
assert {(c['day'],c['hour']):sorted(c['candidate_ids']) for c in man['cells']}=={k:sorted(v) for k,v in cells.items()}
def rank(day,m):return hashlib.sha256(('large_rest_inventory_full_market_v1|'+json.dumps(dict(day=day,m=m),sort_keys=True,separators=(',',':'))).encode()).hexdigest()
expected={(day,min(ms,key=lambda m:rank(day,m))) for (day,hour),ms in cells.items()}
assert {(x['day'],x['m']) for x in man['selected']}==expected
assert all(opens[(x['day'],x['m'])]==(x['ws'],x['decimal_precision']) for x in man['selected'])
assert all(man['pilot'][day]==sorted(m for d,m in expected if d==day)[:2] for day in man['days'])
rows=[]
for day in ('20261024','20261025','20261026','20261027'):
 rows += [json.loads(x) for x in gzip.decompress((D/'days'/f'{day}.jsonl.gz').read_bytes()).decode().splitlines()]
comp=[r for r in rows if r['status']=='complete']
G=pd.read_csv(D/'measure/generations.csv',float_precision='round_trip')
F=pd.read_csv(D/'measure/fills.csv',float_precision='round_trip')
S=pd.read_csv(D/'measure/summary.csv',float_precision='round_trip',dtype={'day':str,'rung':str})
total=S[(S.day=='all')&(S.side=='both')&(S.rung=='all')].iloc[0]
early=[];event_sweeps=0;time_sweeps=0;false_time_sweep_rows=0;zero_recovery_exits=0;reversal_exits=0
markout_by_request={}
for r in comp:
 L=r['ledger'];ods={o['gid']:o for o in L['orders']}
 for o in L['orders']:
  if o['req'] is not None and not o['refused'] and o['req']<=math.ceil(o['act']/50.)*50.:
   early.append((r['day'],r['m'],o['gid'],o['why'],o['req']-o['submit']))
 bt={};bi={}
 for f in L['fills']:
  bt.setdefault((f[2],f[0]),set()).add(f[1]);bi.setdefault(f[5],set()).add(f[1])
 for f,mid in zip(L['fills'],L['fill_mids']):
  time_sweeps+=len(bt[(f[2],f[0])])>=2;event_sweeps+=len(bi[f[5]])>=2
  false_time_sweep_rows+=len(bt[(f[2],f[0])])>=2 and len(bi[f[5]])<2
  o=ods[f[1]];cat='before_request' if o['req'] is None or f[0]<o['req'] else 'cancel_pending'
  a=markout_by_request.setdefault(cat,dict(fills=0,q=0.,edge=0.,mo50=0.,mo1000=0.,n50=0,n1000=0))
  a['fills']+=1;a['q']+=f[3];sg=1 if f[2]==0 else -1
  if mid[0] is not None:a['edge']+=sg*f[3]*(mid[0]-f[4])
  for k,j in [('50',2),('1000',6)]:
   if mid[j] is not None:a['mo'+k]+=sg*f[3]*(mid[j]-f[4]);a['n'+k]+=1
 for t,i,old,new in L['mode_log']:
  if old!='UNRESTRICTED' and new=='UNRESTRICTED':zero_recovery_exits+=1
  elif old!='UNRESTRICTED' and new!=old:reversal_exits+=1
coverage=[]
for day in ('20261024','20261025','20261026','20261027'):
 coverage += [dict(x,day=day) for x in json.loads((P/'logs'/f'coverage_diag_{day}.json').read_text())['markets']]
cov=pd.DataFrame(coverage)
assert sorted((x['day'],x['m'],'eligible' if x['status']=='complete' else x['status'].removeprefix('incomplete:')) for x in rows)==sorted((x['day'],x['m'],x['status']) for x in coverage)
out={'target':ev['target'],'coverage_from_saved_diagnostic':cov.groupby('status').agg(n=('m','size'),last_mark_lag_median_s=('end_minus_last_mark_s','median'),first_gap_median_s=('first_gap_rel_s_median','median') if 'first_gap_rel_s_median' in cov else ('first_gap_rel_s','median')).to_dict('index'),
 'first_review':{'reported_policy_first_review_pulls':int(total.first_review_pulls),'reported_reviewed':int(total.reviewed),'accepted_orders_requested_by_first_live_tick':len(early),'by_reason':pd.Series([x[3] for x in early]).value_counts().to_dict()},
 'sweeps':{'time_grouped_fill_rows':time_sweeps,'trade_grouped_fill_rows':event_sweeps,'time_grouped_but_not_same_trade_rows':false_time_sweep_rows},
 'inventory_modes':{'recovery_to_band_exits':zero_recovery_exits,'direct_reversal_exits':reversal_exits},
 'fill_timing':markout_by_request,
 'mids_age_ms':{'p50':float(F.mid0_age.median()),'p99':float(F.mid0_age.quantile(.99)),'max':float(F.mid0_age.max())},
 'feature_snapshot_counts':{},
 'inventory_event_vs_grid':{'markets_with_missed_peak':sum(x['peak_exact']>x['peak_grid']+1e-9 for x in ev['durations_inventory']),
 'max_peak_understatement':max(x['peak_exact']-x['peak_grid'] for x in ev['durations_inventory']),
 'max_mean_abs_difference':max(abs(x['mean_abs_exact']-x['mean_abs_grid']) for x in ev['durations_inventory']),
 'opening_wait_max':max(x['opening_wait'] for x in ev['durations_inventory'])},
 'closed_round_trip_control':{'bid_price':50.,'ask_price':51.,'size':100.,'mark':80.,'bid_END_value':3000.,'ask_END_value':-2900.,'joint_value':100.,'severe_ask_despite_profitable_flat_round_trip':True}}
for day in ('20261024','20261025','20261026','20261027'):
 T=pd.read_parquet(D/'days'/f'{day}_features.parquet')
 out['feature_snapshot_counts'][day]=dict(rows=len(T),status=T.status.value_counts().to_dict(),kinds=T.kind.value_counts().to_dict(),all_fair_edge_missing=bool(T.fair_edge.isna().all()))
out['markout_headline']={c:float(total[c]) for c in ['edge','mo50','mo50_q','mo50_fills','mo1000','mo1000_q','mo1000_fills','common_edge','common_mo50','common_mo1000','common_q','common_fills','v_severe','top5_market_loss_share']}
out['population_check']={'registry_hashes':registry_ok,'unique_parent_markets':len(candidates),'cells':len(cells),'selected':len(expected),'minimum_hash_and_pilot_reproduced':True}
c=out['closed_round_trip_control']
c['bid_END_value']=c['size']*(c['mark']-c['bid_price']);c['ask_END_value']=c['size']*(c['ask_price']-c['mark'])
c['joint_value']=c['bid_END_value']+c['ask_END_value']
c['severe_ask_despite_profitable_flat_round_trip']=c['ask_END_value'] < -30. and c['joint_value']>0
assert c['joint_value']==100. and c['severe_ask_despite_profitable_flat_round_trip']
(O/'diagnostics.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps(out,indent=2,default=str))
