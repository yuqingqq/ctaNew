"""Raw-leg reconciliation, independent timing joins, and corruptions."""
import collections,copy,csv,gzip,json
from pathlib import Path
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent;R=H.parents[2];MT=R/'data/predict_fun/research/maker_trace_v2_20261123'
d=json.loads((H/'maker3_quotes.json').read_text());ws=1792977000000;maker='0x57fbc891e89184da071ae9d97e745531cd7f718a'
raw=pq.read_table(MT/'legs_20261026.parquet',filters=[('mid','=',3789484)]).to_pylist()
raw=[r for r in raw if r['mk_signer'].lower()==maker and 0<=r['settle_ms']-ws<60000]
with gzip.open(R/'predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz','rt') as f:p={r['order_hash']:r for r in csv.DictReader(f)}
def validate(d):
 assert len(d['rows'])==len({r['hash'] for r in d['rows']})
 assert {r['hash'] for r in d['rows']}=={r['mk_hash'] for r in raw}
 assert sum(r['n_fills_first60'] for r in d['rows'])==len(raw)
 for r in d['rows']:
  ls=[x for x in raw if x['mk_hash']==r['hash']]
  assert len(ls)==r['n_fills_first60']
  assert abs(sum(x['shares'] for x in ls)-r['filled_first60_sh'])<=len(ls)*.51e-6
  assert all(x['y_side']==r['side'] and abs(x['y_price']*100-r['price_c'])<1e-8 and x['status']=='ok' for x in ls)
  assert r['first_upper_s']==(min(x['settle_ms'] for x in ls)-ws)/1000
  assert r['last_upper_s']==(max(x['settle_ms'] for x in ls)-ws)/1000
  a=p[r['hash']];win=[(float(a[k])-ws)/1000 for k in ['effect_lo_ms','effect_hi_ms']] if a['estimate_enabled']=='True' else None
  assert r['post_window_s']==win and r['post_grade']==a['grade']
  assert r['original_shares'] is None and r['end_kind']=='unknown' and r['known_private_cancel'] is False
validate(d)
bad_cases=[]
for k in ['price','quantity','time','omit','promote','cancel','size']:
 x=copy.deepcopy(d)
 if k=='price':x['rows'][0]['price_c']+=1
 if k=='quantity':x['rows'][0]['filled_first60_sh']+=1
 if k=='time':x['rows'][0]['post_window_s'][0]-=1
 if k=='omit':x['rows'].pop()
 if k=='promote':next(r for r in x['rows'] if r['post_window_s'] is None)['post_window_s']=[0,1]
 if k=='cancel':x['rows'][0]['known_private_cancel']=True
 if k=='size':x['rows'][0]['original_shares']=220
 try:validate(x)
 except AssertionError:bad_cases.append(k)
 else:raise AssertionError('escaped '+k)
r=d['rows'];by={x['hash']:x for x in r}
bid=[x for x in r if x['price_c']==56 and x['side']=='bid'];ask58=next(x for x in r if x['price_c']==58 and x['side']=='ask')
results=dict(raw_legs=len(raw),orders=len(r),checks_pass=True,corruptions_refused=bad_cases,
 later_56_post_after_first56_fill=bid[1]['post_window_s'][0]>bid[0]['last_upper_s'],
 ask58_post_after_all_56_fills=ask58['post_window_s'][0]>max(x['last_upper_s'] for x in bid),
 rule_or_private_queue_identified=False)
(H/'maker3_quote_checks.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
