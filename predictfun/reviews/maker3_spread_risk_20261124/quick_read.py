"""Descriptive drill-down of the already accepted payload; no new selection."""
import gzip,json
import core as C
d=json.loads(gzip.decompress((C.OUT/'data.json.gz').read_bytes()))
for mid in [3838531,3789484]:
 c=next(x for x in d['cases'] if x['mid']==mid)
 print(mid,'totals',c['totals'])
 for f in c['fills']:
  if f['role']=='taker' and f['t']>=180:
   q=sum(x['s']*x['q'] for x in c['fills'] if x['t']<f['t'])
   print('take',f['lo'],f['t'],f['native'],f['q'],f['price'],'net prior',q)
 print('wide_seconds',c['occupancy'])
