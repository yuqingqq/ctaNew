"""Read-only UI projection; no simulator imports."""
import collections,gzip,hashlib,json,math,resource
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(12,13))
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research'
UI=R/'predictfun/reviews/maker_dashboard_20261124'
def read(p):return json.loads(gzip.decompress(p.read_bytes())) if str(p).endswith('.gz') else json.loads(p.read_text())
def clean(v):
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(tuple,list,np.ndarray)):return [clean(x) for x in v]
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float):return round(v,8) if math.isfinite(v) else None
 return v
inputs={}
def bind(p):
 inputs[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
old=read(bind(UI/'data.json.gz'));cases=[]
opening_inputs=read(bind(R/'predictfun/reviews/opening_entry_pilot_20261124/manifest.json'))['sources']
expected={}
for receipt in ['entry_gate_pilot_20261124','opening_entry_pilot_20261124']:
 ss=read(bind(R/f'predictfun/reviews/{receipt}/summary.json'))
 # Some receipts flatten the run's rows differently; bind raw run summaries too.
 rs=read(bind(D/receipt/'summary.json'))
 for row in rs['rows']:expected[(row['m'],row['arm'])]=row
for m in old['cases']:
 day=m['day'];mid=m['id'];part='days' if mid==3702960 else 'pilot'
 cache=np.load(bind(D/'ladder_pressure_v1_20261121'/part/f'{day}_arrays.npz'))
 sx=cache[f'{mid}__cache__x1'];status=cache[f'{mid}__cache__status1'];age=cache[f'{mid}__cache__age1']
 signal=[[i*.05,float(v) if int(st)==0 and np.isfinite(v) else None,float(ag)] for i,(v,st,ag) in enumerate(zip(sx,status,age))]
 arms={}
 for arm,folder in [('FIXED_DEFAULT','entry_gate_pilot_20261124'),('OPENING_ONLY','opening_entry_pilot_20261124')]:
  rp=bind(D/folder/f'{mid}_{arm}.json.gz');ap=bind(D/folder/f'{mid}_{arm}.npz')
  assert inputs[str(rp.relative_to(R))]==expected[(mid,arm)]['sha256']
  array_hash=expected[(mid,arm)].get('arrays_sha256',opening_inputs.get(str(ap.relative_to(R))))
  assert array_hash is not None and inputs[str(ap.relative_to(R))]==array_hash
  rec=read(rp);a=np.load(ap);assert rec['ident']['start_abs_ms']==m['ws'] and rec['mark']==m['mark']
  fills=[dict(t=f[0]/1000,gid=f[1],side=f[2],q=f[3],price=f[4],trade=f[5]) for f in rec['fills']]
  by=collections.defaultdict(list)
  for f in fills:by[f['gid']].append(f)
  orders=[]
  for o in rec['orders']:
   end=m['end'];cum=0
   for f in by[o['gid']]:
    cum+=f['q']
    if cum>=o['size']-1e-8:end=min(end,f['t']);break
   orders.append(dict(gid=o['gid'],side=o['side'],rung=o['rung'],price=o['price'],q=o['size'],submit=o['submit']/1000,act=o['act']/1000,stop=min(end,o['remove']/1000),request=None if o['req'] is None else o['req']/1000,refused=o['refused']))
  curve=[[i*.05,float(a['pose'][i]),float(a['posc'][i]),float((a['cash'][i]+a['pose'][i]*a['mid'][i])/100)] for i in range(0,len(a['pose']),2)]
  curve.append([m['end'],rec['position'],None,rec['value']/100])
  first=min(o['submit'] for o in orders);metrics=rec['metrics']
  arms[arm]=dict(orders=orders,fills=fills,curve=curve,metrics=metrics,first_post=first,first_fill=fills[0]['t'] if fills else None,first_orders=[o for o in orders if o['submit']==first],entry_log=rec.get('entry_log',[]))
  if arm=='OPENING_ONLY':
   gate=rec['opening_batch_ms']/1000;resume=gate+.05
   for e in rec['entry_log']:
    assert e['t']<=rec['opening_batch_ms']
    sig=signal[int(round(e['t']/50))][1]
    if e['status']=='ok':assert sig is not None and abs(sig*(1 if e['side']==0 else -1)-e['other_signal'])<1e-6
  a.close()
 cache.close()
 makers=[f for f in m['makers'] if f['rank']==3]
 mc=next(v['curve'] for v in m['maker_curves'] if v['rank']==3)
 cases.append(dict(id=mid,day=day,end=m['end'],mark=m['mark'],book=m['book'],makers=makers,maker_curve=mc,maker_total=m['maker_totals']['3'],arms=arms,signal=signal,gate=gate,resume=resume,threshold_bid=.285456120967865,threshold_ask=-.4666730940341952))
out=clean(dict(cases=cases,version='opening_comparison_v1',asof='2026-11-24',notes='Saved consumed examples; first batch only; same inventory fix; maker execution upper bounds; zero maker fee simulation; no settlement or validation claim.'))
raw=json.dumps(out,separators=(',',':'),allow_nan=False).encode()
(H/'data.json.gz').write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
(H/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
summary=[dict(m=m['id'],gate=m['gate'],resume=m['resume'],first_fills={k:a['first_fill'] for k,a in m['arms'].items()},first_orders={k:a['first_orders'] for k,a in m['arms'].items()},maker_first=min((f['t'] for f in m['makers']),default=None),maker_first_passive=min((f['t'] for f in m['makers'] if f['role']=='maker'),default=None)) for m in cases]
(H/'summary.json').write_text(json.dumps(clean(summary),indent=2)+'\n');print(json.dumps(clean(summary),indent=2));print('packed bytes',len((H/'data.json.gz').read_bytes()))
