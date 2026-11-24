"""Reconcile opening diagnostic against saved execution and signal artifacts.

User-selected, post-hoc market example. No simulation, fitting or new data.
Positive trace plus deliberately wrong timing, quantity and signal claims.
"""
import copy,gzip,hashlib,json
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research'
arm='CURRENT_ASYNC_FAST10';mid=3789484;day='20261026'
lp=D/f'ladder_async_fast_v1_20261122/pilot/{day}.jsonl.gz'
ap=lp.with_name(f'{day}_arrays.npz')
cp=D/f'ladder_pressure_v1_20261121/pilot/{day}_arrays.npz'
with gzip.open(lp,'rt') as f:a=next(json.loads(x)[arm] for x in f if json.loads(x)['m']==mid)
z=np.load(ap);c=np.load(cp);pre=f'{mid}__{arm}__';edge=0.285456120967865
fills=[x for x in a['fills'] if x[0]<800]
classes=[cl for f,cl in zip(a['fills'],a['fill_class']) if f[0]<800]
assert len(classes)==len(fills)
assert all(cl[0]==('through' if cl[1]>f[4] else 'same_price') for f,cl in zip(fills,classes))
commands=z[pre+'commands'];ask=c[f'{mid}__cache__x1'];st=c[f'{mid}__cache__status1']
ts=50*np.arange(len(ask));on=np.isfinite(ask)&(ask>0)&(ask>=edge)&(st==0)
ds=json.loads(gzip.decompress((H/'data.json.gz').read_bytes()))
m=next(x for x in ds['cases'] if x['id']==mid)
df=[x for x in m['ours'][arm]['fills'] if x['t']<.8]
assert len(df)==len(fills)
for p,q in zip(df,fills):
 assert p['t']*1000==q[0] and p['gid']==q[1] and p['side']==q[2]
 assert abs(p['q']-q[3])<.51e-6 and p['price']==q[4]
assert set(x[1] for x in fills)=={2,3}
first=min(x[0] for x in fills)
orders={int(o['gid']):o for o in a['orders']}
out=dict(market=mid,day=day,arm=arm,asof='2026-11-24',
 population='first own simulated fills before 800 ms; selected opening diagnostic',
 fill_clock='saved M50 simulator assumption, not observed exchange execution time',
 first_fill_ms=first,first_fill_legs=len(fills),
 initial_fill_classes=classes,
 all_initial_fills_through=all(x[0]=='through' for x in classes),
 by_price={str(p):sum(x[3] for x in fills if x[4]==p) for p in sorted({x[4] for x in fills})},
 ask_signal_threshold=edge,all_prefill_grid_status_ok=bool(np.all(st[ts<first]==0)),
 max_prefill_grid_signal=float(np.max(ask[ts<first])),
 first_positive_signal_ms=float(ts[np.flatnonzero(ask>0)[0]]),
 first_on_signal_ms=float(ts[np.flatnonzero(on)[0]]),
 requests_before_first_fill=[x for x in a['requests'] if x[0]<first],
 opening_commands=commands[commands[:,0]==0].tolist(),
 ask_orders=[orders[g] for g in (2,3)],
 first_fill_notice_grid_ms=float(ts[np.flatnonzero(z[pre+'posc']<0)[0]]),
 first_fill_notice_position=float(z[pre+'posc'][np.flatnonzero(z[pre+'posc']<0)[0]]),
 initial_ask_markout_1s_c_per_share=sum(f['q']*f['markouts']['1000'] for f in df)/sum(f['q'] for f in df),
 source_sha256={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [lp,ap,cp,H/'data.json.gz']})
def validate(v):
 assert v['first_fill_ms']==first
 assert v['by_price']=={str(p):sum(x[3] for x in fills if x[4]==p) for p in (50.,51.)}
 assert v['first_on_signal_ms']==float(ts[np.flatnonzero(on)[0]])
 assert v['max_prefill_grid_signal']==float(np.max(ask[ts<first]))
 assert v['requests_before_first_fill']==[x for x in a['requests'] if x[0]<first]
 assert v['first_fill_notice_grid_ms']==float(ts[np.flatnonzero(z[pre+'posc']<0)[0]])
validate(out)
bad=[]
for name in ['fill_time','quantity','warning_time','prefill_signal','cancel_pending','notice_time']:
 x=copy.deepcopy(out)
 if name=='fill_time':x['first_fill_ms']+=50
 if name=='quantity':x['by_price']['50.0']+=1
 if name=='warning_time':x['first_on_signal_ms']=400
 if name=='prefill_signal':x['max_prefill_grid_signal']=1
 if name=='cancel_pending':x['requests_before_first_fill']=[[400,2,'policy_request',410]]
 if name=='notice_time':x['first_fill_notice_grid_ms']=450
 try:validate(x)
 except AssertionError:bad.append(name)
 else:raise AssertionError('escaped '+name)
out['checks']=dict(raw_fill_reconciliation=True,intact_pass=True,corruptions_refused=bad)
(H/'opening_fill_evidence.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in out.items() if k not in ['ask_orders','source_sha256','opening_commands']},indent=2))
