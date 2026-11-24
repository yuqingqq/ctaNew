"""Verify the UI projection against saved ledgers/arrays and cache."""
import copy,gzip,json,math,resource
from pathlib import Path
import numpy as np
resource.setrlimit(resource.RLIMIT_CPU,(12,13))
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research';UI=R/'predictfun/reviews/maker_dashboard_20261124'
def read(p):return json.loads(gzip.decompress(p.read_bytes()))
x=read(H/'data.json.gz');old=read(UI/'data.json.gz');loaded={}
for m in x['cases']:
 for arm,folder in [('FIXED_DEFAULT','entry_gate_pilot_20261124'),('OPENING_ONLY','opening_entry_pilot_20261124')]:
  with np.load(D/folder/f'{m["id"]}_{arm}.npz') as z:
   arrays={k:z[k] for k in ['pose','posc','cash','mid']}
  loaded[(m['id'],arm)]=(read(D/folder/f'{m["id"]}_{arm}.json.gz'),arrays)
def near(a,b):
 if b is None or not math.isfinite(float(b)):return a is None
 return a is not None and abs(a-b)<2e-6
def verify(x):
 assert len(x['cases'])==3 and {m['id'] for m in x['cases']}=={3702960,3789484,3877472}
 for m in x['cases']:
  orig=next(v for v in old['cases'] if v['id']==m['id'])
  assert m['book']==orig['book'] and m['makers']==[f for f in orig['makers'] if f['rank']==3]
  assert m['maker_curve']==next(v['curve'] for v in orig['maker_curves'] if v['rank']==3)
  for arm in ['FIXED_DEFAULT','OPENING_ONLY']:
   a=m['arms'][arm];r,z=loaded[(m['id'],arm)]
   assert len(a['fills'])==len(r['fills']) and len(a['orders'])==len(r['orders'])
   for f,v in zip(a['fills'],r['fills']):assert all(near(f[k],val) for k,val in zip(['t','gid','side','q','price','trade'],[v[0]/1000,*v[1:]]))
   cash=sum(-(1-2*f['side'])*f['q']*f['price'] for f in a['fills']);pos=sum((1-2*f['side'])*f['q'] for f in a['fills'])
   assert abs(cash+pos*m['mark']-r['value'])<1e-3
   assert near(a['metrics']['total_c'],r['value'])
   for o,v in zip(a['orders'],r['orders']):
    for k in ['gid','side','rung','price']:assert o[k]==v[k]
    assert o['refused']==v['refused'] and near(o['q'],v['size'])
    assert near(o['submit'],v['submit']/1000) and near(o['act'],v['act']/1000)
    assert near(o['request'],None if v['req'] is None else v['req']/1000)
    # Lifetime uses exact source quantities; display quantities are rounded to
    # eight decimals and can straddle the full-fill tolerance when summed.
    ff=[f for f in r['fills'] if f[1]==o['gid']];amount=0;stop=m['end']
    for f in ff:
     amount+=f[3]
     if amount>=v['size']-1e-8:stop=min(stop,f[0]/1000);break
    assert near(o['stop'],min(stop,v['remove']/1000))
   assert a['first_orders']==[o for o in a['orders'] if o['submit']==a['first_post']]
   assert a['first_post']==min(o['submit'] for o in a['orders']) and a['first_fill']==a['fills'][0]['t']
   for row in a['curve'][:-1]:
    i=round(row[0]/.05);assert near(row[1],z['pose'][i]) and near(row[2],z['posc'][i])
    assert near(row[3],(z['cash'][i]+z['pose'][i]*z['mid'][i])/100)
   assert near(a['curve'][-1][3],r['value']/100)
  assert m['gate']==m['arms']['OPENING_ONLY']['first_post'] and near(m['resume'],m['gate']+.05)
  for e in m['arms']['OPENING_ONLY']['entry_log']:
   assert e['t']/1000<=m['gate']
   if e['status']=='ok':assert near(e['other_signal'],m['signal'][round(e['t']/50)][1]*(1 if e['side']==0 else -1))
 return True
verify(x)
bad_cases=[lambda v:v['cases'][0]['arms']['OPENING_ONLY']['fills'].pop(),lambda v:v['cases'][0].update(resume=100),lambda v:v['cases'][0]['arms']['OPENING_ONLY']['orders'][0].update(price=99),lambda v:v['cases'][0]['arms']['OPENING_ONLY']['curve'][0].__setitem__(1,99),lambda v:v['cases'][1]['makers'].pop(),lambda v:v['cases'][0]['signal'][0].__setitem__(1,999)]
for fn in bad_cases:
 b=copy.deepcopy(x);fn(b)
 try:verify(b)
 except (AssertionError,TypeError):pass
 else:raise AssertionError('corruption escaped')
result=dict(passed=True,arms=6,corruptions_refused=len(bad_cases),fills=sum(len(a['fills']) for m in x['cases'] for a in m['arms'].values()),orders=sum(len(a['orders']) for m in x['cases'] for a in m['arms'].values()),maker3_legs=sum(len(m['makers']) for m in x['cases']),no_replay=True)
(H/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
