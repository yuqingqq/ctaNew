"""Independent saved-artifact checks. No simulator, fitting or new market data."""
import base64,collections,copy,csv,gzip,hashlib,json,math
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent; R=H.parents[2]
L=R/'data/predict_fun/research/ladder_async_fast_v1_20261122'
def near(a,b):
 if a is None or b is None:return a is b
 return abs(float(a)-float(b))<2e-5
def require(ok,msg):
 if not ok:raise ValueError(msg)
def check_arm(m,a,raw,z):
 fills=a['fills'];require(len(fills)==len(raw['fills']),'fill membership')
 for f,r in zip(fills,raw['fills']):
  require(all(near(f[k],v) for k,v in zip(['t','gid','side','q','price','trade'],[r[0]/1000,r[1],r[2],r[3],r[4],r[5]])),'fill identity')
  require(near(f['end_value'],(1-2*r[2])*r[3]*(m['mark']-r[4])/100),'fill value')
 cash=sum(-(1-2*f['side'])*f['q']*f['price'] for f in fills);pos=sum((1-2*f['side'])*f['q'] for f in fills)
 require(abs(cash-raw['cash'])<=sum(f['price'] for f in fills)*.500001e-6+1e-7 and abs(pos-raw['position'])<=len(fills)*.500001e-6+1e-7,'cash inventory')
 require(abs(a['summary']['pnl_usd']-(cash+pos*m['mark'])/100)<=sum(abs(m['mark']-f['price']) for f in fills)*.500001e-8+1e-6,'END value')
 require(a['front_treatment'] if 'front_treatment' in a else a['summary']['front_treatment']==a['label'].startswith('FRONT'),'front label')
 require(len(a['orders'])==len(raw['orders']),'order membership')
 for o,r in zip(a['orders'],raw['orders']):
  require(all(near(o[k],r[k]) for k in ['gid','side','rung','price','size','remaining'] if k!='remaining'),'order identity')
  require(near(o['act'],r['act']/1000) and near(o['submit'],r['submit']/1000),'order clock')
  require(near(o['request'],None if r['req'] is None else r['req']/1000),'request clock')
  ff=[f for f in fills if f['gid']==o['gid']];q=0;end=m['end']
  for f in ff:
   q+=f['q']
   if q>=o['size']-1e-8:end=min(end,f['t']);break
  require(near(o['stop'],min(end,r['remove']/1000)),'order lifetime')
  for key in ['ahead0','first','pull']:
   v=raw['queue_venue'][key].get(str(o['gid']));out=o['queue'][key]
   require((out is None) if v is None else all(near(x,y) for x,y in zip(out,[v[0]/1000,v[1],v[2]])),'queue sample')
  for f in ff:require(f['after_request']==(r['req'] is not None and r['req']<f['t']*1000<r['remove']),'pending classification')
 prefix=f"{m['id']}__{a['label']}__"
 for row in a['curve'][:-1]:
  i=round(row[0]/.05);p=z[prefix+'pose'][i];mid=z[prefix+'mid'][i]
  expected=[p,z[prefix+'posc'][i],(z[prefix+'cash'][i]+p*mid)/100 if math.isfinite(mid) else None]
  require(all(near(x,y) for x,y in zip(row[1:4],expected)),'observer curve')
 require(near(a['curve'][-1][3],a['summary']['pnl_usd']),'final curve')
def classify(lo,hi,orders,side,price,better):
 if lo is None or hi is None:return 'unknown'
 spans=[(o['act'],o['stop']) for o in orders if not o['refused'] and o['side']==side and (o['price']>=price-1e-6 if better and side==0 else o['price']<=price+1e-6 if better else abs(o['price']-price)<1e-6)]
 points=sorted(set([lo,hi]+[x for ab in spans for x in ab if lo<x<hi]));checks=points+[(a+b)/2 for a,b in zip(points,points[1:])];covered=[any(a<=t<b for a,b in spans) for t in checks]
 return 'throughout' if all(covered) else 'partial' if any(covered) else 'none'
def main():
 data=json.loads(gzip.decompress((H/'data.json.gz').read_bytes()));require(len(data['cases'])==3,'market membership')
 require(base64.b64decode((H/'data.js').read_text().split('"')[1])==(H/'data.json.gz').read_bytes(),'browser data parity')
 for name,v in json.loads((H/'manifest_v2.json').read_text())['files'].items():
  p=R/name;h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  require(h.hexdigest()==v['sha256'],'source '+name)
 with (R/'predictfun/reviews/maker_trace_20261122/v2/matched_gap_r1/per_market.csv').open() as f:accepted=list(csv.DictReader(f))
 seen=collections.Counter();falsifiers=[]
 for m in data['cases']:
  part='rest' if m['day']=='20261024' else 'pilot'
  with gzip.open(L/part/(m['day']+'.jsonl.gz'),'rt') as f:raw=next(json.loads(line) for line in f if json.loads(line)['m']==m['id'])
  with np.load(L/part/(m['day']+'_arrays.npz')) as z:
   for label,a in m['ours'].items():
    check_arm(m,a,raw[label],z);seen['arms']+=1;seen['fills']+=len(a['fills']);seen['orders']+=len(a['orders'])
   a=m['ours']['CURRENT_ASYNC_FAST10']
   for field,delta in [('side',1),('price',1),('t',1),('end_value',1)]:
    bad=copy.deepcopy(a);bad['fills'][0][field]+=delta
    try:check_arm(m,bad,raw[a['label']],z)
    except ValueError:falsifiers.append(field);continue
    raise ValueError('escaped '+field)
   for mutation in ['front','queue']:
    bad=copy.deepcopy(a)
    if mutation=='front':bad['summary']['front_treatment']=True
    else:next(o for o in bad['orders'] if o['queue']['ahead0'] is not None)['queue']['ahead0'][1]+=100
    try:check_arm(m,bad,raw[a['label']],z)
    except ValueError:falsifiers.append(mutation);continue
    raise ValueError('escaped '+mutation)
  require(sum(m['source_counts'].values())-m['source_counts']['source']+len(m['makers'])==m['source_counts']['source'],'source count closure')
  for rank in [1,2,3]:
   fs=[f for f in m['makers'] if f['rank']==rank];v=sum(f['end_value'] for f in fs);src=next(r for r in accepted if int(r['m'])==m['id'] and r['party']==f'rank{rank}' and r['view']=='zero')
   require(abs(v-(float(src['total']) if src['total'] else 0))<.001,'maker accepted accounting')
   require(near(v,m['maker_totals'][str(rank)]),'maker displayed accounting')
  keys=[(f['sid'],f['mk_idx'],f['role'],f['rank']) for f in m['makers']];require(len(keys)==len(set(keys)),'maker duplicates')
  for f in m['makers']:
   require(near(f['end_value'],(1-2*f['side'])*f['q']*(m['mark']-f['price'])/100-f['fee']),'maker row economics')
   for arm,a in m['ours'].items():
    for k,better in [('same',False),('equal_or_better',True)]:require(f['presence'][arm][k]==classify(f['lo'],f['t'],a['orders'],f['side'],f['price'],better),'quote presence')
  seen['maker_legs']+=len(m['makers'])
 # Touching an exact removal boundary is not coverage throughout.
 require(classify(1,2,[dict(act=0,stop=2,refused=False,side=0,price=50)],0,50,False)=='partial','strict boundary control')
 result=dict(passed=True,counts=dict(seen),falsifiers_caught=len(falsifiers),no_profitability_or_causality_claim=True)
 (H/'checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
