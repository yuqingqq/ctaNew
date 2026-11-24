import gzip,json,collections
from pathlib import Path
h=Path(__file__).resolve().parent;d=json.loads(gzip.decompress((h/'data.json.gz').read_bytes()));out=[]
for m in d['cases']:
 fs=m['makers']
 if m['id']==3702960:f=next(x for x in fs if x['rank']==3 and x['side']==1 and x['price']==24)
 elif m['id']==3789484:f=next(x for x in fs if x['rank']==1 and x['side']==0 and x['price']==46 and x['t']>250)
 else:f=next(x for x in fs if x['hash']=='0xddd8ab1d85e704a84aad728770ddc1bed353830e6692fe2b14fdbfde8f827df9' and x['role']=='maker')
 a=m['ours']['CURRENT_ASYNC_FAST10'];r=[x for x in a['curve'] if f['lo']<=x[0]<=f['t']];waits=sorted(set(w[4] for w in a['waits'] if w[2]==f['side'] and w[0]<=f['t'] and w[1]>f['lo']));orders=[dict(gid=o['gid'],price=o['price'],reason=o['reason'],act=o['act'],stop=o['stop']) for o in a['orders'] if o['side']==f['side'] and not o['refused'] and o['act']<=f['t'] and o['stop']>f['lo']]
 out.append(dict(m=m['id'],fill=f,own_sampled_inventory=[min(x[1] for x in r),max(x[1] for x in r)],own_modes=sorted(set(x[4] for x in r)),wait_reasons=waits,overlapping_orders=orders))
(h/'examples.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
