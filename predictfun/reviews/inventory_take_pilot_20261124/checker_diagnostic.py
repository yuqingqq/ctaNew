import gzip,json,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(3,4))
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research/inventory_take_pilot_20261124'
out=[]
for arm in ['TAKE_100','TAKE_220']:
    r=json.loads(gzip.decompress((D/f'3702960_{arm}.json.gz').read_bytes()))
    for z in r['ioc_results']:
        if z['gid']!=30:continue
        t=z['act'];opp=1-z['side']
        for o in r['orders']:
            filled=sum(f[3] for f in r['fills'] if f[1]==o['gid'] and f[0]<=t)
            if str(o['gid']) not in r['iocs'] and o['side']==opp and o['act']<=t<o['remove'] and o['size']-filled>1e-9 and (o['price']<=z['limit'] if z['side']==0 else o['price']>=z['limit']):
                out.append(dict(arm=arm,ioc=z['gid'],order=o,filled_before=filled))
assert out and all(x['order']['refused'] for x in out)
print(json.dumps(out,indent=2))
