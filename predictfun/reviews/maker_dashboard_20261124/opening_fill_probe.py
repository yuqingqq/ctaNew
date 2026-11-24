"""Saved-trace diagnosis of the first own ask sweep. No policy replay."""
import gzip,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent;R=H.parents[2];D=R/'data/predict_fun/research'
M=next(x for x in json.loads(gzip.decompress((H/'data.json.gz').read_bytes()))['cases'] if x['id']==3789484)
root=D/'ladder_async_fast_v1_20261122/pilot';raw=None
with gzip.open(root/'20261026.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line)
  if r['m']==3789484:raw=r;break
assert raw is not None
arm='CURRENT_ASYNC_FAST10';a=raw[arm];o=M['ours'][arm]
print('RECORD_KEYS',list(raw));print('ARM_KEYS',list(a));print('IDENT',a['ident']);print('PATH_INFO',raw.get('path_info'))
print('ORDERS',json.dumps(o['orders'][:6]));print('FILLS',json.dumps([f for f in o['fills'] if f['t']<.8]));print('COMMANDS',json.dumps([x for x in o['commands'] if x[0]<.8]));print('GUARDS',json.dumps([x for x in a['guard_log'] if x[0]<800]));print('STRUCTURAL',json.dumps([x for x in a['structural'] if x['t']<800]) if 'structural' in a else '')
for key in ['events','requests','mode_log','struct_log']:
 if key in a:print(key,json.dumps(a[key][:20]))
z=np.load(root/'20261026_arrays.npz');prefix=f'3789484__{arm}__'
print('ARRAY_KEYS',[(k,z[k].shape) for k in z.files if k.startswith(prefix)])
for k in z.files:
 if k.startswith(prefix):print('ARRAY',k,json.dumps(z[k][:17].tolist()))
mt=D/'maker_trace_v2_20261123';fr=pq.read_table(mt/'depth/20261026_3789484_frames.parquet').to_pandas();lv=pq.read_table(mt/'depth/20261026_3789484_levels.parquet').to_pandas()
fr=fr[(fr.upd_ms>=M['ws']-300)&(fr.upd_ms<=M['ws']+1000)].sort_values(['upd_ms','recv_ns','fid'])
out=[]
for f in fr.to_dict('records'):
 l=lv[lv.fid==f['fid']];bid=l[(l.side=='bid')&(l.shares>0)];ask=l[(l.side=='ask')&(l.shares>0)]
 out.append(dict(fid=f['fid'],venue_ms=f['upd_ms']-M['ws'],receipt_ms=f['recv_ns']/1e6-M['ws'],bid=int(bid.ptick.max()),ask=int(ask.ptick.min()),asks=l[(l.side=='ask')&(l.ptick.between(49,56))][['ptick','shares']].to_dict('records'),status=f['status']))
print('NATIVE',json.dumps(out))
print('MAKER3',json.dumps([f for f in M['makers'] if f['rank']==3 and f['t']<5]))
print('BOOK',json.dumps([r for r in M['book'] if r[0]<1.6]));print('BTC',json.dumps([r for r in M['btc'] if r[0]<1.6]))
