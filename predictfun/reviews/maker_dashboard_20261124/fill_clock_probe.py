"""Read-only diagnosis of the user-named 63c fill, saved market 3789484.
No replay, no fitted/selected rule. Preserve original payload and source files.
"""
import bisect, gzip, json
from pathlib import Path
import pyarrow.parquet as pq
H = Path(__file__).resolve().parent
MT = Path('/home/yuqing/ctaNew/data/predict_fun/research/maker_trace_v2_20261123')
HASH = '0x70d3ee3a032dca76f47697efb402735c8467c4823c0b11bc3d932610dddca13c'
m = next(x for x in json.loads(gzip.decompress((H/'data.json.gz').read_bytes()))['cases'] if x['id']==3789484)
fs = [x for x in m['makers'] if x['hash']==HASH]
print('PAYLOAD', json.dumps(fs)); print('OPENING', m['ws'])
for f in fs:
 for k in ['book','depth']:
  rs=m[k]; i=bisect.bisect_right([r[0] for r in rs],f['t'])-1
  print('DISPLAY_AT_UPPER',f['id'],k,json.dumps(rs[max(0,i-1):i+2]))
lo=min(f['lo'] if f['lo'] is not None else f['t'] for f in fs)-1
hi=max(f['t'] for f in fs)+1
life=pq.read_table(MT/'life/20261026_3789484_fills.parquet').to_pandas()
print('LIFE_SCHEMA',list(life.columns))
print('LIFE_ROWS',life[life.mk_hash==HASH].to_json(orient='records'))
fr=pq.read_table(MT/'depth/20261026_3789484_frames.parquet').to_pandas()
lv=pq.read_table(MT/'depth/20261026_3789484_levels.parquet').to_pandas()
print('FRAME_SCHEMA',list(fr.columns));print('LEVEL_SCHEMA',list(lv.columns))
fr['rel']=(fr.upd_ms-m['ws'])/1000
fr=fr[(fr.rel>=lo)&(fr.rel<=hi)].sort_values(['upd_ms','recv_ns','fid'])
print('NATIVE_FRAMES',fr.to_json(orient='records'))
sel=lv[lv.fid.isin(fr.fid)]
print('NATIVE_LEVELS',sel.to_json(orient='records'))
