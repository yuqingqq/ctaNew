"""Schema/count-only preparation. No economics or signal values printed."""
import gzip,json,time
from pathlib import Path
import pyarrow.parquet as pq
R=Path('/home/yuqing/ctaNew');D=R/'data/predict_fun/research';M=D/'maker_trace_v2_20261123'
for day,mid,part in [('20261024',3702960,'rest'),('20261026',3789484,'pilot'),('20261027',3877472,'pilot')]:
 p=D/'ladder_async_fast_v1_20261122'/part/(day+'.jsonl.gz')
 with gzip.open(p,'rt') as f:
  for line in f:
   x=json.loads(line)
   if x['m']!=mid:continue
   print(day,mid,'outer',list(x));r=x['CURRENT_ASYNC_FAST10'];print('record',list(r));print('ident',r['ident']);print('queue example shape',r['queue_venue'][0] if isinstance(r['queue_venue'],list) and r['queue_venue'] else {k:(type(v).__name__,len(v)) for k,v in r['queue_venue'].items()} if isinstance(r['queue_venue'],dict) else None);print('fill_mids',type(r['fill_mids']).__name__,list(r['fill_mids'])[:10] if isinstance(r['fill_mids'],dict) else [type(v).__name__ for v in r['fill_mids'][:1]]);print('order_keys',list(r['orders'][0]));break
 for p in [M/'depth'/f'{day}_{mid}_frames.parquet',M/'depth'/f'{day}_{mid}_levels.parquet',M/'life'/f'{day}_{mid}_fills.parquet',M/'btc'/f'btc_{day}.parquet']:
  f=pq.ParquetFile(p);print(p.name,f.metadata.num_rows,f.schema_arrow.names)
f=pq.ParquetFile(M/'econ/fills_valued.parquet');print('valued_schema',f.schema_arrow.names,'groups',f.metadata.num_row_groups)
