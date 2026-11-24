"""Counts/schema/identity gate only; no new linkage outcome."""
import csv, hashlib, json, sys, time
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path('/home/yuqing/ctaNew');HERE=Path(__file__).resolve().parent
T=ROOT/'predictfun/reviews/maker_terminal_chain_20261124';RAW=ROOT/'data/predict_fun/research/maker_terminal_chain_20261124'
V=ROOT/'data/predict_fun/research/maker_trace_v2_20261123';B=ROOT/'data/predict_fun/research/maker_trace_v1_20261122'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(1024*1024),b''):h.update(x)
 return h.hexdigest()
start=time.process_time();rows=list(csv.DictReader((T/'orders.csv').open()));assert len(rows)==88 and len({r['order_hash'] for r in rows})==88
pairs=sorted({(r['day'],int(r['market'])) for r in rows});assert all(d in ('20261024','20261025','20261026','20261027') for d,m in pairs)
paths=[T/'orders.csv',T/'manifest.json',T/'summary.json',T/'checks.json',T/'check.py',RAW/'events.json']+sorted(RAW.glob('batch_*.json'))
TI=ROOT/'predictfun/reviews/maker_timing_estimates_20261124'
paths += [TI/'outputs'/x for x in ('post_estimates.csv.gz','union_windows.csv.gz','signed_chain_estimates.csv')]
paths += [TI/'outputs/summary.json',TI/'readout.json',ROOT/'predictfun/reviews/maker_indirect_relations_20261124/plan.md']
meta=[]
for d,m in pairs:
 for kind in ('frames','levels','fills','orders'):
  p=(B/'depth' if kind in ('frames','levels') else V/'life')/f'{d}_{m}_{kind}.parquet'
  pf=pq.ParquetFile(p);meta.append(dict(day=d,mid=m,kind=kind,path=str(p.relative_to(ROOT)),rows=pf.metadata.num_rows,bytes=p.stat().st_size,columns=pf.schema_arrow.names));paths.append(p)
 paths.append(V/'life'/f'{d}_{m}.json')
files={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
old=json.loads((T/'summary.json').read_text());assert files[str((RAW/'events.json').relative_to(ROOT))]==old['outputs']['events.json']
for p in RAW.glob('batch_*.json'):assert files[str(p.relative_to(ROOT))]==old['outputs'][p.name]
assert files[str((T/'orders.csv').relative_to(ROOT))]==old['orders_sha256']
result=dict(orders=88,markets=len(pairs),pairs=pairs,files=files,parquets=meta,total_bytes=sum((ROOT/p).stat().st_size for p in files),cpu_seconds=time.process_time()-start)
assert result['total_bytes']<150*1024*1024,'bounded source cap'
assert sum(x['rows'] for x in meta if x['kind']=='frames')<250000,'frame cap'
assert not (HERE/'inputs.json').exists();(HERE/'inputs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('files','parquets')},indent=2));print('Schemas:',{k:next(x['columns'] for x in meta if x['kind']==k) for k in ('frames','levels','fills','orders')})
