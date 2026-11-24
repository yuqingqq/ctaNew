"""Counts, schemas and byte identities only; no new chain inference."""
import csv,json,hashlib,sys,importlib.util
from pathlib import Path
import pyarrow.parquet as pq
R=Path('/home/yuqing/ctaNew');H=Path(__file__).resolve().parent;Q=R/'predictfun/reviews/maker_quantity_linkage_20261124';T=R/'predictfun/reviews/maker_timing_estimates_20261124';V=R/'data/predict_fun/research/maker_trace_v2_20261123';C=R/'predictfun/reviews/maker_trace_20261122/v2/chains';S=R/'predictfun/reviews/maker_terminal_chain_20261124'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
rows=list(csv.DictReader((Q/'outputs/order_summary.csv').open()));pairs=sorted({(r['day'],int(r['mid'])) for r in rows});assert len(rows)==88 and len(pairs)==18
paths=[Q/'outputs/order_summary.csv',Q/'outputs/chain_bounds.csv',Q/'outputs/normalization.json',Q/'outputs/result.json',Q/'checks.json',T/'outputs/post_estimates.csv.gz',T/'outputs/signed_chain_estimates.csv',S/'orders.csv',S/'manifest.json',R/'data/predict_fun/research/maker_terminal_chain_20261124/events.json',C/'take_logic_v2/panel.parquet',C/'take_logic_v2/events.parquet',C/'take_logic_v2/receipt.json',C/'posting_opportunity/rows.parquet',C/'posting_opportunity/receipt.json',R/'predictfun/reviews/maker_condition_action_20261124/outputs/overview.csv',R/'predictfun/reviews/maker_condition_action_20261124/checks.json']
norm=json.loads((Q/'outputs/normalization.json').read_text())
for n in norm:
 p=R/n['normalized_file'];assert sha(p)==n['normalized_sha256'];paths.append(p)
for d,m in pairs:paths += [V/'life'/f'{d}_{m}_fills.parquet',V/'life'/f'{d}_{m}_orders.parquet',V/'life'/f'{d}_{m}.json']
for d in sorted({d for d,m in pairs}):paths.append(V/f'records_{d}.parquet')
files={str(p.relative_to(R)):sha(p) for p in sorted(set(paths))};meta=[]
for p in paths:
 if p.suffix=='.parquet':
  z=pq.ParquetFile(p);meta.append(dict(path=str(p.relative_to(R)),rows=z.metadata.num_rows,columns=z.schema_arrow.names,bytes=p.stat().st_size))
spec=importlib.util.spec_from_file_location('cpu',R/'predictfun/reviews/maker_trace_20261122/tools/cpu_ledger.py');cpu=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu);spent=sum(x['cpu_s'] for x in cpu.ledger())/3600+cpu.EXTRA_H
assert spent+.006+.05<=2 and 3.3107+spent+.006+.05<=5.3,'resource gate'
result=dict(population=88,pairs=pairs,files=files,parquets=meta,total_bytes=sum((R/p).stat().st_size for p in files),task_before=spent,packet_cap_coreh=.006,task_cap=2.,line_cap=5.3,reserve=.05)
assert result['total_bytes']<512*1024*1024 and not (H/'inputs.json').exists();(H/'inputs.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('files','parquets','pairs')},indent=2))
for p in meta:
 if any(n in p['path'] for n in ('panel.parquet','rows.parquet','events.parquet')):print(p['path'],p['columns'])
