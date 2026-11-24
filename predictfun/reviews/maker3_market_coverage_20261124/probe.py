"""Schema/orientation only for existing maker-trace coverage, no new data."""
import json,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_CPU,(8,9))
import pyarrow.parquet as pq
D=Path('/home/yuqing/ctaNew/data/predict_fun/research/maker_trace_v2_20261123')
for f in ['markets_20261024.parquet','econ/maker_market.parquet']:
 p=D/f;t=pq.read_table(p);print(f,t.schema);print(t.slice(0,3).to_pylist())
for f in ['inventory_20261024.json','extract_20261024.json','econ/econ.json']:
 x=json.loads((D/f).read_text());print(f,'keys',list(x))
 print(json.dumps(x)[:2500])
ids=json.loads((D/'econ/identities.json').read_text());print('TOP3',ids['ex_ante']['top10_shares'][:3])
