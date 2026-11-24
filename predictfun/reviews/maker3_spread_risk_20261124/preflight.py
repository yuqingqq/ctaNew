import gzip,json
from pathlib import Path
import pyarrow.parquet as pq
R=Path('/home/yuqing/ctaNew');D=R/'data/predict_fun/research';V=D/'maker_trace_v2_20261123'
for p in [V/'econ/fills_valued.parquet',V/'tob_20261026.parquet',V/'life/20261026_3789484_fills.parquet',V/'depth/20261026_3789484_frames.parquet']:
 if p.exists():print(p,pq.read_schema(p).names)
old=json.loads(gzip.decompress((R/'predictfun/reviews/maker_dashboard_20261124/data.json.gz').read_bytes()));m=old['cases'][0]
print('dashboard fields',list(m));print('fill fields',list(m['makers'][0]));print('book example',m['book'][:1])
print('tob paths',[p.name for p in V.glob('*tob*')]);print('econ paths',[p.name for p in (V/'econ').glob('*')])
