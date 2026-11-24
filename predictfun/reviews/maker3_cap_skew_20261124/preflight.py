from pathlib import Path
import pyarrow.parquet as pq
R=Path('/home/yuqing/ctaNew')
C=R/'predictfun/reviews/maker_trace_20261122/v2/chains'
for p in [C/'posting_opportunity/rows.parquet',C/'take_logic_v2/panel.parquet',C/'take_logic_v2/events.parquet']:
 print(str(p),pq.read_schema(p).names,flush=True)
