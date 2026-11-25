from pathlib import Path
import pyarrow.parquet as pq
R=Path('/home/yuqing/ctaNew')
for fn in ['predictfun/reviews/maker_trace_20261122/v2/chains/take_logic_v2/panel.parquet','data/predict_fun/research/maker_trace_v2_20261123/tob_20261024.parquet']:
 p=R/fn;f=pq.ParquetFile(p);print(fn,'bytes',p.stat().st_size,'rows',f.metadata.num_rows,'schema',str(f.schema_arrow))
