from pathlib import Path
import gzip,csv
import pyarrow.parquet as pq
D=Path('/home/yuqing/ctaNew/data/predict_fun/research/maker_trace_v2_20261123')
for f in ['legs_20261024.parquet','tob_20261024.parquet']:
 p=pq.ParquetFile(D/f);print(f,p.metadata.num_rows,p.schema_arrow)
with gzip.open('/home/yuqing/ctaNew/predictfun/reviews/maker_timing_estimates_20261124/outputs/post_estimates.csv.gz','rt') as f:
 r=csv.DictReader(f);print('post columns',r.fieldnames);print('example',next(r))
