"""Verify the named fill against canonical market records and native book.
Diagnostic only: no interpretation of snapshot differences as an exact match.
"""
import json,hashlib
from pathlib import Path
import pyarrow.parquet as pq
H=Path(__file__).resolve().parent
MT=Path('/home/yuqing/ctaNew/data/predict_fun/research/maker_trace_v2_20261123')
e=json.loads((H/'fill_clock_evidence.json').read_text());sids=[f['sid'] for f in e['fills']]
legs=pq.read_table(MT/'legs_20261026.parquet',filters=[('sid','in',sids)]).to_pandas()
recs=pq.read_table(MT/'records_20261026.parquet',filters=[('sid','in',sids)]).to_pandas()
print('CANONICAL_LEGS',legs.to_json(orient='records'));print('CANONICAL_RECORDS',recs.to_json(orient='records'))
print('SCHEMAS',list(legs.columns),list(recs.columns))
