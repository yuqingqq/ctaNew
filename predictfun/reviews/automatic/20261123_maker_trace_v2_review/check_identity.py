"""Follow up the failed whole-file equality assumption: compare economic columns, keep lifecycle changes explicit."""
import json,hashlib,time
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
ROOT=Path('/home/yuqing/ctaNew')
PIN=Path('/tmp/pf-mt-review-0e21193f')
OUT=Path(__file__).resolve().parent
REC=PIN/'predictfun/reviews/maker_trace_20261122/v2'
V1=ROOT/'data/predict_fun/research/maker_trace_v1_20261122/econ'
V2=ROOT/'data/predict_fun/research/maker_trace_v2_20261123/econ'
def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
t0=time.process_time()
eq={n:sha(V1/n)==sha(V2/n) for n in ('fills_valued.parquet','maker_market.parquet','identities.json')}
a,b=pq.ParquetFile(V1/'fills_valued.parquet'),pq.ParquetFile(V2/'fills_valued.parquet')
assert a.schema_arrow==b.schema_arrow
assert a.metadata.num_rows==b.metadata.num_rows
# Exact equality after dictionary decoding; all columns, streamed. Both are ordered by the same pipeline.
diff={k:0 for k in a.schema_arrow.names}
rows=0
for x,y in zip(a.iter_batches(batch_size=100000),b.iter_batches(batch_size=100000)):
    assert x.num_rows==y.num_rows
    rows+=x.num_rows
    for name in diff:
        u,v=x.column(name),y.column(name)
        if pa.types.is_dictionary(u.type): u,v=pc.cast(u,pa.string()),pc.cast(v,pa.string())
        if pa.types.is_floating(u.type):
            same=pc.or_kleene(pc.equal(u,v),pc.and_kleene(pc.is_nan(u),pc.is_nan(v)))
        else:same=pc.equal(u,v)
        same=pc.or_kleene(same,pc.and_(pc.is_null(u),pc.is_null(v)))
        diff[name]+=pc.sum(pc.cast(pc.invert(pc.fill_null(same,False)),pa.int64())).as_py()
    if time.process_time()-t0>65:raise RuntimeError('bounded review CPU guard')
assert rows==a.metadata.num_rows
changed={k:v for k,v in diff.items() if v}
assert set(changed)<=set(('life_rep','life_exh','life_front','life_status')),changed
cr=json.loads((REC/'closes/closes_receipt.json').read_text())
output_hashes={}
for name,v in cr['outputs'].items():
    p=REC/'closes'/name
    if not p.exists():p=ROOT/'predictfun/reviews/maker_trace_20261122/v2/closes'/name
    output_hashes[name]=sha(p)==v['sha256']
assert all(output_hashes.values())
# Reconcile the six saved narrative claims for FIFO and concentration at source tables.
report=dict(target='0e21193f2d403d08bb5172aaf727a6a93110e8a9',byte_equal=eq,
            fills_rows_compared=rows,changed_fields=changed,economic_columns_exact=True,
            closes_output_hashes=output_hashes,cpu_s=time.process_time()-t0,
            preceding_failure='controls_check.log: inappropriate whole-file byte equality for lifecycle-enriched fills, after all synthetic controls passed')
(OUT/'identity_evidence.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
