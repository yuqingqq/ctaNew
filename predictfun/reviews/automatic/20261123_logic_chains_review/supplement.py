"""Small saved-table consistency, status and budget checks; no full builder run."""
import hashlib
import json
import os
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
os.environ['MT_RUN']='v2'
R=Path('/home/yuqing/ctaNew'); P=Path('/tmp/pf-chains-review-8d68808d'); H=Path(__file__).resolve().parent
B=R/'predictfun/reviews/maker_trace_20261122'
sys.path.insert(0,str(P/'predictfun/reviews/maker_trace_20261122/tools'))
import chain_take as T
import chain_pull as U
import matched_gap as G
import cpu_ledger as CPU
def read(p): return pd.read_csv(p,float_precision='round_trip',dtype={'day':str})
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(x): return G.X._clean(x)
out={}
take=B/'v2/chains/take'
tk,ct,act,sm=[read(take/(n+'.csv')) for n in ('takes','take_controls','actions','summary')]
recomputed=T.summary_table(tk,ct,act,T.C.MEASURE_DAYS)
pd.testing.assert_frame_equal(recomputed,sm,check_dtype=False,check_exact=True)
out['take_summary_reproduces_saved_inputs']=True
out['take_output_hashes_review_snapshot']={p.name:sha(p) for p in take.glob('*.csv')}
out['take_status_counts']=tk.status.value_counts().to_dict()
out['take_classes_excluded_before']=tk[tk.n_excl_before.gt(0)].take_class.value_counts().to_dict()
cx=U.coverage_mask(np.array([0,200]),np.array([200,400]))
out['presence_relay_falsifier']={'union_mask':cx(None,np.array([0,100,200,300,400])).tolist(),
                               'one_order_covers_full_0_400':False}
post=read(B/'v2/chains/post_fill/orders.csv')
life=R/'data/predict_fun/research/maker_trace_v2_20261123/life'
part=[]
for (d,m),g in post.groupby(['day','mid']):
    f=pq.read_table(life/f'{d}_{m}_orders.parquet',columns=['scenario','mk_hash','post_class']).to_pandas()
    f=f[f.scenario.eq('S_NOPRIO')].rename(columns={'post_class':'life_class'})
    part.append(g.merge(f[['mk_hash','life_class']],on='mk_hash',validate='one_to_one'))
q=pd.concat(part); short=q.post_class.isin(['late_post','short_rest'])
out['short_with_lifecycle_constrained_and_status_ok']=int((short & q.life_class.eq('constrained') & q.status.eq('ok')).sum())
out['short_constrained_statuses']=q[short & q.life_class.eq('constrained')].status.value_counts().to_dict()
out['short_conflict_flags_count']=int((short & q.status.str.contains('capacity_bounds_void|clock_capacity_conflict|post_after_first_fill')).sum())
s=read(B/'v2/chains/pull/summary.csv')
mask=s['view'].eq('S_FIFO') & s['scope'].eq('pooled') & s['D'].astype(str).eq('250') & s['control'].eq('fill_anchored')
out['fill_anchored_by_day']=s[mask & s['feature'].isin(['btc_against_1s_some','inv_built_5s_some'])].to_dict('records')
# Read only selected BOOK/top-three columns from already accepted fills_valued.
root=R/'data/predict_fun/research/maker_trace_v2_20261123'
fp=pq.ParquetFile(root/'econ/fills_valued.parquet',read_dictionary=['addr','status'])
gr,_=G.X.day_groups(fp,T.C.MEASURE_DAYS,'fills_valued')
ids=G.X.load_top10(root/'econ/identities.json')[:3]
cols=['day','mid','addr','sid','role','s','q','settle_ms','status','src_status']
tab=G.read_selected(fp,sorted({i for d in gr for i,_ in gr[d]}),cols,list(post.mid.unique()),ids,{},'status',{'addr','status'})
f=tab.to_pandas(); ex=f[f.status.ne('ok')]
out['excluded_book_top3_legs']={'n':len(ex),'statuses':ex.status.value_counts().to_dict(),
                              'source_statuses':ex.src_status.value_counts().to_dict(),
                              'finite_positive_qty':int((np.isfinite(ex.q)&ex.q.gt(0)).sum())}
spent=sum(r['cpu_s'] for r in CPU.ledger())/3600+CPU.EXTRA_H+.010 # prior 6ddad3533 allowance absent from old helper
packet_tags=re.compile(r'^(mt_(?:ctx|oe|pf|pull|take)_|mt_x_(?:oe|pf|pull|take)_v2$)')
packet=sum(r['cpu_s'] for r in CPU.ledger() if packet_tags.search(r['tag']))/3600
out['budget']={'task_before_current_review_h':spent,'line_before_current_review_h':CPU.LINE_BEFORE+spent,
               'packet_dev_h':packet,'current_review_charge_h':.010,'task_cap_h':1.85,'line_cap_h':5.25,'packet_cap_h':.11}
(H/'supplement.json').write_text(json.dumps(clean(out),indent=2)+'\n')
print(json.dumps(clean(out),indent=2))
