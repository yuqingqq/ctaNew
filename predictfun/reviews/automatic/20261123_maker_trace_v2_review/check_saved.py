"""Independent, bounded saved-data audit; no fitting, tape replay, or output mutation."""
import os
os.environ['MT_RUN'] = 'v2'
import sys, json, hashlib, time
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PIN = Path('/tmp/pf-mt-review-0e21193f')
ROOT = Path('/home/yuqing/ctaNew')
OUT = Path(__file__).resolve().parent
REC = PIN / 'predictfun/reviews/maker_trace_20261122/v2'
sys.path.insert(0, str(PIN / 'predictfun/research'))
sys.path.insert(0, str(PIN / 'predictfun/reviews/maker_trace_20261122/tools'))
import closes as C
T0 = time.process_time()

def sha(p):
    with open(p, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

receipt = json.loads((REC / 'closes/closes_receipt.json').read_text())
hashes = {}
for k, v in receipt['inputs'].items():
    hashes['input:' + k] = sha(v['path']) == v['sha256']
data = ROOT / 'data/predict_fun/research/maker_trace_v2_20261123/econ'
addrs = C.load_top10(data / 'identities.json')
pf = pq.ParquetFile(data / 'fills_valued.parquet')
days = ('20261024', '20261025', '20261026', '20261027')
groups, _ = C.day_groups(pf, days, 'review')
cs = pd.read_csv(REC / 'closes/closes_summary.csv', float_precision='round_trip')
rs = pd.read_csv(REC / 'closes/residual_summary.csv', float_precision='round_trip')
cols = ('day','mid','sid','mk_idx','px_rank','role','addr','s','q','p','settle_ms','self_trade','status','st_res','fee_fill')
dic = ('day','role','addr','status','st_res')
fifo = defaultdict(lambda: np.zeros(3))  # sh, gross, fee
inventory = []
day_counts = {}
gaps = []
for d in days:
    t = C.read_rows(pf, groups[d], d, cols, dic, addrs, {}, 'review')
    f = t.to_pandas()
    del t
    # Exclude an entire top-3 pair with an excluded fill, as decision.build_pairs does.
    bad_mid = f.loc[(f.status != 'ok') | (f.settle_ms < 0), ['addr','mid']].drop_duplicates()
    excluded = set(map(tuple, bad_mid.to_numpy()))
    f = f[(f.status == 'ok') & ~f.self_trade].copy()
    f['ri'] = f.role.map({'maker':0,'taker':1})
    f = f.sort_values(['addr','mid','settle_ms','sid','px_rank','mk_idx','ri'],kind='stable')
    day_counts[d] = len(f)
    for (addr,mid), g in f.groupby(['addr','mid'], sort=False, observed=True):
        rank = addrs.index(addr) + 1
        qfifo = deque()
        pos = peak = 0.
        for row in g.itertuples(index=False):
            pos += row.s * row.q
            peak = max(peak,abs(pos))
            rem = row.q
            while rem > 1e-8 and qfifo and qfifo[0][0] != row.s:
                old = qfifo[0]
                sh = min(rem,old[1])
                gross = sh * old[0] * (row.p - old[2])
                fee = sh * (old[3] + row.fee_fill / row.q)
                if row.st_res in ('ok','tie'):
                    fifo[(rank,d,old[4]*2+row.ri)] += [sh,gross,fee]
                old[1] -= sh
                rem -= sh
                if old[1] <= 1e-8: qfifo.popleft()
            if rem > 1e-8:
                qfifo.append([row.s,rem,row.p,row.fee_fill/row.q,row.ri])
        expected = rs[(rs.scope == f'rank{rank}') & (rs.day.astype(str) == d)].iloc[0]
        assert peak <= expected.max_abs_pos_max + 1e-5
        if rank <= 3 and (addr,mid) not in excluded:
            a = g.assign(sq=g.s*g.q).groupby(['sid','role'], sort=False, observed=True).agg(settle=('settle_ms','first'), sq=('sq','sum'), s=('s','first'))
            a = a.reset_index().sort_values('settle',kind='stable')
            cp = a.sq.cumsum().to_numpy()
            st = a.settle.to_numpy()
            j = np.searchsorted(st,st,side='left')-1
            p = np.where(j>=0,cp[np.maximum(j,0)],0.)
            for role in ('maker','taker'):
                mask = (a.role.to_numpy()==role) & (abs(p)>1e-9)
                direction = a.s.to_numpy()*p<0
                actual = abs(p+a.sq.to_numpy())<abs(p)-1e-9
                inventory.append(dict(day=d,rank=rank,role=role,n=int(mask.sum()),direction=int((mask&direction).sum()),
                                      actual=int((mask&actual).sum()),overshoot=int((mask&direction&~actual).sum())))
    del f
    print('day done', d, 'cpu', round(time.process_time()-T0,2), flush=True)

for row in cs.itertuples(index=False):
    if row.day == 'all' or row.scope in ('top3','top10') or row.role_pair == 'all': continue
    actual = fifo.get((int(row.rank),str(row.day),C.PAIRS.index(row.role_pair)),np.zeros(3))
    expected = np.array([row.sh,row.gross,row.fee_fill])
    gaps.append(float(np.max(abs(actual-expected))))
assert max(gaps)<1e-5, max(gaps)
inv = pd.DataFrame(inventory).groupby(['rank','role'],as_index=False)[['n','direction','actual','overshoot']].sum()
for k in ('direction','actual','overshoot'): inv[k+'_rate']=inv[k]/inv.n
published = pd.read_csv(REC/'decision/inventory.csv',float_precision='round_trip')
ivcmp = []
for row in inv.itertuples(index=False):
    pub = published[(published.scope == f'rank{row.rank}') & (published.day.astype(str)=='all') &
                    (published.role==row.role) & (published.pos_bucket=='all')].iloc[0]
    ivcmp.append(dict(rank=row.rank,role=row.role,our_n=row.n,published_n=int(pub.n),our_direction=row.direction,
                      published_direction=int(pub.n_reduce),actual=row.actual,overshoot=row.overshoot))

# Report-level economics: no refit, all role-pair summaries and fee components.
headline = cs[(cs.scope=='top10') & (cs.day.astype(str)=='all')].to_dict('records')
day_role = cs[(cs.scope=='top10') & (cs.day.astype(str)!='all') & (cs.role_pair!='all')].to_dict('records')
out = dict(target='0e21193f2d403d08bb5172aaf727a6a93110e8a9',input_hashes=hashes,
           fifo_rows_by_day=day_counts,fifo_max_summary_abs_gap=max(gaps),inventory=ivcmp,
           top10_role_pairs=headline,top10_daily_role_pairs=day_role,cpu_s=time.process_time()-T0)
assert all(hashes.values())
(OUT/'saved_evidence.json').write_text(json.dumps(out,indent=2,default=str)+'\n')
print(json.dumps(dict(cpu_s=out['cpu_s'],fifo_max_summary_abs_gap=max(gaps),inventory=ivcmp),indent=2))
