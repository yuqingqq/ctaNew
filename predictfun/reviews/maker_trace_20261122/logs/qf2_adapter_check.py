"""Integrator check: posting_opportunity.read_qf2 on queue_front_v2's real Oct 24 pilot output (schema + counts only)."""
import sys
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/reviews/maker_trace_20261122/tools')
import posting_opportunity as PO
import pandas as pd
qd = '/home/yuqing/ctaNew/predictfun/reviews/maker_trace_20261122/v2/chains/queue_front_v2_pilot_20261024_r1'
L = pd.read_csv(qd + '/levels.csv', usecols=['day', 'mid'], dtype=dict(day=str))
mk = sorted({(d, int(m)) for d, m in zip(L['day'], L['mid'])})
D, info = PO.read_qf2(qd, mk, {})
n = sum(len(v) for v in D.values()); nc = sum(int((v['status'] == 'front_candidate').sum()) for v in D.values())
nh = sum(int(((v['status'] == 'front_candidate') & (v['holder'] != '')).sum()) for v in D.values())
print('markets', len(mk), 'rows', n, 'front_candidate', nc, 'with holder', nh, 'info', info)
assert nc == nh == 344, (nc, nh)
