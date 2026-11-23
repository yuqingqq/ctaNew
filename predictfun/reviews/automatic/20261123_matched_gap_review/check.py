"""Independent saved-evidence review; no fit, replay, or new data. Run via PF_SLOT=codex run_light."""
import collections
import hashlib
import json
import math
import os
import sys
from pathlib import Path

os.environ['MT_RUN'] = 'v2'
PIN = Path('/tmp/pf-gap-review-55625bfa')
ROOT = Path('/home/yuqing/ctaNew')
HERE = Path(__file__).resolve().parent
SRC = PIN / 'predictfun/reviews/maker_trace_20261122'
sys.path.insert(0, str(SRC / 'tools'))
import matched_gap as G
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

R = G.build(write=False, n_expected=19, log=lambda *a: None)
saved = SRC / 'v2/matched_gap_r1'
receipt = json.loads((saved / 'receipt.json').read_text())
out = {'target': '55625bfa6ce8cf44002c9cc23b8f4b2d79a82700'}
out['output_reproduction'] = {}
for name in ('per_market', 'closes_by_role'):
    expected = receipt['outputs'][name + '.csv']['sha256']
    actual = hashlib.sha256(R[name].to_csv(index=False).encode()).hexdigest()
    assert actual == expected == sha(saved / (name + '.csv'))
    out['output_reproduction'][name] = {'rows': len(R[name]), 'byte_equal': True}
assert G.X._clean(R['answer']) == json.loads((saved / 'answer.json').read_text())
assert all(v['sha256'] == receipt['inputs'][k]['sha256'] for k, v in R['receipt']['inputs'].items())
assert R['receipt']['sources'] == receipt['sources']
out['inputs_sources_answer_equal'] = True

# Separate deque accounting; no call to the reviewed FIFO/account/combine helpers.
def fifo(legs, queue=None):
    q = collections.deque([] if queue is None else [list(x) for x in queue])
    gross = fees = shares = 0.
    for t, side, amount, price, fee in legs:
        unitfee = fee / amount
        left = amount
        while left > 0 and q and q[0][0] != side:
            s, n, p, uf, birth = q[0]
            size = min(left, n)
            gross += s * size * (price - p)
            fees += size * (uf + unitfee)
            shares += size
            left -= size
            q[0][1] -= size
            if q[0][1] <= 0:
                q.popleft()
        if left > 0:
            q.append([side, left, price, unitfee, t])
    return q, gross, fees, shares

def acct(legs, mark, pre, opening):
    q0, *_ = fifo(pre)
    if sum(x[1] for x in q0) < G.DUST_SH:
        q0.clear()
    init = [[s, q, opening, 0., t] for s, q, p, f, t in q0]
    q, gross, fee, shares = fifo(legs, init)
    residual = sum(s * n * (mark - p) - n * uf for s, n, p, uf, t in q)
    return np.array([gross, fee, shares, gross - fee + residual])

P = G.default_paths()
addrs = G.X.load_top10(P['econ'] / 'identities.json')
cols = list(dict.fromkeys(list(G.FV_COLS) + ['t_ms', 'fee_dy', 'fee_dc', 'fee_type']))
pf = pq.ParquetFile(P['econ'] / 'fills_valued.parquet', read_dictionary=sorted(G.FV_DICT))
groups, _ = G.X.day_groups(pf, G.C.MEASURE_DAYS, 'fills_valued')
T = G.read_selected(pf, sorted({i for d in groups for i, _ in groups[d]}), cols,
                    [m for d, m in R['mks']], addrs, {}, 'independent', G.FV_DICT)
F = T.to_pandas().sort_values(['settle_ms', 'sid', 'px_rank', 'mk_idx', 'role'], kind='stable')
PM = R['per_market']
maxdev = 0.
ncheck = 0
clock = []
fee_drift = collections.defaultdict(float)
ind = {}
for key, mk in R['mks'].items():
    d, m = key
    start, end, mark = mk['start'], mk['end'], mk['end_mark']
    fm = F[(F.mid == m) & (F.status == 'ok')]
    row = {'day': d, 'm': m, 'n_ok': len(fm),
           'at_start': int((fm.settle_ms == start).sum()), 'at_end': int((fm.settle_ms == end).sum()),
           'lo_hi_straddle_start': int(((fm.t_ms < start) & (fm.settle_ms >= start)).sum()),
           'lo_hi_straddle_end': int(((fm.t_ms <= end) & (fm.settle_ms > end)).sum()),
           'n_no_earlier_anchor': int((fm.t_ms == fm.settle_ms).sum()),
           'ours_engine_plus50_after_end': sum(l[0] + 50 > end for l in mk['legs'])}
    for h in (50, 100, 2000):
        row[f'near_end_{h}ms'] = int((abs(fm.settle_ms - end) <= h).sum())
    clock.append(row)
    ours = [(t, s, q, p, fee) for t, s, q, p, role, fee, it, kind in mk['legs']]
    for view in G.VIEWS:
        ind[(d, m, 'ours', view)] = acct(ours, mark, [], math.nan)
        members = []
        for rank, addr in enumerate(addrs):
            a = fm[(fm.addr == addr) & ~fm.self_trade]
            pre = [(r.settle_ms, r.s, r.q, r.p, r.fee_fill) for r in a.itertuples() if r.settle_ms < start]
            legs = [(r.settle_ms, r.s, r.q, r.p, r.fee_fill) for r in a.itertuples() if start <= r.settle_ms <= end]
            v = acct(legs, mark, pre if view == 'carry' else [], R['oms'][key]['mark'])
            ss = fm[(fm.addr == addr) & fm.self_trade & (fm.settle_ms >= start) & (fm.settle_ms <= end)]
            v[-1] += float((ss.s * ss.q * (mark - ss.p) - ss.fee_fill).sum())
            members.append(v)
        for party, ranks in G.MEMBERS.items():
            ind[(d, m, party, view)] = np.sum([members[i] for i in ranks], axis=0)
    for party, ranks in G.MEMBERS.items():
        a = fm[fm.addr.isin([addrs[i] for i in ranks]) & (fm.settle_ms >= start) & (fm.settle_ms <= end)]
        fee_drift[party] += float((a.fee_dy * (mark - a.p)).sum())
for r in PM.itertuples():
    if r.status != 'ok':
        continue
    got = ind[(r.day, r.m, r.party, r.view)]
    want = np.array([r.close_pnl, r.close_fee, r.close_sh, r.total])
    dv = max(abs(got - want))
    assert np.allclose(got, want, rtol=1e-10, atol=1e-7), (r.m, r.party, r.view, got, want)
    maxdev = max(maxdev, dv)
    ncheck += 1
out['independent_fifo'] = {'rows_checked': ncheck, 'max_abs_deviation': maxdev}
out['clock_boundary'] = clock
out['fee_token_END_minus_fill_price_USD'] = dict(fee_drift)
out['roles'] = R['closes_by_role'].query("view == 'carry' and entry_kind == 'all'").to_dict('records')
out['frozen_readings'] = {}
for party in G.COMPARE:
    rows = PM[(PM.party == party) & (PM.view == 'carry') & PM.both_active.eq(True)]
    readings = []
    for r in rows.itertuples():
        o, t = r.o_c_sh, r.t_c_sh
        expected = ('mixed_unsupported' if not (math.isfinite(o) and math.isfinite(t)) else
                    'our_turnover_loses' if o < 0 < t else
                    'exposure_gap' if ((o >= 0 and t >= 0) or (o < 0 and t < 0)) and abs(r.d_exposure) > abs(r.d_turnover) else
                    'both_adverse' if o < 0 and t < 0 else 'mixed_unsupported')
        if len(rows) < 5:
            expected = 'mixed_unsupported'
        assert r.reading == expected
        readings.append(expected)
    out['frozen_readings'][party] = dict(collections.Counter(readings))
old = pd.read_csv(SRC / 'v2/matched_gap/per_market.csv', dtype={'day': str}, float_precision='round_trip')
out['old_fix_changed_rows'] = []
for r, o in zip(PM.to_dict('records'), old.to_dict('records')):
    changed = [c for c in old if not ((pd.isna(r[c]) and pd.isna(o[c])) or r[c] == o[c])]
    if changed:
        out['old_fix_changed_rows'].append({'key': [r['day'], r['m'], r['party'], r['view']], 'fields': changed,
                                            'total_delta': r['total'] - o['total']})
# Positive/negative controls on this independent instrument.
assert np.allclose(acct([(0, 1, 2, .4, 0.), (1, -1, 2, .5, .02)], .6, [], .5), [.2, .02, 2, .18])
assert not np.allclose(acct([], .6, [(0, 1, 2, .4, 0.)], .5), acct([], .6, [], .5))
assert not np.isclose(.2 - .02, .2 + .02)
out['independent_controls'] = ['known profitable close', 'missing carried-only position differs', 'fee sign differs']
out['receipt_counts'] = R['receipt']['counts']
(HERE / 'evidence.json').write_text(json.dumps(G.X._clean(out), indent=2) + '\n')
print(json.dumps({k: out[k] for k in ['output_reproduction', 'inputs_sources_answer_equal', 'independent_fifo', 'frozen_readings', 'old_fix_changed_rows', 'fee_token_END_minus_fill_price_USD']}, indent=2))
