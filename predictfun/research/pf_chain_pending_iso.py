"""Which order entries does WS settlementsPending list? Isolated settlements only.

Isolated = a settlement tx in one of our live markets with no other settlement in that market whose block is within
+-1.5 s. For it, expected entries: each maker order at its price (filled shares x price) and the taker order at its
signed LIMIT price (filled shares x limit), YES space. Observed: WS pending increases in that market with
t in [block - 1.5 s, block + 0.2 s]. Reports, separately for takers whose limit equals the fill price and those whose
limit is better than... worse than the fill (slippage room), how often each expected entry appears, on which side,
and how much WS increase USD in the window is left unexplained.
Usage: python3 -m predictfun.research.pf_chain_pending_iso 20261023 1400 1415
Writes onchain/<slice>/pending_iso.json.
"""
import sys, json, time, collections
import numpy as np, pandas as pd
from . import pf_io as io

day, h0, h1 = sys.argv[1], sys.argv[2], sys.argv[3]
DIR = f'{io.OUT}/onchain/{day}_{h0}_{h1}'
FS = json.load(open(f'{DIR}/fetch_summary.json')); T0, T1 = FS['t0'], FS['t1']
O = pd.read_parquet(f'{DIR}/orders.parquet')
O = O[np.isfinite(O.shares)]
reg = pd.read_parquet(f'{io.OUT}/step1/{day}/registry.parquet', columns=['mid', 'ws', 'we'])
live = {int(m): (int(ws) * 1000, int(we) * 1000) for m, ws, we in zip(reg.mid, reg.ws, reg.we)}
O = O[[m in live and live[m][0] <= b < live[m][1] for m, b in zip(O.mid, O.block_ms)]]
O = O[(O.block_ms >= T0 * 1000) & (O.block_ms < T1 * 1000)]
txb = O.groupby('tx').agg(mid=('mid', 'first'), block_ms=('block_ms', 'first')).reset_index().sort_values(['mid', 'block_ms'])
iso = []
for mid, g in txb.groupby('mid'):
    b = g.block_ms.values; tx = g.tx.values
    for i in range(len(b)):
        if (i == 0 or b[i] - b[i - 1] > 1500) and (i == len(b) - 1 or b[i + 1] - b[i] > 1500):
            iso.append(tx[i])
iso = set(iso)

hours = sorted({time.strftime('%Y%m%d_%H', time.gmtime(t)) for t in (T0 - 60, T0, T1 - 1, T1 + 60)})
prev = {}; INC = collections.defaultdict(list)
for h in hours:
    for recv, d in io.ws_books(h):
        mid = d.get('marketId'); t = d['updateTimestampMs']; w_ = live.get(mid)
        if w_ is None or not (w_[0] <= t < w_[1]):
            continue
        cur = {}
        for si, key in ((0, 'bids'), (1, 'asks')):
            for p, v in (d.get('settlementsPending') or {}).get(key) or []:
                k = (si, int(round(float(p) * 100))); cur[k] = cur.get(k, 0.0) + float(v)
        old = prev.get(mid)
        if old is not None:
            for k, v in cur.items():
                dl = v - old.get(k, 0.0)
                if dl > 1e-9:
                    INC[mid].append((t, k[0], k[1], dl))
        prev[mid] = cur
def near(x, y): return abs(x - y) <= max(0.002 * y, 0.005)

res = collections.defaultdict(collections.Counter); left = collections.defaultdict(float); seen_usd = collections.defaultdict(float)
for tx, g in O[O.tx.isin(iso)].groupby('tx'):
    mid = g.mid.iloc[0]; b = g.block_ms.iloc[0]
    tk = g[g.role == 'taker'].iloc[0]; mk = g[g.role == 'maker']
    grp = 'limit = fill price' if tk.limit_tick_yes == tk.exec_tick_yes else 'limit has slippage room'
    inc = [x for x in INC.get(mid, []) if b - 1500 <= x[0] <= b + 200]
    used = [False] * len(inc)
    def find(side, pt, val):
        for j, (t, s, p, u) in enumerate(inc):
            if not used[j] and p == pt and near(u, val):
                used[j] = True
                return 'same side' if s == side else 'other side'
        return None
    res[grp]['n_tx'] += 1
    ok_m = all(find(r.yes_side, int(r.limit_tick_yes), r.shares * r.limit_tick_yes / 100) for r in mk.itertuples())
    res[grp]['all maker entries found'] += ok_m
    f = find(int(tk.yes_side), int(tk.limit_tick_yes), tk.shares * tk.limit_tick_yes / 100)
    res[grp][f'taker entry at limit: {f or "not found"}'] += 1
    seen_usd[grp] += sum(x[3] for x in inc)
    left[grp] += sum(x[3] for j, x in enumerate(inc) if not used[j])
out = dict(slice=f'{day} {h0}-{h1}', isolated_tx=len(iso),
           by_taker_type={k: dict(v, ws_inc_usd_in_windows=round(seen_usd[k], 2), unexplained_usd=round(left[k], 2)) for k, v in res.items()})
io.write_json(f'{DIR}/pending_iso.json', out)
print(json.dumps(out, indent=1))
