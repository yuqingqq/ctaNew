"""Diagnostic companion to pf_ws_pending_diag: at which price do 'orphan' WS increases sit?

Orphan = live settlementsPending increase at (market, YES price) with no REST maker leg at that price in
seconds b-1..b+1 (b = increase time rounded to the second). For orphans that have a REST leg in the same
market within +-1 s at another price, report the USD-weighted offset to the nearest such price, the share
at the complement price (100 - p), and the book side / REST leg side / outcome mix.
Usage: python3 -m predictfun.research.pf_ws_pending_offsets 20261023
Writes step1/<day>/pending_offsets.json and pending_orphans.parquet.
"""
import sys, collections
import numpy as np, pandas as pd
from . import pf_io as io

day = sys.argv[1]; S1 = f'{io.OUT}/step1/{day}'
reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'we'])
win = {int(m): (int(ws) * 1000 - 2000, int(we) * 1000 + 3000) for m, ws, we in zip(reg.mid, reg.ws, reg.we)}
live_ms = {int(m): (int(ws) * 1000, int(we) * 1000) for m, ws, we in zip(reg.mid, reg.ws, reg.we)}
L = pd.read_parquet(f'{S1}/legs_timed.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 'taker_index', 'taker_quote'])
L['usd'] = L.shares * L.ptick / 100.0
AT = collections.defaultdict(float)                 # (mid, pt, s) -> USD
PR = collections.defaultdict(lambda: collections.defaultdict(float))   # (mid, s) -> pt -> USD
SD = collections.defaultdict(float)                 # (mid, pt, s, leg side) -> USD
for m, s, sd, p, u in zip(L.mid.values, L.exec_s.values, L.side.values, L.ptick.values, L.usd.values):
    AT[(m, p, s)] += u; PR[(m, s)][p] += u; SD[(m, p, s, sd)] += u

prev = {}
orph = []          # (mid, t, book side, pt, usd)
for h in io.hours_of(day, extra_after=1, extra_before=1):
    for recv, d in io.ws_books(h):
        mid = d.get('marketId'); w = win.get(mid); t = d['updateTimestampMs']
        if w is None or not (w[0] <= t <= w[1]):
            continue
        cur = {}
        for si, key in ((0, 'bids'), (1, 'asks')):
            for p, v in (d.get('settlementsPending') or {}).get(key) or []:
                k = (si, int(round(float(p) * 100))); cur[k] = cur.get(k, 0.0) + float(v)
        old = prev.get(mid)
        if old is not None and live_ms[mid][0] <= t < live_ms[mid][1]:
            bk = int(round(t / 1000.0))
            for k, v in cur.items():
                dl = v - old.get(k, 0.0)
                if dl > 1e-9 and all(AT.get((mid, k[1], bk + j), 0.0) <= 1e-9 for j in (-1, 0, 1)):
                    orph.append((mid, t, k[0], k[1], dl))
        prev[mid] = cur
    io.mem_check(h)

O = pd.DataFrame(orph, columns=['mid', 't', 'bside', 'pt', 'usd'])
O.to_parquet(f'{S1}/pending_orphans.parquet', index=False)
off = collections.Counter(); comp = 0.0; none = 0.0; tot = float(O.usd.sum())
by_bside = O.groupby('bside').usd.sum()
pt_hist = collections.Counter()
for m, t, bs, p, u in zip(O.mid.values, O.t.values, O.bside.values, O.pt.values, O.usd.values):
    bk = int(round(t / 1000.0)); pts = collections.defaultdict(float)
    for j in (-1, 0, 1):
        for q, v in PR.get((m, bk + j), {}).items():
            pts[q] += v
    pt_hist[min(p, 100 - p) // 5 * 5] += u
    if not pts:
        none += u; continue
    if (100 - p) in pts:
        comp += u
    near = min(pts, key=lambda q: (abs(q - p), -pts[q]))
    off[int(np.clip(p - near, -6, 6))] += u
out = dict(day=day, n_orphans=int(len(O)), orphan_usd=round(tot),
           by_book_side={('bids' if k == 0 else 'asks'): round(float(v)) for k, v in by_bside.items()},
           no_rest_leg_in_market_pm1s_usd=round(none),
           rest_leg_at_complement_price_usd=round(comp),
           offset_to_nearest_rest_price_ticks_usd={str(k): round(v) for k, v in sorted(off.items())},
           orphan_usd_by_distance_from_edge_c={f'{k}-{k + 4}': round(v) for k, v in sorted(pt_hist.items())},
           orphan_size_usd_pct=dict(zip(['p10', 'p50', 'p90', 'p99'], np.percentile(O.usd, [10, 50, 90, 99]).round(2).tolist())))
io.write_json(f'{S1}/pending_offsets.json', out)
import json; print(json.dumps(out, indent=1))
