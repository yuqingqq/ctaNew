"""Diagnostic: are 'orphan' WS pending increases a second booking of a REST taker fill?

Example (10-23 14:23:36.850, market 3668233): one REST fill, taker sold Up 114.334 sh to a maker bid at 0.99.
Pending showed bids 0.99 = 113.1907 (= REST, matched) AND asks 0.98 = 112.0473 (= the same 114.334 sh at 0.98).
Test over the day: orphan (from pf_ws_pending_offsets) whose implied shares (usd / p) equal a REST taker fill's
amountFilled in that market within +-1 s, at any price. Reports explained USD share, the price offset to the taker's
YES-space limit price, and whether the pending side equals the taker's YES-space side.
Usage: python3 -m predictfun.research.pf_ws_pending_taker 20261023
Writes step1/<day>/pending_taker.json.
"""
import sys, collections
import numpy as np, pandas as pd
from . import pf_io as io

day = sys.argv[1]; S1 = f'{io.OUT}/step1/{day}'
O = pd.read_parquet(f'{S1}/pending_orphans.parquet')
mids = set(O.mid.unique())
TK = collections.defaultdict(list)          # (mid, exec_s) -> [(shares, YES side 0 bid / 1 ask, YES limit tick, exec tick)]
seen = set()
for h in io.hours_of(day, extra_after=1, extra_before=1):
    for recv, params, path, pl in io.rest_rows(h, '/v1/orders/matches'):
        for x in pl.get('data') or []:
            mid = (x.get('market') or {}).get('id')
            if mid not in mids:
                continue
            k = x['transactionHash'] + x['taker']['hash']
            if k in seen:
                continue
            seen.add(k)
            try:
                s = int(io.iso_s(x['executedAt']))
            except (KeyError, TypeError, ValueError):
                continue
            sd, py = io.yes_side(x['taker'])
            pe = int(x['priceExecuted']) / io.E18
            pe_yes = pe if x['taker']['outcome']['indexSet'] == 1 else 1 - pe
            TK[(mid, s)].append((int(x['amountFilled']) / io.E18, 0 if sd == 'bid' else 1, int(round(py * 100)), int(round(pe_yes * 100))))
    io.mem_check(h)
del seen
tot = float(O.usd.sum()); hit = 0.0
off_lim = collections.Counter(); off_exe = collections.Counter(); side_rel = collections.Counter()
for m, t, bs, pt, u in zip(O.mid.values, O.t.values, O.bside.values, O.pt.values, O.usd.values):
    sh = u / (pt / 100.0); b = int(round(t / 1000.0)); best = None
    for k in (-1, 0, 1):
        for a, sd, lim, exe in TK.get((m, b + k), []):
            if abs(a - sh) <= max(0.002 * a, 0.005) and (best is None or abs(pt - exe) < abs(pt - best[2])):
                best = (sd, lim, exe)
    if best is None:
        continue
    hit += u
    off_lim[int(np.clip(pt - best[1], -3, 3))] += u
    off_exe[int(np.clip(pt - best[2], -3, 3))] += u
    side_rel['pending side == taker YES side' if bs == best[0] else 'pending side != taker YES side'] += u
out = dict(day=day, orphan_usd=round(tot), explained_by_taker_fill_shares_usd=round(hit), explained_share=round(hit / tot, 4),
           offset_pt_minus_taker_limit_ticks={str(k): round(v / max(hit, 1e-9), 4) for k, v in sorted(off_lim.items())},
           offset_pt_minus_exec_price_ticks={str(k): round(v / max(hit, 1e-9), 4) for k, v in sorted(off_exe.items())},
           side={k: round(v / max(hit, 1e-9), 4) for k, v in side_rel.items()})
io.write_json(f'{S1}/pending_taker.json', out)
import json; print(json.dumps(out, indent=1))
