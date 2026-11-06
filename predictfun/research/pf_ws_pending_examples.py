"""Worked examples: REST trades vs WS settlementsPending at one price, for one hour of a read day.

Finds, in live windows (quietest span of each kind): (A) one trade -> one increase of the same value; (B) two or more trades -> one increase
equal to their sum; (C) netting: the old pending balance at a price is cleared in the same snapshot where a new
trade appears, so the change is new - old (and the old balance matches an earlier REST trade); (D) an increase
with no REST trade at that price within +-1 s, with the market's REST trades at other prices.
Prints a timeline per example: REST legs at that price (executedAt second, YES-space side, shares, USD at the
YES price) and every WS snapshot where that price's pending value changed.
Usage: python3 -m predictfun.research.pf_ws_pending_examples 20261023 14
"""
import sys, collections, datetime as dt
import numpy as np, pandas as pd
from . import pf_io as io

day, hh = sys.argv[1], sys.argv[2]; S1 = f'{io.OUT}/step1/{day}'
reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'we'])
live = {int(m): (int(ws) * 1000, int(we) * 1000) for m, ws, we in zip(reg.mid, reg.ws, reg.we)}
ser = collections.defaultdict(list)        # (mid, pt) -> [(t, bid value, ask value)] at every frame
prev = {}
for recv, d in io.ws_books(f'{day}_{hh}'):
    mid = d.get('marketId'); t = d['updateTimestampMs']; w = live.get(mid)
    if w is None or not (w[0] <= t < w[1]):
        continue
    cur = collections.defaultdict(lambda: [0.0, 0.0])
    for si, key in ((0, 'bids'), (1, 'asks')):
        for p, v in (d.get('settlementsPending') or {}).get(key) or []:
            cur[int(round(float(p) * 100))][si] += float(v)
    old = prev.get(mid, {})
    for pt in set(cur) | set(old):
        a = tuple(cur.get(pt, (0.0, 0.0))); b = tuple(old.get(pt, (0.0, 0.0)))
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1e-9:
            ser[(mid, pt)].append((t, a[0], a[1], b[0], b[1]))
    prev[mid] = {k: tuple(v) for k, v in cur.items()}

L = pd.read_parquet(f'{S1}/legs_timed.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'shares', 't_ms', 'how'])
t0 = dt.datetime.strptime(f'{day}{hh}', '%Y%m%d%H').replace(tzinfo=dt.timezone.utc).timestamp()
L = L[(L.exec_s >= t0) & (L.exec_s < t0 + 3600) & L.mid.isin(live)].copy()
L['usd'] = L.shares * L.ptick / 100.0
LG = {k: g for k, g in L.groupby(['mid', 'ptick'])}

def ts(ms): return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime('%H:%M:%S.%f')[:-3]
def near(x, y): return abs(x - y) <= max(0.011 * abs(y), 0.01)

def show(title, mid, pt, t_lo, t_hi):
    print(f'\n=== {title}: market {mid}, YES price {pt}c ===')
    g = LG.get((mid, pt))
    rows = g[(g.exec_s * 1000 >= t_lo - 1000) & (g.exec_s * 1000 <= t_hi + 1000)] if g is not None else []
    print('REST trades at this price (executedAt is rounded to the second):')
    for r in rows.itertuples() if len(rows) else []:
        print(f'  {ts(r.exec_s * 1000)[:-4]}  maker {"bid" if r.side == 0 else "ask"}  {r.shares:9.4f} sh  x {pt / 100:.2f} = ${r.usd:8.4f}')
    print('WS snapshots where pending at this price changed (bids side / asks side, $):')
    for t, b, a, pb, pa in ser[(mid, pt)]:
        if t_lo - 1000 <= t <= t_hi + 1000:
            print(f'  {ts(t)}  bids {pb:8.4f} -> {b:8.4f} ({b - pb:+8.4f})   asks {pa:8.4f} -> {a:8.4f} ({a - pa:+8.4f})')

def tight(x, y): return abs(x - y) <= max(0.002 * abs(y), 0.005)

cands = collections.defaultdict(list)
for (mid, pt), evs in ser.items():
    g = LG.get((mid, pt))
    if g is None:
        continue
    secs = collections.defaultdict(list)
    for r in g.itertuples():
        secs[r.exec_s].append(r.usd)
    et = np.array([e[0] for e in evs])
    for i, (t, b, a, pb, pa) in enumerate(evs):
        s = int(round(t / 1000))
        busy = sum(len(secs.get(s + k, [])) for k in range(-2, 3)) + int(((et >= t - 2000) & (et <= t + 2000)).sum())
        cand = [u for k in (-1, 0, 1) for u in secs.get(s + k, [])]
        same = secs.get(s, [])
        for side, v, pv in ((0, b, pb), (1, a, pa)):
            dlt = v - pv
            if len(cand) == 1 and pv == 0 and tight(dlt, cand[0]):
                cands['A'].append((busy, mid, pt, t, t))
            if len(same) >= 2 and len(cand) == len(same) and pv == 0 and tight(dlt, sum(same)) \
                    and not any(tight(dlt, u) for u in same):
                cands['B'].append((busy, mid, pt, t, t))
            # C: earlier snapshot took this side 0 -> pv (an earlier trade); now pv -> v where v = a new trade
            if pv > 0.5 and v > 0.5 and not tight(v, pv) and any(tight(v, u) for u in same) \
                    and not any(tight(dlt, u) for u in cand):
                for e in reversed(evs[:i]):
                    ev_v, ev_pv = (e[1], e[3]) if side == 0 else (e[2], e[4])
                    if t - e[0] > 5000:
                        break
                    if tight(ev_v, pv):
                        if ev_pv == 0 and any(tight(pv, u) for k in range(-5, 1) for u in secs.get(int(round(e[0] / 1000)) + k, [])):
                            cands['C'].append((dlt >= 0, busy, mid, pt, e[0], t))
                        break
found = {}
for k in 'AB':
    if cands[k]:
        found[k] = min(cands[k])[1:]
if cands['C']:
    found['C'] = min(cands['C'])[2:]
titles = dict(A='A. one trade -> one increase (matched to the ms)', B='B. several trades -> one increase (merged)',
              C='C. settlement and new trade in the same snapshot (netted)')
for k in 'ABC':
    if k in found:
        show(titles[k], *found[k])
    else:
        print(f'\n=== {titles[k]}: no example found in this hour ===')

# D: an increase with no REST trade at that price within +-1 s (an "orphan"), in the quietest span
LM = {m: g for m, g in L.groupby('mid')}
for (mid, pt), evs in ser.items():
    g = LG.get((mid, pt)); gm = LM.get(mid)
    have = set() if g is None else set(g.exec_s.values)
    for t, b, a, pb, pa in evs:
        s_ = int(round(t / 1000))
        if max(b - pb, a - pa) > 0.5 and not any(s_ + k in have for k in (-1, 0, 1)) and gm is not None:
            near_m = gm[(gm.exec_s >= s_ - 1) & (gm.exec_s <= s_ + 1)]
            if len(near_m):
                cands['D'].append((len(near_m), mid, pt, t))
if cands['D']:
    n_, mid, pt, t = min(cands['D'])
    print(f'\n=== D. WS increase with no REST trade at that price (open question): market {mid}, YES price {pt}c ===')
    gm = LM[mid]; s_ = int(round(t / 1000))
    print('REST trades in this market, any price, within +-1 s:')
    for r in gm[(gm.exec_s >= s_ - 1) & (gm.exec_s <= s_ + 1)].itertuples():
        print(f'  {ts(r.exec_s * 1000)[:-4]}  maker {"bid" if r.side == 0 else "ask"}  {r.ptick}c  {r.shares:9.4f} sh = ${r.usd:8.4f}')
    print('WS snapshots where pending at this price changed:')
    for e in ser[(mid, pt)]:
        if t - 2000 <= e[0] <= t + 2000:
            print(f'  {ts(e[0])}  bids {e[3]:8.4f} -> {e[1]:8.4f} ({e[1] - e[3]:+8.4f})   asks {e[4]:8.4f} -> {e[2]:8.4f} ({e[2] - e[4]:+8.4f})')
