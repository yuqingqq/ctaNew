"""Step 1 probe (engineering, one hour): what do WS book / settlementsPending
changes look like around a REST match? Prints a few single-maker examples and
summary counts. Usage: python3 -m predictfun.research.pf_explore_semantics 20261023_14
"""
import sys, collections, bisect
from . import pf_io as io

hour = sys.argv[1]
nshow = int(sys.argv[2]) if len(sys.argv) > 2 else 6

# market registry from category payloads
mk = {}
for recv, params, path, pl in io.rest_rows(hour, '/v1/categories/'):
    d = pl.get('data') or {}
    for m in d.get('markets', []) or []:
        outs = {o['indexSet']: o['name'] for o in m.get('outcomes', [])}
        mk[m['id']] = dict(ws=io.iso_s(d.get('startsAt') or m.get('startsAt')) if (d.get('startsAt') or m.get('startsAt')) else None,
                           outs=outs, slug=d.get('slug'))
print('markets in categories:', len(mk), 'outcome maps:', collections.Counter(tuple(sorted(v['outs'].items())) for v in mk.values()))

# matches (compact): keep only what the probe prints
legs = {}
h0 = io.iso_s(hour[:4] + '-' + hour[4:6] + '-' + hour[6:8] + 'T' + hour[9:11] + ':00:00Z')
for recv, params, path, pl in io.rest_rows(hour, '/v1/orders/matches'):
    for x in pl.get('data') or []:
        k = x['transactionHash']
        if k in legs:
            continue
        mids = x['market']['id']; ts = io.iso_s(x['executedAt'])
        legs[k] = dict(recv=recv, mid=mids, ts=ts, at=x['executedAt'], amountFilled=int(x['amountFilled']),
                       makers=[dict(amount=m['amount'], price=m['price'], quoteType=m['quoteType'],
                                    outcome=dict(indexSet=m['outcome']['indexSet'], name=m['outcome']['name'])) for m in x['makers']],
                       taker=dict(name=x['taker']['outcome']['name'], qt=x['taker']['quoteType']))
def live(L):
    w = mk.get(L['mid'], {}).get('ws')
    return w is not None and w <= L['ts'] < w + 300 and h0 + 2 <= L['ts'] < h0 + 3598
single = [L for L in legs.values() if len(L['makers']) == 1 and live(L)]
print('distinct tx', len(legs), 'live in-hour', sum(1 for L in legs.values() if live(L)), 'single-maker live', len(single))
need = {L['mid'] for L in single}
frames = collections.defaultdict(list)
for recv, d in io.ws_books(hour):
    if d['marketId'] in need:
        frames[d['marketId']].append((d['updateTimestampMs'], recv, d))
for v in frames.values():
    v.sort(key=lambda x: (x[0], x[1]))
print('ws markets kept', len(frames), 'frames', sum(len(v) for v in frames.values()), '[mem]', io.rss_mib())

def lvl(d, side, p):
    for q, s in d.get(side + 's') or []:
        if abs(q - p) < 1e-9:
            return s
    return 0.0

def pend(d, side, p):
    for q, s in (d.get('settlementsPending') or {}).get(side + 's') or []:
        if abs(q - p) < 1e-9:
            return s
    return 0.0

# sum(maker amounts) vs amountFilled
eq = collections.Counter()
for L in legs.values():
    tot = sum(int(m['amount']) for m in L['makers'])
    eq['equal' if tot == L['amountFilled'] else 'differ'] += 1
print('sum(maker.amount) == amountFilled:', eq)

shown = 0
import random
random.seed(7)
for L in random.sample(single, min(len(single), 400)):
    m = L['makers'][0]; mid = L['mid']
    if mid not in frames:
        continue
    side, py = io.yes_side(m); ts = L['ts']
    a = int(m['amount']) / io.E18
    if a < 5:
        continue
    fr = frames[mid]; keys = [f[0] for f in fr]
    i0 = bisect.bisect_left(keys, ts * 1000 - 1500); i1 = bisect.bisect_left(keys, ts * 1000 + 2500)
    print(f"\n== mkt {mid} t={ts-mk[mid]['ws']:.0f}s executedAt {L['at']} maker {m['outcome']['name']} {m['quoteType']} "
          f"@{int(m['price'])/io.E18:.2f} -> YES {side} {py:.2f} amt {a:.3f}; taker {L['taker']['name']} {L['taker']['qt']}")
    for t, rv, d in fr[max(0, i0 - 1):i1]:
        los = d.get('lastOrderSettled') or {}
        print(f"  {t - ts*1000:+7.0f}ms  book {side}@{py:.2f}={lvl(d, side, py):9.3f} other={lvl(d, 'ask' if side=='bid' else 'bid', py):8.3f}  "
              f"pend bid={pend(d,'bid',py):8.3f} ask={pend(d,'ask',py):8.3f}  bb={d['bids'][0][0] if d.get('bids') else None} "
              f"ba={d['asks'][0][0] if d.get('asks') else None} v={d.get('version')} los={los.get('id')}/{los.get('price')}/{los.get('side')}/{los.get('outcome')}")
    shown += 1
    if shown >= nshow:
        break
print('[mem] VmRSS MiB', io.rss_mib())
