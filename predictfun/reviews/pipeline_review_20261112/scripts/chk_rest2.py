import gzip, json, sys, collections, datetime as dt
iso = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
pk = collections.Counter(); polls = collections.defaultdict(list)
keys = {}; coll = collections.Counter()
for p in sys.argv[1:]:
    with gzip.open(p, 'rt') as f:
        for line in f:
            if '"/v1/orders/matches"' not in line[:600]:
                continue
            r = json.loads(line)
            if r.get('status') != 200:
                pk['non200'] += 1; continue
            pr = r.get('params') or {}
            pk[tuple(sorted(pr.keys()))] += 1
            pl = json.loads(r['raw']); D = pl.get('data') or []
            ts = [iso(x['executedAt']) for x in D]
            frac = sum(1 for x in D if '.' in x['executedAt'] and not x['executedAt'].endswith('.000Z'))
            pk['records_with_nonzero_fraction'] += frac; pk['records'] += len(D)
            polls[pr.get('marketId')].append((r['recv_ns'], len(D), min(ts) if ts else None, max(ts) if ts else None, 'after' in pr or 'cursor' in pr))
            for x in D:
                tx = x['transactionHash']; th = x['taker'].get('hash')
                for m in x['makers']:
                    k = tx[2:18] + m['hash'][2:18]
                    v = (tx, th, m['amount'], x['executedAt'])
                    o = keys.get(k)
                    if o is None:
                        keys[k] = v
                    elif o != v:
                        coll['diff_' + ('tx' if o[0] != v[0] else 'taker' if o[1] != v[1] else 'amount' if o[2] != v[2] else 'time')] += 1
print('param keys', dict(pk))
print('leg keys', len(keys), 'collisions with different content', dict(coll))
gaps = 0; full = 0; npoll = 0; gap_markets = set()
for mk, v in polls.items():
    v.sort()
    for a, b in zip(v, v[1:]):
        npoll += 1
        if b[1] >= 100:
            full += 1
            if a[3] is not None and b[2] is not None and b[2] > a[3]:
                gaps += 1; gap_markets.add(mk)
print('consecutive poll pairs', npoll, 'full pages', full, 'full page whose oldest record is newer than previous poll newest (possible missed matches)', gaps, 'markets', len(gap_markets))
