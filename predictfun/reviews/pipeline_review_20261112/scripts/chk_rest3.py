import gzip, json, sys, collections, datetime as dt
iso = lambda s: dt.datetime.fromisoformat(s.replace('Z', '+00:00')).timestamp()
polls = collections.defaultdict(list)
for p in sys.argv[1:]:
    with gzip.open(p, 'rt') as f:
        for line in f:
            if '"/v1/orders/matches"' not in line[:600]:
                continue
            r = json.loads(line)
            if r.get('status') != 200:
                continue
            pr = r.get('params') or {}
            D = json.loads(r['raw']).get('data') or []
            ts = [iso(x['executedAt']) for x in D]
            if ts:
                polls[pr.get('marketId')].append((r['recv_ns'], len(D), min(ts), max(ts), 'after' in pr))
st = collections.Counter()
for mk, v in polls.items():
    v.sort(); mx = None
    for i, (rv, n, lo, hi, aft) in enumerate(v):
        if not aft and n >= 100 and mx is not None and lo > mx:
            st['first_page_full_and_disjoint'] += 1
            later = [w for w in v[i + 1:] if w[4] and w[0] - rv < 120e9]
            reach = min((w[2] for w in later), default=None)
            if reach is not None and reach <= mx:
                st['closed_by_cursor_within_120s'] += 1
            else:
                st['NOT_closed'] += 1
        mx = hi if mx is None else max(mx, hi)
    st['polls'] += len(v)
print(dict(st))
