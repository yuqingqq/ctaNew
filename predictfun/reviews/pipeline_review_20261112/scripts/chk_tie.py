import gzip, json, sys
mid = int(sys.argv[2]); n = 0
with gzip.open(sys.argv[1], 'rt') as f:
    for line in f:
        if '"/v1/categories/' not in line[:700] or str(mid) not in line:
            continue
        r = json.loads(line)
        if r.get('status') != 200:
            continue
        d = json.loads(r['raw']).get('data') or {}
        for m in d.get('markets') or []:
            if m['id'] == mid and m.get('resolution'):
                print(r['recv_ns'], 'resolution', json.dumps(m['resolution'])[:400], '| outcomes', [(o['name'], o.get('status')) for o in m['outcomes']], '| variant', {k: (m.get('variantData') or {}).get(k) for k in ('startPrice', 'endPrice')}, 'status', m.get('status'))
                n += 1
        if n >= 2:
            break
