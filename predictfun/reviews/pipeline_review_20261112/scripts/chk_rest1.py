import gzip, json, sys
p = sys.argv[1]
with gzip.open(p, 'rt') as f:
    for line in f:
        if '"/v1/orders/matches"' not in line[:600]:
            continue
        r = json.loads(line); pl = json.loads(r['raw'])
        x = pl['data'][0]
        def strip(o, depth=0):
            if isinstance(o, dict):
                return {k: (strip(v, depth + 1) if k not in ('description', 'imageUrl', 'question', 'title') else '...') for k, v in o.items()}
            if isinstance(o, list):
                return [strip(v, depth + 1) for v in o[:2]]
            return o
        print(json.dumps(strip(x), indent=1)[:3500]); print('params', r['params'], 'n', len(pl['data']), 'cursor' in pl)
        break
