import gzip, json, sys, collections
st = collections.Counter(); seen = set()
with gzip.open(sys.argv[1], 'rt') as f:
    for line in f:
        if '"/v1/orders/matches"' not in line[:600]:
            continue
        r = json.loads(line)
        if r.get('status') != 200:
            continue
        for x in json.loads(r['raw']).get('data') or []:
            key = x['transactionHash'] + (x['taker'].get('hash') or '')
            if key in seen:
                continue
            seen.add(key)
            ms = x['makers']; af = int(x['amountFilled']); sm = sum(int(m['amount']) for m in ms)
            st['matches'] += 1; st['multi_maker'] += len(ms) > 1
            st['sum_maker_eq_filled'] += sm == af
            st['sum_maker_gt_filled'] += sm > af
            st['sum_maker_lt_filled'] += sm < af
            tx_takers = None
        # nothing
print(dict(st))
