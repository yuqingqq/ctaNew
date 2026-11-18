# Feasibility of the frame identity rule only: counts, no time differences printed.
import gzip, json, collections
MK = {4335911: ('v6', 1794124542214.35, '20261108_07'), 4337531: ('v6', 1794132021814.82, '20261108_10'), 4365071: ('basis', 1794163251268.75, '20261108_18')}
RUN = {'v6': '/home/yuqing/pmmm-data/tokyo-live-data-20261109/tokyo-live-20261109/v6/run-1794108465172/inputs.jsonl.gz',
       'basis': '/home/yuqing/pmmm-data/tokyo-live-data-20261109/tokyo-live-20261109/basis/run-1794147344737/inputs.jsonl.gz'}
W = 60_000.
def tick(p): return int(round(float(p) * 100))
def key_bot(fr):
    return (tuple((int(p), round(float(s), 4)) for p, s in fr['bids'][:5]), tuple((int(p), round(float(s), 4)) for p, s in fr['asks'][:5]),
            tuple(sorted((int(a), int(b), round(float(c), 4)) for a, b, c in fr['pending'])))
def key_res(d):
    b = sorted(d.get('bids') or [], key=lambda z: -z[0])[:5]; a = sorted(d.get('asks') or [], key=lambda z: z[0])[:5]
    sp = d.get('settlementsPending') or {}; cur = collections.defaultdict(float)
    for si, k in ((0, 'bids'), (1, 'asks')):
        for p, v in sp.get(k) or []: cur[(si, tick(p))] += float(v)
    return (tuple((tick(p), round(float(s), 4)) for p, s in b), tuple((tick(p), round(float(s), 4)) for p, s in a),
            tuple(sorted((si, t, round(v, 4)) for (si, t), v in cur.items())))
for mid, (run, tc, hour) in MK.items():
    bot = []
    def lines(p):
        with gzip.open(p, 'rt') as f:
            try:
                for line in f:
                    if line.endswith('\n'): yield line
            except EOFError:
                return
    for line in lines(RUN[run]):
            if not line.startswith('{"Frame":{"market":%d,' % mid): continue
            fr = json.loads(line)['Frame']
            if abs(fr['known_ms'] - tc) <= W: bot.append((fr['known_ms'], key_bot(fr)))
    res = []
    for h in sorted({hour, hour[:-2] + '%02d' % (int(hour[-2:]) - 1), hour[:-2] + '%02d' % (int(hour[-2:]) + 1)}):
        try:
            with gzip.open(f'data/predict_fun/runs/20261021T162823Z_0dd45f2d5a/predict_ws/{h}.jsonl.gz', 'rt') as f:
                for line in f:
                    if f'predictOrderbook/{mid}"' not in line and f'predictOrderbook/{mid}\\"' not in line: continue
                    r = json.loads(line); m = json.loads(r['raw'])
                    if m.get('type') != 'M': continue
                    d = m.get('data')
                    if isinstance(d, dict) and 'updateTimestampMs' in d and abs(d['updateTimestampMs'] - tc) <= W + 5000: res.append((d['updateTimestampMs'], key_res(d)))
        except FileNotFoundError: pass
    byk = collections.defaultdict(list)
    for u, k in res: byk[k].append(u)
    st = collections.Counter()
    for t, k in bot:
        c = [u for u in byk.get(k, []) if abs(u - t) <= 5000]
        st['none' if not c else 'one' if len(c) == 1 else 'several'] += 1
    print(mid, run, 'bot frames', len(bot), 'research frames', len(res), 'distinct research keys', len(byk), dict(st))
