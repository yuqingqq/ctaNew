import sys, glob, json, numpy as np, pandas as pd
S = '/home/yuqing/ctaNew/data/predict_fun/research/step1'
for d in sys.argv[1:]:
    ck = json.load(open(f'{S}/{d}/clock_rl_d2.json'))
    out = []; neg = 0; n = 0; dup = 0; nonmono = 0
    for fn in sorted(glob.glob(f'{S}/{d}/books/part_{d}_*.parquet')):
        b = pd.read_parquet(fn, columns=['mid', 't_ms', 'recv_ns'])
        lat = b.recv_ns.values / 1e6 - b.t_ms.values
        out.append((fn[-10:-8], np.median(lat), np.percentile(lat, 1)))
        neg += int((lat < 0).sum()); n += len(b)
        dup += int(b.duplicated(['mid', 't_ms']).sum())
        # receipt order vs venue order inside a market (file order = receipt order?)
        for m, g in b.groupby('mid', sort=False):
            nonmono += int((np.diff(g.t_ms.values) < 0).sum())
    med = np.array([o[1] for o in out])
    print(d, 'shift', round(ck['shift_ms'], 1), 'day-median implied', round(ck['shift_ms'] + 53.1, 1), '| hourly median recv-t (ms) min/max', round(med.min(), 1), round(med.max(), 1),
          'hours', len(out), '| hourly p01 min', round(min(o[2] for o in out), 1), '| frames recv<t', neg, 'of', n, '| dup (mid,t_ms) frames', dup, '| t_ms decreasing in file order', nonmono)
    print('   hourly medians', [round(x, 0) for x in med])
