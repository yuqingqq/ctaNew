import sys, glob, json, numpy as np, pandas as pd
S = '/home/yuqing/ctaNew/data/predict_fun/research/step1'
for d in sys.argv[1:]:
    reg = pd.read_parquet(f'{S}/{d}/registry.parquet', columns=['mid', 'ws']); W0 = dict(zip(reg.mid, reg.ws * 1000.0))
    B = pd.concat([pd.read_parquet(f, columns=['mid', 't_ms']) for f in sorted(glob.glob(f'{S}/{d}/books/part_{d}_*.parquet'))], ignore_index=True)
    B['rel'] = B.t_ms - B.mid.map(W0); B = B[(B.rel >= -5000) & (B.rel <= 61000)].sort_values(['mid', 't_ms'], kind='stable')
    LT = pd.read_parquet(f'{S}/{d}/legs_timed2.parquet', columns=['mid', 'exec_s', 'shares', 't_ms', 'how', 't_rel'])
    LT = LT[(LT.how > 0) & (LT.t_rel >= 0) & (LT.t_rel < 60)]
    gaps = []; tr_in = 0; sh_in = 0.0
    for m, g in B.groupby('mid', sort=False):
        t = g.t_ms.values; dt = np.diff(t); k = np.where(dt > 2000)[0]
        if not len(k):
            continue
        lt = LT[LT.mid == m].t_ms.values; ls = LT[LT.mid == m].shares.values
        for i in k:
            if t[i + 1] - W0[m] < 0:
                continue
            inside = (lt > t[i] + 100) & (lt < t[i + 1] - 100)
            gaps.append((m, (t[i] - W0[m]) / 1000, dt[i] / 1000, int(inside.sum())))
            tr_in += int(inside.sum()); sh_in += float(ls[inside].sum())
    s = json.load(open(f'{S}/{d}/summary.json'))['collector_events']
    print(d, 'frame gaps >2 s overlapping [ws, ws+61s]', len(gaps), 'with WS-timed legs inside', sum(1 for g in gaps if g[3] > 0), 'legs', tr_in, 'shares', round(sh_in, 1), '| events', {k: v for k, v in s.items() if 'conn' in str(k).lower() or 'gap' in str(k).lower() or 'error' in str(k).lower() or 'stale' in str(k).lower()})
    print('   largest', sorted(gaps, key=lambda g: -g[2])[:4])
