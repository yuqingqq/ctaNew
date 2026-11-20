"""Why the venue-future path is None for original 98 (ask), retain, decision 1,600 ms (the census's 2 causality failures): which build_path
condition fails under the draw the census used (seed int(d) % 100019), and under later deterministic seeds."""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_r2_shadow as R2  # noqa: E402
X, QP, KP = R2.X, R2.QP, R2.KP
day = '20261023'; src = KP.Tape(); O = pd.read_parquet(KP.OUT / day / 'orders.parquet'); o = O[(O.oid == 98) & (O.side == 1)].iloc[0]
m = int(o.m); data = src.load(day, [m], src.offset(day)); t0 = float(o.t); g = 1600.; d = t0 + g
for W in src.pairs(data):
    if int(W[0]['m']) != m:
        continue
    print('original', int(o.oid), 'side', int(o.side), 'market', m, 'status', o.status, 'base path', QP.build_path(W, t0, t0 + R2.END)[1])
    for sd in [int(d) % 100019] + [int(d) % 100019 + i for i in range(1, 8)]:
        Vw = R2.venue_windows(W, d, np.random.default_rng(sd)); pth, st = QP.build_path(Vw, t0, t0 + R2.END)
        print('seed', sd, 'venue-future path status:', st)
    jv = int(np.searchsorted(np.asarray(W[0]['tf'], float), d, side='right') - 1); e = int(np.searchsorted(W[0]['tf'], t0 + R2.END, side='right'))
    print('frames after the venue frame by d up to END:', e - jv - 1, '; END - d =', R2.END - g, 'ms')
