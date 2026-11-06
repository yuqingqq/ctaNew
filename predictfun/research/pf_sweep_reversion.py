"""Are the bad sweeps informed, or big orders whose impact fades? (2026-11-03; part of the cross-venue check declared 23:49Z.)
For market sweeps over 2+ prices (step4/sweep_precursor/sweeps.parquet) and our replay swept fills: the venue mid move after
the trade, signed for the makers (negative = against them), at 1 s, 10 s, 60 s, and the share of the 1 s move still there at
60 s. Classes as pf_sweep_precursor (informed at 1 s). Statuses: no mid at a horizon (window over or book empty).
Usage: python3 -m predictfun.research.pf_sweep_reversion
"""
import os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
S4 = f'{io.OUT}/step4'; SW = pd.read_parquet(f'{S4}/sweep_precursor/sweeps.parquet'); OU = pd.read_parquet(f'{S4}/sweep_precursor/ours.parquet')
H = (1000.0, 10000.0, 60000.0); acc = []; st = dict(no_mid=0)
for d in DAYS:
    S1 = f'{io.OUT}/step1/{d}'
    bk = pd.concat([pd.read_parquet(f'{S1}/books/{f}', columns=['mid', 't_ms', 'b1_p', 'a1_p']) for f in sorted(os.listdir(f'{S1}/books'))])
    bk = bk[(bk.b1_p >= 0) & (bk.a1_p > bk.b1_p)].sort_values(['mid', 't_ms'], kind='stable')
    reg = pd.read_parquet(f'{S1}/registry.parquet', columns=['mid', 'ws', 'we'])
    # the event files carry no market id: find it from the time (the live window containing it)
    ws = reg.ws.values * 1000.0; mids = reg.mid.values; o = np.argsort(ws); ws, mids = ws[o], mids[o]
    BY = {m: (g.t_ms.values.astype(float), (g.b1_p.values + g.a1_p.values) / 2.0) for m, g in bk.groupby('mid', sort=False)}; del bk
    for kind, X in (('sweep', SW[SW.day == d]), ('ours', OU[OU.day == d])):
        t = X.ta.values.astype(float); i = np.searchsorted(ws, t, side='right') - 1
        sg = np.where(X.dd.values < 0, 1.0, -1.0)                                  # dd -1 = bid makers swept
        res = {h: np.full(len(X), np.nan) for h in H}; m0 = np.full(len(X), np.nan)
        for n in range(len(X)):
            if i[n] < 0 or t[n] >= ws[i[n]] + 300000.0:
                continue
            v = BY.get(mids[i[n]])
            if v is None:
                continue
            tv, mv = v; j0 = np.searchsorted(tv, t[n], side='left') - 1
            if j0 < 0:
                continue
            m0[n] = mv[j0]
            for h in H:
                if t[n] + h < ws[i[n]] + 300000.0:
                    res[h][n] = sg[n] * (mv[np.searchsorted(tv, t[n] + h, side='right') - 1] - m0[n])
        R = pd.DataFrame(dict(kind=kind if kind == 'sweep' else X.grp.values, q=X.q.values, inf=X.informed.values, net=X.net.values if 'net' in X else np.nan,
                              **{f'm{h / 1000:g}': res[h] for h in H}))
        st['no_mid'] += int(R.m60.isna().sum()); acc.append(R)
    del BY
    io.mem_check(f'day {d}')
A = pd.concat(acc, ignore_index=True)
A['cls'] = np.where(A.kind == 'sweep', np.where(A.inf <= -2, 'bad', np.where(A.inf >= 0, 'good', 'mid')), np.where(A.net <= -2, 'bad', 'rest'))
print(f"statuses {st} (60 s past the window's end or no book; those rows drop from the 10 s / 60 s columns)")
print('mid move after the trade, signed for the makers (c), share-weighted; kept = 60 s move / 1 s move')
for (k, c), g in A.groupby(['kind', 'cls']):
    ok = g.m60.notna() & g.m1.notna(); w = g.q[ok]
    m1, m10, m60 = (np.average(g[x][ok], weights=w) for x in ('m1', 'm10', 'm60'))
    print(f"  {k:20s} {c:5s} n {ok.sum():6d}  1 s {m1:+6.2f}  10 s {m10:+6.2f}  60 s {m60:+6.2f}  kept {m60 / m1 if abs(m1) > 1e-9 else np.nan:5.0%}")
io.write_json(f'{S4}/report_sweep_reversion.json', {f'{k}|{c}': dict(n=int(len(g)), m1=float(np.nanmean(g.m1)), m60=float(np.nanmean(g.m60))) for (k, c), g in A.groupby(['kind', 'cls'])})
