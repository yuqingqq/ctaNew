"""Population counts for the untimed-leg rule (no label values read): for each built day, the probes refused as missing_trade_timing
under the current contract (any untimed leg in the window), and how many of them an untimed leg could actually reach under two bounded
rules - (a) the leg's exec-second band overlaps [ws, end] (accounting 3abb190f's rule), (b) it overlaps the action's own window
[t, end] at or beyond the order's price on the order's side."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/yuqing/ctaNew/predictfun/research')
import pf_stop_ev as E   # noqa: E402

K = E.K
out = []
for day in E.DAYS9:
    jf = f'{E.OUTD}/day_{day}.json'
    if not os.path.exists(jf):
        continue
    A = pd.read_parquet(f'{E.OUTD}/actions_{day}_all_anchors_sensitivity.parquet', columns=['m', 'side', 'rung', 'decision_ms', 'status', 'x_price'])
    D = dict(LT=None, W0=None)
    reg = pd.read_parquet(f'{K.IO}/step1/{day}/registry.parquet', columns=['mid', 'ws'])
    W0 = {int(m): float(w) * 1000.0 for m, w in zip(reg.mid, reg.ws)}
    T = pd.read_parquet(f'{K.IO}/step1/{day}/legs_timed2.parquet', columns=['mid', 'exec_s', 'side', 'ptick', 'how', 't_ms', 't_e2'])
    tk = pd.read_parquet(f'{K.IO}/step1/{day}/legs_tk.parquet', columns=['tk']).tk.values.astype('int64')
    rid = np.arange(len(T)); grp = np.where(tk >= 0, tk, -1 - rid)
    wsT = np.where(T.how.values > 0, T.t_ms.values.astype(float), np.nan)
    e2T = np.where((T.how.values == 0) & np.isfinite(T.t_e2.values.astype(float)), T.t_e2.values.astype(float), np.nan)
    a = pd.Series(wsT).groupby(grp).transform('min').values; b = pd.Series(e2T).groupby(grp).transform('min').values
    anc = np.where(np.isfinite(a), a, b)
    band = json.load(open(jf))['untimed']['band_ms']
    ws = T.mid.map(W0).values.astype(float); ex = T.exec_s.values.astype(float) * 1000.0
    u = ~np.isfinite(anc) & np.isfinite(ws) & (ex >= ws - 2000.0) & (ex < ws + 61000.0)
    U = pd.DataFrame(dict(m=T.mid.values[u], side=T.side.values[u], p=T.ptick.values[u], lo=ex[u] + band[0] - E.SHF, hi=ex[u] + band[1] - E.SHF))
    byw = {int(m): g for m, g in U.groupby('m')}
    mt = A[A.status.values == 'missing_trade_timing']
    na = nb = 0
    for m, sd, t, pr in zip(mt.m.values, mt.side.values, mt.decision_ms.values, mt.x_price.values):
        g = byw.get(int(m)); end = min(t + 1000.0, W0[int(m)] + 60200.0)
        if g is None:
            continue
        na += int(((g.lo.values <= end) & (g.hi.values >= W0[int(m)])).any())
        reach = (g.p.values <= pr) if sd == 0 else (g.p.values >= pr)
        nb += int(((g.side.values == sd) & reach & (g.lo.values <= end) & (g.hi.values >= t)).any())
    nwin = len(set(A.m.values.tolist())); wu = len(byw)
    out.append(dict(day=day, probes=int(len(A)), missing_trade_timing=int(len(mt)), windows=nwin, windows_with_untimed=wu,
                    refused_bounded_prefix=na, refused_own_window_reach=nb))
    print(out[-1], flush=True)
json.dump(out, open('/tmp/claude-1001/-home-yuqing-ctaNew/aea5bfb9-4ea4-41d0-bf73-bb98d818e3d4/scratchpad/untimed_counts.json', 'w'), indent=1)
