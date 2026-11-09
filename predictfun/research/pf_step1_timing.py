"""Step 1c: time REST maker legs with WS settlementsPending increments.

Finding (10-23, step 1b + units probe): a pending increment's value is USD at the YES price
(shares x p_yes), booked on either YES side, within about +-0.5 s of the REST executedAt second.
Here every live maker leg (and, failing that, its whole (market, second, side, price) cell) is
matched to the nearest unused increment at the same price, either side, within [S-1.5 s, S+1.5 s],
whose value equals shares x p within max(1.1%, $0.01).

Usage: python3 -m predictfun.research.pf_step1_timing 20261023
Writes step1/<day>/legs_timed.parquet (t_ms = matched WS source time or NaN) + timing.json.
"""
import sys, bisect, collections
from .pf_trade_tape import refuse_legacy_clock
refuse_legacy_clock()  # Superseded: nearby anonymous increments cannot establish exact fill timing.
import numpy as np, pandas as pd
from . import pf_io as io

day = sys.argv[1]; S1 = f'{io.OUT}/step1/{day}'
L = pd.read_parquet(f'{S1}/legs.parquet')
L = L[L.phase == 'live'].reset_index(drop=True)
P = pd.read_parquet(f'{S1}/pend_inc.parquet', columns=['mid', 't_ms', 'side', 'ptick', 'inc'])
pidx = collections.defaultdict(list)
for i, (mid, t, pt, inc) in enumerate(zip(P.mid.values, P.t_ms.values, P.ptick.values, P.inc.values)):
    pidx[(mid, pt)].append((t, inc, i))
for v in pidx.values():
    v.sort()
ptimes = {k: [a for a, _, _ in v] for k, v in pidx.items()}
used = np.zeros(len(P), dtype=bool)


def match(mid, s, pt, value):
    k = (mid, pt)
    v = pidx.get(k)
    if not v:
        return None
    ts = ptimes[k]; base = s * 1000
    a = bisect.bisect_left(ts, base - 1500); b = bisect.bisect_left(ts, base + 1500)
    best = None
    tol = max(0.011 * value, 0.01)
    for x in range(a, b):
        t, inc, i = v[x]
        if used[i] or abs(inc - value) > tol:
            continue
        if best is None or abs(t - base) < abs(best[0] - base):
            best = (t, i)
    if best is None:
        return None
    used[best[1]] = True
    return best[0]


L = L.sort_values(['mid', 'exec_s']).reset_index(drop=True)
t_ms = np.full(len(L), np.nan)
how = np.zeros(len(L), dtype=np.int8)            # 0 untimed, 1 leg match, 2 cell match
for i, (mid, s, pt, sh) in enumerate(zip(L.mid.values, L.exec_s.values, L.ptick.values, L.shares.values)):
    t = match(mid, s, pt, sh * pt / 100.0)
    if t is not None:
        t_ms[i] = t; how[i] = 1
io.mem_check('legs')
# cell-level pass for legs still untimed
L['t_ms'] = t_ms; L['how'] = how
rest = L[L.how == 0]
for (mid, s, side, pt), g in rest.groupby(['mid', 'exec_s', 'side', 'ptick']):
    t = match(mid, s, pt, g.shares.sum() * pt / 100.0)
    if t is not None:
        L.loc[g.index, 't_ms'] = t; L.loc[g.index, 'how'] = 2
off = (L.t_ms - L.exec_s * 1000).dropna().values
vol = L.shares.sum()
out = dict(
    day=day, n_live_legs=int(len(L)), live_shares=round(float(vol), 1),
    timed_leg_share=round(float((L.how > 0).mean()), 4), timed_volume_share=round(float(L.shares[L.how > 0].sum() / vol), 4),
    by_method=dict(leg=int((L.how == 1).sum()), cell=int((L.how == 2).sum()), none=int((L.how == 0).sum())),
    offset_ms_pct=dict(zip(['p01', 'p05', 'p25', 'p50', 'p75', 'p95', 'p99'], np.percentile(off, [1, 5, 25, 50, 75, 95, 99]).round(0).tolist())),
    share_offset_in_m500_500=round(float(((off >= -500) & (off < 500)).mean()), 4),
    share_offset_in_0_1000=round(float(((off >= 0) & (off < 1000)).mean()), 4),
    timed_share_by_size={lab: round(float((L.how[m] > 0).mean()), 3) for lab, m in
                         (('<1sh', L.shares < 1), ('1-5', (L.shares >= 1) & (L.shares < 5)), ('5-50', (L.shares >= 5) & (L.shares < 50)), ('>=50', L.shares >= 50))},
    pend_increments=int(len(P)), pend_increments_used=int(used.sum()),
)
L.to_parquet(f'{S1}/legs_timed.parquet', index=False)
io.write_json(f'{S1}/timing.json', out)
print(out)
