"""Placement-relative lock grid (round 4; exec-port request 2026-11-02: Predict's removalLockedUntil counts from placement).
Live-like C2-P, sequential; old order fillable until max(decision + 20 ms, activation + lambda), new order 20 ms later.
lambda = 0 (XP0, parity control vs round 2's XS0, must match to the cent) / 50 / 100 / 200 / 500 / 1000 ms x throttle
none / 1 place/s / 2 places/s. Cells: 6-day gross, days > 0, +1 s c/sh, shares, share of cancels held by the lock.
Break-even lambda per column by linear interpolation where the 6-day sum changes sign. Order age at cancel from the XP0
arm: age at the cancel decision = t_cancel - activation; held under lambda iff the cancel lands (decision + 20) before
activation + lambda. Usage: python3 -m predictfun.research.pf_live_report4
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
B = f'{io.OUT}/step4/live_tokyo20_custom_e2_preds_lodo6_'; D4 = B + 'livedev4'
PL = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1'; A1 = '/mid@A1:5:25:0'
LAM = [0, 50, 100, 200, 500, 1000]; TH = {'none': '', '1/s': '+XT1', '2/s': '+XT2'}
S4 = {d: json.load(open(f'{D4}/{d}/summary.json'))['arms'] for d in DAYS}
S2 = {d: json.load(open(f'{B}livedev2/{d}/summary.json'))['arms'] for d in DAYS}


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def cell(S, name):
    if not all(name in S[d] for d in DAYS):
        return None
    v = [S[d][name] for d in DAYS]; g = [x['gross_usd'] for x in v]; sh = sum(x['shares'] for x in v)
    c = {}
    for x in v:
        for k, n in x['live'].get('counters', {}).items():
            c[k] = c.get(k, 0) + n
    canc = sum(x['live']['requests'].get('cancel', 0) for x in v)
    return dict(sum=round(sum(g), 2), by_day=g, pos=sum(x > 0 for x in g), shares=round(sh),
                mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3), cancels=canc,
                held_share=round(c.get('plock_held', 0) / max(canc, 1), 3))


# parity control: XP0 must equal XS0 (round 2) exactly, per day
par = {}
for th in ('', '+XT1', '+XT2'):
    a = f'{PL}+XP0{th}{A1}'; b = f'{PL}+XS0{th}{A1}'
    if all(a in S4[d] for d in DAYS) and all(b in S2[d] for d in DAYS):
        par[th or 'none'] = all(S4[d][a]['gross_usd'] == S2[d][b]['gross_usd'] and S4[d][a]['fills'] == S2[d][b]['fills']
                                and S4[d][a]['live']['requests'] == S2[d][b]['live']['requests'] for d in DAYS)
grid = {}
for c, th in TH.items():
    col = {}
    for lam in LAM:
        r = cell(S4, f'{PL}+XP{lam}{th}{A1}')
        if r is None and lam == 0:
            r = cell(S2, f'{PL}+XS0{th}{A1}')                    # lambda 0 = sequential L 0 (identical by the parity control)
        col[lam] = r
    grid[c] = col
be = {}
for c, col in grid.items():
    pts = [(l, col[l]['sum']) for l in LAM if col[l] is not None]; be[c] = None
    for (l0, s0), (l1, s1) in zip(pts, pts[1:]):
        if s0 > 0 >= s1:
            be[c] = round(l0 + (l1 - l0) * s0 / (s0 - s1), 1); break
q = pd.concat([pd.read_parquet(f'{D4}/{d}/requests/{stem(PL + "+XP0" + A1)}.parquet') for d in DAYS]).sort_values(['mid', 'side', 't_ms'], kind='stable')
ages = []
for _, g in q.groupby(['mid', 'side'], sort=False):
    tp = None
    for t, k in zip(g.t_ms.values, g.kind.values):
        if k == 'place':
            tp = t
        elif k == 'cancel' and tp is not None:
            ages.append(t - (tp + 20.0)); tp = None             # age at the cancel decision (activation = place decision + 20)
a = np.array(ages)
age = dict(cancels=int(len(a)), pct_ms={p: float(np.percentile(a, p)) for p in (10, 25, 50, 75, 90)},
           held_share_by_lambda={l: round(float(np.mean(a + 20.0 < l)), 3) for l in (50, 100, 200, 500, 1000)})
out = dict(days=DAYS, design='sequential, placement-relative lock', parity_xp0_equals_xs0=par, grid={c: {str(l): x for l, x in col.items()} for c, col in grid.items()},
           break_even_lambda_ms=be, order_age_at_cancel_decision=age)
io.write_json(f'{D4}/report_plock_grid.json', out)
print('parity XP0 == XS0 (round 2):', par)
print('placement-relative lock grid, live-like C2-P, 6-day gross $ (days > 0) [+1 s c/sh] {share of cancels held}')
print(f"{'lam ms':>6s} " + ' '.join(f'{c:>34s}' for c in TH))
for l in LAM:
    print(f'{l:6d} ' + ' '.join((f"{x['sum']:10.2f} ({x['pos']}/6) [{x['mo1_c_sh']:5.2f}] {{{x['held_share']:.2f}}}" if x else 'n/a').rjust(34) for x in (grid[c][l] for c in TH)))
print('break-even lambda (ms):', be)
print('order age at cancel decision (ms):', age['pct_ms'], 'held share by lambda (no-lock schedule):', age['held_share_by_lambda'])
for c in TH:
    print(c, 'by day:', {l: grid[c][l]['by_day'] for l in LAM if grid[c][l]})
