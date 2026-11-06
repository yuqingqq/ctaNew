"""Concurrent vs sequential under the placement-relative lock (rounds 4 + 5; exec-port request 2026-11-02).
Concurrent (XQ, the frozen Tokyo version's replace): old order fillable beside its replacement until max(decision + 20,
activation + lambda). Sequential (XP, round 4): the side waits for the old order's removal, the new order 20 ms later.
Live-like C2-P, six read days. Per cell: 6-day gross, days > 0, +1 s c/sh, shares, per-window max |net| (max, p99, share of
windows > 75 = G6 soft cap + one 25-share order, share > 125 = G6 hard cap). Lookup columns per (lambda, throttle): the
design with the larger 6-day sum and a descriptive predicate positive_6d = (sum > 0 and >= 4 of 6 days > 0); the go /
no-go decision is the user's, not this script's. Parity: XQ0 unthrottled equals the round-2 live-like arm to the cent.
Usage: python3 -m predictfun.research.pf_live_report5
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
B = f'{io.OUT}/step4/live_tokyo20_custom_e2_preds_lodo6_'
PL = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1'; A1 = '/mid@A1:5:25:0'
LAM = [0, 50, 100, 200, 500]; TH = {'none': '', '1/s': '+XT1', '2/s': '+XT2'}
SRC = {r: {d: json.load(open(f'{B}{r}/{d}/summary.json'))['arms'] for d in DAYS} for r in ('livedev2', 'livedev4', 'livedev5a', 'livedev5b')}


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def find(name):
    for r, S in SRC.items():
        if all(name in S[d] for d in DAYS):
            return r
    return None


def cell(name, netstats=False):
    r = find(name)
    if r is None:
        return None
    v = [SRC[r][d][name] for d in DAYS]; g = [x['gross_usd'] for x in v]; sh = sum(x['shares'] for x in v)
    out = dict(src=r, sum=round(sum(g), 2), by_day=g, pos=sum(x > 0 for x in g), shares=round(sh),
               mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3))
    if netstats:
        wm = []
        for d in DAYS:
            f = pd.read_parquet(f'{B}{r}/{d}/fills/{stem(name)}.parquet', columns=['mid', 't', 'side', 'q']).sort_values(['mid', 't'], kind='stable')
            if len(f):
                c = pd.Series(np.where(f.side.values == 0, 1.0, -1.0) * f.q.values).groupby(f.mid.values).cumsum().abs()
                wm.append(c.groupby(f.mid.values).max().values)
        wm = np.concatenate(wm) if wm else np.zeros(1)
        out['window_max_abs_net'] = dict(max=round(float(wm.max()), 1), p99=round(float(np.percentile(wm, 99)), 1),
                                         share_gt75=round(float((wm > 75 + 1e-6).mean()), 4), share_gt125=round(float((wm > 125 + 1e-6).mean()), 4))
    return out


conc = {c: {l: cell(f'{PL}+XQ{l}{th}{A1}', True) for l in LAM} for c, th in TH.items()}
seqg = {c: {l: (cell(f'{PL}+XP{l}{th}{A1}', True) if l else (cell(f'{PL}+XS0{th}{A1}', True))) for l in LAM} for c, th in TH.items()}
base = cell(f'{PL}{A1}')
parity = conc['none'][0] is not None and conc['none'][0]['by_day'] == base['by_day']
look = {}
for c in TH:
    for l in LAM:
        a, b = conc[c][l], seqg[c][l]
        if a is None or b is None:
            continue
        best = 'concurrent' if a['sum'] >= b['sum'] else 'sequential'; x = a if best == 'concurrent' else b
        look[f'{l}|{c}'] = dict(concurrent=a['sum'], sequential=b['sum'], better=best, better_sum=x['sum'], better_pos=x['pos'],
                                positive_6d=bool(x['sum'] > 0 and x['pos'] >= 4))
be = {}
for nm, G in (('concurrent', conc), ('sequential', seqg)):
    for c in TH:
        pts = [(l, G[c][l]['sum']) for l in LAM if G[c][l] is not None]; k = f'{nm}|{c}'; be[k] = None
        for (l0, s0), (l1, s1) in zip(pts, pts[1:]):
            if s0 > 0 >= s1:
                be[k] = round(l0 + (l1 - l0) * s0 / (s0 - s1), 1); break
out = dict(days=DAYS, parity_xq0_equals_live_like=parity, concurrent=conc, sequential=seqg, lookup=look, break_even_lambda_ms=be)
io.write_json(f'{B}livedev5b/report_conc_vs_seq.json', out)
print('parity XQ0 (unthrottled) == live-like round 2:', parity)
print(f"{'lam':>4s} {'thr':>4s} | {'concurrent $':>12s} {'d+':>3s} {'mo1':>6s} {'max|net|':>8s} {'p99':>6s} {'>125':>6s} | {'sequential $':>12s} {'d+':>3s} {'max|net|':>8s} | better  pos6d")
for c in TH:
    for l in LAM:
        a, b = conc[c][l], seqg[c][l]
        if a is None or b is None:
            continue
        lk = look[f'{l}|{c}']
        print(f"{l:4d} {c:>4s} | {a['sum']:12.2f} {a['pos']:3d} {a['mo1_c_sh']:6.2f} {a['window_max_abs_net']['max']:8.1f} {a['window_max_abs_net']['p99']:6.1f} "
              f"{a['window_max_abs_net']['share_gt125']:6.4f} | {b['sum']:12.2f} {b['pos']:3d} {b['window_max_abs_net']['max']:8.1f} | {lk['better']:10s} {lk['positive_6d']}")
print('break-even lambda (ms):', be)
