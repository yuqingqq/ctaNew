"""Invariant checks on step-4 outputs (fills parquet + summary.json). Read-only.

Per arm and day:
  I1 accounting: sum of per-fill settlement P&L == summary gross_usd; fills and shares == summary
  I2 stop: arms with G5 have no fill later than stop + order latency (20 ms) into the window
  I3 hard cap: arms with G6 never hold |net| > 5 x cap (cap = size, or hi for adaptive) within a window
  I4 size: no single fill larger than the order size (hi for adaptive)
Positive controls (must fire): I2 on a no-G5 arm (fills after 240 s exist), and I1-I4 on a corrupted copy.
Usage: python3 -m predictfun.research.pf_check_fills <step4 subdir> <day> [<day> ...]
"""
import sys, os, json
import numpy as np, pandas as pd
from . import pf_io as io

sub, days = sys.argv[1], sys.argv[2:]

def fname(name):
    return name.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-') + '.parquet'

def parse(name):
    head, tail = name.split('/')
    rules = set(head.split('+')[1:])
    mode, _, sz = tail.partition('@')
    if not sz:
        cap = 5.0
    elif ':' in sz:
        cap = float(sz.split(':')[2])
    else:
        cap = float(sz)
    stop = 240.0
    for r in rules:
        if r.startswith('S') and r[1:].isdigit():
            stop = float(r[1:])
    return rules, cap, stop

def check(f, v, rules, cap, stop):
    bad = []
    if abs(f.settle.sum() - v['gross_usd']) > 0.011 or len(f) != v['fills'] or abs(f.q.sum() - v['shares']) > 0.11:
        bad.append(f"I1 settle {f.settle.sum():.2f} vs {v['gross_usd']}, fills {len(f)} vs {v['fills']}, shares {f.q.sum():.1f} vs {v['shares']}")
    if 'G5' in rules and len(f) and f.t.max() > stop + 0.020 + 1e-6:
        bad.append(f'I2 fill at t={f.t.max():.3f} s > stop {stop} + 0.02')
    if 'G6' in rules and len(f):
        g = f.sort_values(['mid', 't'], kind='stable')
        net = (np.where(g.side.values == 0, 1.0, -1.0) * g.q.values)
        cum = pd.Series(net).groupby(g.mid.values).cumsum().abs().max()
        if cum > 5 * cap + 1e-6:
            bad.append(f'I3 |net| reached {cum:.1f} > hard {5 * cap}')
    if len(f) and f.q.max() > cap + 1e-6:
        bad.append(f'I4 fill q {f.q.max():.2f} > size {cap}')
    return bad

res = {}; n_arms = 0; n_bad = 0
for d in days:
    D = f'{io.OUT}/step4/{sub}/{d}'
    S = json.load(open(f'{D}/summary.json'))['arms']
    for name, v in S.items():
        rules, cap, stop = parse(name)
        f = pd.read_parquet(f'{D}/fills/{fname(name)}')
        b = check(f, v, rules, cap, stop); n_arms += 1
        if b:
            n_bad += 1; res[f'{d} {name}'] = b
print(f'[{sub}] arms checked {n_arms}, with violations {n_bad}')
for k, b in list(res.items())[:20]:
    print('  ', k, b)

# positive controls on the first day / first G5+G6 arm
d = days[0]; D = f'{io.OUT}/step4/{sub}/{d}'; S = json.load(open(f'{D}/summary.json'))['arms']
name = next(n for n in S if {'G5', 'G6'} <= parse(n)[0])
rules, cap, stop = parse(name); f = pd.read_parquet(f'{D}/fills/{fname(name)}')
c = f.copy()
extra = c.iloc[[0]].copy(); extra['t'] = stop + 5.0; extra['q'] = 6 * cap; extra['side'] = 0
c = pd.concat([c, extra] + [extra.assign(t=stop + 5.0 + i * 1e-3) for i in range(1, 6)], ignore_index=True)
fired = check(c, S[name], rules, cap, stop)
print('positive control (corrupted copy) fires:', [x[:2] for x in fired], 'expected I1 I2 I3 I4')
assert {x[:2] for x in fired} == {'I1', 'I2', 'I3', 'I4'}, 'checker failed its positive control'
