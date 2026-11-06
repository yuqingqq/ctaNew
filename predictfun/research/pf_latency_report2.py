"""Latency test L2c report (STATUS.yml 2026-11-03; supersedes L2 / L2b; the L2b text below describes superseded inputs): live-like C2-P with split order / cancel latency.
Order live after sending: 22 ms central (Predict's 'Accepted' stamps lag acceptance; live between ~2 and ~44 ms), 40 ms
bracket. Cancel effective: 37 ms (Predict's 'Cancelled' stamps, p50), 25 ms lower bound (Predict's clock may lead ours
by ~12 ms). Decision 10 ms (now) or 2 ms (exec-port config reorder_ms 0, advance_ms 1). Model A (frozen spec, out of
day). L1 column = 22 / 22 ms, decision 10. Per cell: 6-day gross, days > 0, +1 s c/sh; plus max |net| in a window and
requests per quoting minute for the central path (22 / 37, decision 10).
Usage: python3 -m predictfun.research.pf_latency_report2
"""
import json, os
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']
S4 = f'{io.OUT}/step4'
COLS = {'L1 22/22 f30 d10': 'live_tokyoreal_custom_e2_preds_rl_a_l1', 'main 40/22 f100 d10': 'live_tokyoreal_custom_e2_preds_rl_a_l2c_40_22_f100_d10',
        'order 22': 'live_tokyoreal_custom_e2_preds_rl_a_l2c_22_22_f100_d10', 'fills 30': 'live_tokyoreal_custom_e2_preds_rl_a_l2c_40_22_f30_d10',
        'decision 2': 'live_tokyoreal_custom_e2_preds_rl_a_d2_l2c_40_22_f100_d2'}
PL = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1'; A1 = '/mid@A1:5:25:0'
ROWS = [('concurrent, none', f'{PL}{A1}'), ('concurrent, venue', f'{PL}+XV{A1}'), ('sequential, none', f'{PL}+XS0{A1}'),
        ('sequential, venue', f'{PL}+XS0+XV{A1}'), ('sequential, 1/s (v6)', f'{PL}+XS0+XT1{A1}')]
SUM = {c: {d: json.load(open(f'{S4}/{v}/{d}/summary.json'))['arms'] for d in DAYS} for c, v in COLS.items() if os.path.isdir(f'{S4}/{v}')}


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def cell(c, name, extra=False):
    if c not in SUM or not all(name in SUM[c][d] for d in DAYS):
        return None
    v = [SUM[c][d][name] for d in DAYS]; g = [x['gross_usd'] for x in v]; sh = sum(x['shares'] for x in v)
    out = dict(sum=round(sum(g), 2), by_day=g, pos=sum(x > 0 for x in g), shares=round(sh), mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3))
    if extra:
        base = f'{S4}/{COLS[c]}'; wm = []; q = []; W = []
        for d in DAYS:
            f = pd.read_parquet(f'{base}/{d}/fills/{stem(name)}.parquet', columns=['mid', 't', 'side', 'q']).sort_values(['mid', 't'], kind='stable')
            if len(f):
                wm.append(pd.Series(np.where(f.side.values == 0, 1.0, -1.0) * f.q.values).groupby(f.mid.values).cumsum().abs().groupby(f.mid.values).max().values)
            q.append(pd.read_parquet(f'{base}/{d}/requests/{stem(name)}.parquet', columns=['t_ms'])); W.append(pd.read_parquet(f'{base}/{d}/win/{stem(name)}.parquet', columns=['ws']))
        q = pd.concat(q); W = pd.concat(W); qm = np.concatenate([np.arange(int(w // 60000), int(w // 60000) + 4) for w in W.ws.values])
        cm = pd.Series((q.t_ms // 60000).astype(np.int64)).value_counts().reindex(qm, fill_value=0).values
        out.update(max_abs_net=round(float(np.concatenate(wm).max()), 1) if wm else 0.0, req_min_mean=round(float(cm.mean()), 1), min_gt500=round(float((cm > 500).mean()), 4))
    return out


tab = {lab: {c: cell(c, name, extra=(c == 'main 40/22 f100 d10')) for c in COLS} for lab, name in ROWS}
out = dict(days=DAYS, columns=COLS, table=tab)
io.write_json(f'{S4}/{COLS["main 40/22 f100 d10"]}/report_latency_l2c.json', out)
print(f"{'arm':22s} " + ' '.join(f'{c:>20s}' for c in COLS))
for lab, row in tab.items():
    print(f'{lab:22s} ' + ' '.join((f"{x['sum']:8.0f} ({x['pos']}/6) {x['mo1_c_sh']:5.2f}" if x else 'n/a').rjust(20) for x in row.values()))
for lab, row in tab.items():
    m = row['main 40/22 f100 d10']
    if m:
        print(f"  {lab}: main path by day {m['by_day']}; max |net| {m['max_abs_net']}, req/min {m['req_min_mean']}, minutes > 500 {m['min_gt500']}")
