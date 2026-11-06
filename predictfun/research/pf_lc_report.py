"""Round LC report (declared in STATUS.yml 2026-11-03 13:10Z): leftover control on base + S60, all 8 days.

Decision metric = mtm_stop: P&L marked to the venue mid at stop + 100 ms (spread captured minus adverse selection; the
outcome excluded). Risk = carry_stop (net x (outcome - mid)): expected ~0, reported as its per-window sd. Rule (computed):
the best 8-day mtm_stop variant is adopted only if its mtm_stop sum > base's AND it beats base on >= 6/8 days.
Usage: python3 -m predictfun.research.pf_lc_report
"""
import json
import numpy as np, pandas as pd
from . import pf_io as io

DAYS = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']
S4 = f'{io.OUT}/step4'
def sub(d):
    return f"{S4}/live_tokyoreal_custom_e2_{'preds_c3' if d >= '20261101' else 'preds_o3'}_lc"
SUM = {d: json.load(open(f'{sub(d)}/{d}/summary.json'))['arms'] for d in DAYS}
B = 'MICRO+G4+G2+G6+G5+G3+G1+M025+XNA+XMIN1+XS0+XV+S60'; A1 = '/mid@A1:5:25:0'
ARMS = [('base + S60', B + A1)] + [(v, f'{B}+{v}{A1}') for v in ('CAP10', 'CAP25', 'RO45', 'RO50', 'RO45+CAP25')] + \
       [('size 5..10', f'{B}/mid@A1:5:10:0'), ('size fixed 5', f'{B}/mid')]


def stem(n):
    return n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def row(name):
    v = [SUM[d][name] for d in DAYS]; sh = sum(x['shares'] for x in v)
    W = pd.concat([pd.read_parquet(f'{sub(d)}/{d}/win/{stem(name)}.parquet', columns=['mtm_stop', 'carry_stop', 'net_end']) for d in DAYS])
    W = W[W.mtm_stop.notna()]
    m = [x['mtm_stop_usd'] for x in v]
    return dict(mtm_by_day=m, mtm=round(sum(m), 2), mtm_pos=sum(x > 0 for x in m), carry=round(sum(x['carry_stop_usd'] for x in v), 2),
                gross=round(sum(x['gross_usd'] for x in v), 2), mtm_t=round(float(W.mtm_stop.mean() / (W.mtm_stop.std() / np.sqrt(len(W)))), 2),
                carry_sd_window=round(float(W.carry_stop.std()), 2), abs_net_mean=round(float(W.net_end.abs().mean()), 1), shares=round(sh),
                mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3), mtm_c_sh=round(100 * sum(m) / max(sh, 1e-9), 3))


tab = {lab: row(n) for lab, n in ARMS}
base = tab['base + S60']
best = max((k for k in tab if k != 'base + S60'), key=lambda k: tab[k]['mtm'])
better = sum(x > y for x, y in zip(tab[best]['mtm_by_day'], base['mtm_by_day']))
adopt = bool(tab[best]['mtm'] > base['mtm'] and better >= 6)
out = dict(days=DAYS, metric='mtm_stop', table=tab, best=best, best_days_better=better, adopted=adopt)
io.write_json(f'{S4}/report_lc.json', out)
print(f"{'variant':14s} {'mark@stop $':>11s} {'d+':>3s} {'t':>5s} {'c/sh':>6s} | {'carry $':>8s} {'carry sd/win':>12s} {'|net|':>5s} | {'gross $':>8s} {'shares':>7s} {'+1s c/sh':>8s}")
for k, r in tab.items():
    print(f"{k:14s} {r['mtm']:11.0f} {r['mtm_pos']:3d} {r['mtm_t']:5.2f} {r['mtm_c_sh']:6.2f} | {r['carry']:8.0f} {r['carry_sd_window']:12.2f} {r['abs_net_mean']:5.1f} | {r['gross']:8.0f} {r['shares']:7d} {r['mo1_c_sh']:8.3f}")
print(f"best by mark-at-stop: {best}, better than base on {better}/8 days -> adopted {adopt}")
for k in ('base + S60', best):
    print(f"  {k}: mark-at-stop by day {[round(x) for x in tab[k]['mtm_by_day']]}")
