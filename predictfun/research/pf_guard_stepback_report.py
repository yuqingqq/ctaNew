"""Step-back vs withhold for the trained guard (declared in STATUS.yml 2026-11-04): at pull k = 50 / 60 / 70%, a flagged side is
either withheld (V0, tag gd), quoted one tick behind the best (V0 + SBV, tag gds) or re-priced from the 1 s prediction (V0 + SBP, tag gdp), replacing M025 ('instead') or added to it
('added'). Parity: the withhold arm rerun in the gds runs must equal its gd run day for day. Per arm: mark-at-stop on the 6 read
days / 11-01-02, days > 0, shares, +1 s c/sh, bad share (<= -2 c/sh at 1 s), bad $ / good $; then 11-03 (unseen).
Usage: python3 -m predictfun.research.pf_guard_stepback_report
"""
import json, os
import pandas as pd
from . import pf_io as io

D8 = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')
arm = lambda tok: f'{B}+{tok}+XNA+XMIN1+S60/mid@A1:5:25:0'


def stats(days, k, n, tag):
    S = {d: json.load(open(f'{S4}/live_tokyoreal_custom_e2_preds_g{k}_{tag}/{d}/summary.json'))['arms'][n] for d in days}
    f = pd.concat([pd.read_parquet(f'{S4}/live_tokyoreal_custom_e2_preds_g{k}_{tag}/{d}/fills/{stem(n)}.parquet', columns=['q', 'mo1']) for d in days])
    c = 100 * f.mo1 / f.q; q = f.q.sum(); bad = c <= -2; good = c >= 0
    m = [S[d]['mtm_stop_usd'] for d in days]
    return dict(mtm=m, shares=float(q), mo1=float(100 * f.mo1.sum() / q), bad=float(f.q[bad].sum() / q), ratio=float(-f.mo1[bad].sum() / f.mo1[good].sum()))


for k in (50, 60, 70):
    a = stats(D8 + ['20261103'], k, arm('V0'), 'gd'); b = stats(D8 + ['20261103'], k, arm('V0'), 'gds')
    if a['mtm'] != b['mtm'] or abs(a['shares'] - b['shares']) > 1e-6:
        raise SystemExit(f'PARITY FAIL at k={k}: withhold arm differs between gd and gds runs')
print('parity: the withhold arm rerun equals its earlier run at k = 50 / 60 / 70, day for day')
out = {}
print(f"\n{'arm':30s} {'read mark@stop':>14s} {'11-01/02':>8s} {'d+':>4s} {'shares':>7s} {'+1s c/sh':>8s} {'bad %':>6s} {'ratio':>6s} | 11-03: {'mark':>6s} {'+1s':>7s} {'bad %':>6s} {'ratio':>6s}")
for k in (50, 60, 70):
    for lab, tok, tag in (('withhold instead', 'V0', 'gd'), ('step back instead', 'V0+SBV', 'gds'), ('pred step instead', 'V0+SBP', 'gdp'),
                          ('withhold added', 'M025+V0', 'gd'), ('step back added', 'M025+V0+SBV', 'gds'), ('pred step added', 'M025+V0+SBP', 'gdp')):
        if not os.path.exists(f'{S4}/live_tokyoreal_custom_e2_preds_g{k}_{tag}/20261103/summary.json'):
            print(f"{lab + f' {k}%':30s} (not run yet)"); continue
        r = stats(D8, k, arm(tok), tag); f = stats(['20261103'], k, arm(tok), tag); out[f'{k}|{lab}'] = dict(read=r, fresh=f)
        print(f"{lab + f' {k}%':30s} {sum(r['mtm'][:6]):14.0f} {sum(r['mtm'][6:]):8.0f} {sum(x > 0 for x in r['mtm']):2d}/8 {r['shares']:7.0f} {r['mo1']:8.3f} {r['bad']:6.1%} {r['ratio']:6.0%} | "
              f"{f['mtm'][0]:13.0f} {f['mo1']:+7.3f} {f['bad']:6.1%} {f['ratio']:6.0%}")
io.write_json(f'{S4}/report_guard_stepback.json', out)
