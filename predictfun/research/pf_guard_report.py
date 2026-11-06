"""Stage 3 report (declared in STATUS.yml 2026-11-04): the trained per-side guard in the replay vs the current guard, against the
top-maker target (bad fills <= ~10% of shares, bad $ <= ~45-50% of good $, net ~+0.3 c/sh at 1 s; top 5: 9.0%, 40%, +0.42).
Per arm: mark-at-stop on the 6 read days / 11-01-02, days > 0, shares, +1 s c/sh, bad share (<= -2 c/sh at 1 s), bad $ / good $.
Parity: the reference arm run on preds_g20 (pred_1000 copied) must equal the anchor test's real-model P60 day for day.
Usage: python3 -m predictfun.research.pf_guard_report [fresh_day ...]
"""
import json, sys
import pandas as pd
from . import pf_io as io

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; FR = ('20261101', '20261102')
NEW = sys.argv[1:]; S4 = f'{io.OUT}/step4'; B = 'MICRO+G4+G2+G6+G5+G3+G1'
REF = f'{B}+M025+XNA+XMIN1+S60/mid@A1:5:25:0'
stem = lambda n: n.replace('/', '__').replace('+', '_').replace('@', '_sz').replace(':', '-')


def stats(days, pre, n, tag):
    S = {d: json.load(open(f'{S4}/live_tokyoreal_custom_e2_{pre(d)}_{tag}/{d}/summary.json'))['arms'][n] for d in days}
    f = pd.concat([pd.read_parquet(f'{S4}/live_tokyoreal_custom_e2_{pre(d)}_{tag}/{d}/fills/{stem(n)}.parquet', columns=['q', 'mo1']) for d in days])
    c = 100 * f.mo1 / f.q; q = f.q.sum(); bad = c <= -2; good = c >= 0
    m = [S[d]['mtm_stop_usd'] for d in days]
    return dict(mtm=m, shares=float(q), mo1=float(100 * f.mo1.sum() / q), bad_share=float(f.q[bad].sum() / q), bad_over_good=float(-f.mo1[bad].sum() / f.mo1[good].sum()))


# parity
a = stats(READ, lambda d: 'preds_c3' if d in FR else 'preds_o3', REF, 'anc'); b = stats(READ, lambda d: 'preds_g20', REF, 'gd')
if a['mtm'] != b['mtm'] or abs(a['shares'] - b['shares']) > 1e-6:
    raise SystemExit(f"PARITY FAIL: reference arm differs on preds_g20: {a['mtm']} vs {b['mtm']}")
print('parity: the reference arm on preds_g20 equals the anchor test run, day for day')
ARMS = [('current guard (M025)', 20, REF)] + [(f'{lab} k={k}%', k, f"{B}+{'V0' if lab == 'replace' else 'M025+V0'}+XNA+XMIN1+S60/mid@A1:5:25:0")
                                            for lab in ('replace', 'add') for k in (10, 20, 30)]
out = {}
print(f"\n{'arm':24s} {'read mark@stop':>14s} {'11-01/02':>8s} {'d+':>4s} {'shares':>7s} {'+1s c/sh':>8s} {'bad %':>6s} {'bad$/good$':>10s}   (target: bad <= ~10%, ratio <= ~45-50%, +0.3)")
for lab, k, n in ARMS:
    r = stats(READ, lambda d: f'preds_g{k}', n, 'gd'); out[lab] = r
    print(f"{lab:24s} {sum(r['mtm'][:6]):14.0f} {sum(r['mtm'][6:]):8.0f} {sum(x > 0 for x in r['mtm']):2d}/8 {r['shares']:7.0f} {r['mo1']:8.3f} {r['bad_share']:6.1%} {r['bad_over_good']:10.0%}")
for d in NEW:
    print(f'\nfresh day {d} (no model here has seen it):')
    for lab, k, n in ARMS:
        r = stats([d], lambda d_: f'preds_g{k}', n, 'gd'); out[f'{d}|{lab}'] = r
        print(f"  {lab:22s} mark@stop {r['mtm'][0]:7.0f}  shares {r['shares']:6.0f}  +1 s {r['mo1']:6.3f} c/sh  bad {r['bad_share']:5.1%}  bad$/good$ {r['bad_over_good']:4.0%}")
io.write_json(f'{S4}/report_guard_stage3.json', out)
