"""Fill model v2 (code-review fixes, 2026-11-01) vs v1 for the key arms; C1's declared rule re-evaluated under v2.

v1 references: C1 arms from tokyo20_custom_e2_c1fix (fixed intent loop), lodo6 arms from the O2 run (same loop), O1 combo
early/late from o1combo (older loop, marked). Dev 10-23..25 | second look 10-26/27/31. mo1 = mark 1 s after the fill.
Usage: python3 -m predictfun.research.pf_v2_report
"""
import json, os
from . import pf_io as io

DEV = ['20261023', '20261024', '20261025']; C1 = ['20261026', '20261027', '20261031']; ALL = DEV + C1
def load(tag):
    out = {}
    for d in ALL:
        p = f'{io.OUT}/step4/{tag}/{d}/summary.json'
        if os.path.exists(p):
            out[d] = json.load(open(p))['arms']
    return out
V2C, V2O = load('tokyo20_custom_e2_v2c1'), load('tokyo20_custom_e2_preds_lodo6_v2o')
V1C, V1O, V1X = load('tokyo20_custom_e2_c1fix'), load('tokyo20_custom_e2_preds_lodo6_o2'), load('tokyo20_custom_e2_preds_lodo6_o1combo')
ST = 'MICRO+G4+G2+G6+G5+G3+G1'
def row(S, arm, days):
    if not all(d in S and arm in S[d] for d in days):
        return None
    g = [S[d][arm]['gross_usd'] for d in days]; sh = sum(S[d][arm]['shares'] for d in days); mo = sum(S[d][arm]['mo1_usd'] for d in days)
    return dict(by_day=g, sum=round(sum(g), 2), pos=sum(x > 0 for x in g), shares=round(sh), c_sh=round(100 * sum(g) / max(sh, 1e-9), 3),
                mo1_c_sh=round(100 * mo / max(sh, 1e-9), 3), inv_p95=max(S[d][arm]['abs_net_end_p95'] for d in days))
ROWS = [('C1 candidate D1+M05 (C1 preds)', V2C, V1C, f'{ST}+M05/mid'), ('  early', V2C, V1C, f'{ST}+M05/early'),
        ('  late', V2C, V1C, f'{ST}+M05/late'), ('D1 alone', V2C, V1C, f'{ST}/mid'),
        ('D1+M05 (lodo6)', V2O, V1O, f'{ST}+M05/mid'), ('O1 combo G1+G5+M025, 5 sh', V2O, V1O, 'MICRO+G5+G1+M025/mid'),
        ('  combo early', V2O, V1X, 'MICRO+G5+G1+M025/early'), ('  combo late', V2O, V1X, 'MICRO+G5+G1+M025/late'),
        ('D1+M025, 5 sh', V2O, V1O, f'{ST}+M025/mid'), ('D1+M025, 25 sh', V2O, V1O, f'{ST}+M025/mid@25'),
        ('D1+M025, A1 5..25 (>=0)', V2O, V1O, f'{ST}+M025/mid@A1:5:25:0'), ('  A1 early', V2O, None, f'{ST}+M025/early@A1:5:25:0'),
        ('  A1 late', V2O, None, f'{ST}+M025/late@A1:5:25:0'), ('D1+M025, A1 5..25 (>=0.25)', V2O, V1O, f'{ST}+M025/mid@A1:5:25:0.25'),
        ('combo, 25 sh', V2O, V1O, 'MICRO+G5+G1+M025/mid@25'), ('combo, A6 first 120 s', V2O, V1O, 'MICRO+G5+G1+M025/mid@A6:5:25:120'),
        ('combo, A1 5..25 (>=0)', V2O, V1O, 'MICRO+G5+G1+M025/mid@A1:5:25:0')]
out = {}
for lab, A, B, arm in ROWS:
    out[lab] = dict(arm=arm, v2=dict(dev=row(A, arm, DEV), c1=row(A, arm, C1)),
                    v1=dict(dev=row(B, arm, DEV), c1=row(B, arm, C1)) if B is not None else None)
cand, d1 = out['C1 candidate D1+M05 (C1 preds)']['v2']['c1'], out['D1 alone']['v2']['c1']
delta = [round(x - y, 2) for x, y in zip(cand['by_day'], d1['by_day'])]
out['C1_rule_v2'] = dict(candidate_sum=cand['sum'], candidate_minus_d1=delta, passes=bool(cand['sum'] > 0 and all(x > 0 for x in delta)))
# declared O2 adoption rule re-evaluated under v2 (vs fixed 25 on dev days, >= 2/3 days, +1 s mark not lower)
def adopt(arm_lab, ref_lab):
    a, r = out[arm_lab]['v2']['dev'], out[ref_lab]['v2']['dev']
    d = [x - y for x, y in zip(a['by_day'], r['by_day'])]
    return dict(delta_sum=round(sum(d), 2), days=sum(x > 0 for x in d), mo1_ok=a['mo1_c_sh'] >= r['mo1_c_sh'],
                adopted=bool(sum(d) > 0 and sum(x > 0 for x in d) >= 2 and a['mo1_c_sh'] >= r['mo1_c_sh']))
out['O2_rule_v2'] = {'B2 + A1(>=0)': adopt('D1+M025, A1 5..25 (>=0)', 'D1+M025, 25 sh'),
                     'B2 + A1(>=0.25)': adopt('D1+M025, A1 5..25 (>=0.25)', 'D1+M025, 25 sh'),
                     'B1 + A6': adopt('combo, A6 first 120 s', 'combo, 25 sh'), 'B1 + A1(>=0)': adopt('combo, A1 5..25 (>=0)', 'combo, 25 sh')}
io.write_json(f'{io.OUT}/step4/report_v2.json', out)
def fmt(r):
    return 'n/a' if r is None else f"{r['sum']:9.2f} {r['by_day']} mo1 {r['mo1_c_sh']:6.3f}"
for lab, v in out.items():
    if lab.endswith('_v2'):
        continue
    v1 = v['v1']
    print(f"{lab:32s} v2 dev {fmt(v['v2']['dev'])} | c1 {fmt(v['v2']['c1'])}  ||  v1 dev {fmt(v1['dev']) if v1 else 'n/a'} | c1 {fmt(v1['c1']) if v1 else 'n/a'}")
print('C1 rule under v2:', out['C1_rule_v2'])
print('O2 rule under v2:', out['O2_rule_v2'])
