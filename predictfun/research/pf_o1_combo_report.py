"""Round O1 step 2 report: declared combination (MICRO + G1 + G5 + M025: theta 0.25, G2/G3/G4/G6 dropped) vs D1 + M05,
plus post-hoc arms (D1 + M025 at 5/25/100 sh; D1 + M05 at 25/100 sh), preds_lodo6, tokyo20, E2 timing.
Dev 10-23..25 | second look 10-26/27/31. mo1 = mark at +1 s (c/share), the short-horizon edge.
Usage: python3 -m predictfun.research.pf_o1_combo_report
"""
import json
from . import pf_io as io

DEV = ['20261023', '20261024', '20261025']; C1 = ['20261026', '20261027', '20261031']
A = {d: json.load(open(f'{io.OUT}/step4/tokyo20_custom_e2_preds_lodo6_o1combo/{d}/summary.json'))['arms'] for d in DEV + C1}
O = {d: json.load(open(f'{io.OUT}/step4/tokyo20_opt_e2_preds_lodo6/{d}/summary.json'))['arms'] for d in DEV + C1}
ST = 'MICRO+G4+G2+G6+G5+G3+G1'
def row(S, arm, days):
    g = [S[d][arm]['gross_usd'] for d in days]; sh = sum(S[d][arm]['shares'] for d in days); mo = sum(S[d][arm]['mo1_usd'] for d in days)
    return dict(by_day=g, sum=round(sum(g), 2), pos=sum(x > 0 for x in g), c_sh=round(100 * sum(g) / sh, 3), mo1_c_sh=round(100 * mo / sh, 3),
                shares=round(sh), inv_p95=max(S[d][arm]['abs_net_end_p95'] for d in days))
arms = [('D1 + M05 (base)', O, f'{ST}+M05/mid'), ('D1 + M025', O, f'{ST}+M025/mid'),
        ('combo G1+G5+M025 [declared]', A, 'MICRO+G5+G1+M025/mid'), ('combo early', A, 'MICRO+G5+G1+M025/early'),
        ('combo late', A, 'MICRO+G5+G1+M025/late'), ('D1 + M025 early', A, f'{ST}+M025/early'), ('D1 + M025 late', A, f'{ST}+M025/late'),
        ('D1 + M05 @25', A, f'{ST}+M05/mid@25'), ('D1 + M025 @25', A, f'{ST}+M025/mid@25'),
        ('D1 + M05 @100', A, f'{ST}+M05/mid@100'), ('D1 + M025 @100', A, f'{ST}+M025/mid@100')]
out = {}
for lab, S, arm in arms:
    out[lab] = dict(arm=arm, dev=row(S, arm, DEV), c1=row(S, arm, C1))
b = out['D1 + M05 (base)']
for lab in ('D1 + M025', 'combo G1+G5+M025 [declared]'):
    for part in ('dev', 'c1'):
        d = [x - y for x, y in zip(out[lab][part]['by_day'], b[part]['by_day'])]
        out[lab][f'{part}_delta_vs_base'] = dict(by_day=[round(x, 2) for x in d], sum=round(sum(d), 2), pos=sum(x > 0 for x in d))
io.write_json(f'{io.OUT}/step4/report_o1_combo.json', out)
for lab, v in out.items():
    print(f"{lab:30s} dev {v['dev']['sum']:9.2f} {v['dev']['by_day']} pos {v['dev']['pos']} c/sh {v['dev']['c_sh']:6.3f} mo1 {v['dev']['mo1_c_sh']:6.3f} | "
          f"c1 {v['c1']['sum']:9.2f} {v['c1']['by_day']} pos {v['c1']['pos']} c/sh {v['c1']['c_sh']:6.3f} mo1 {v['c1']['mo1_c_sh']:6.3f} | inv p95 {v['c1']['inv_p95']}")
for lab in ('D1 + M025', 'combo G1+G5+M025 [declared]'):
    print(lab, 'vs base: dev', out[lab]['dev_delta_vs_base'], 'c1', out[lab]['c1_delta_vs_base'])
