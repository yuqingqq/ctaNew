"""Round O2 report (declared in STATUS.yml 2026-11-01 14:06Z): adaptive size, plus restatement with the fixed loop.

Bases B1 = MICRO+G5+G1+M025 (O1 combo), B2 = D1 stack + M025; sizes fixed 5 / 25 and adaptive 5..25 (A1 model, A3 depth,
A4 inventory, A6 time). Adoption (per base): beats fixed 25 on 10-23..25 in sum AND >= 2/3 days AND +1 s mark c/sh >= fixed 25.
Restatement: C1 candidate (D1 + M05, C1 preds) and O1 arms with the fixed loop vs their earlier numbers.
Usage: python3 -m predictfun.research.pf_o2_report
"""
import json
from . import pf_io as io

DEV = ['20261023', '20261024', '20261025']; C1 = ['20261026', '20261027', '20261031']
O = {d: json.load(open(f'{io.OUT}/step4/tokyo20_custom_e2_preds_lodo6_o2/{d}/summary.json'))['arms'] for d in DEV + C1}
F = {d: json.load(open(f'{io.OUT}/step4/tokyo20_custom_e2_c1fix/{d}/summary.json'))['arms'] for d in DEV + C1}
ST = 'MICRO+G4+G2+G6+G5+G3+G1'
def row(S, arm, days):
    g = [S[d][arm]['gross_usd'] for d in days]; sh = sum(S[d][arm]['shares'] for d in days); mo = sum(S[d][arm]['mo1_usd'] for d in days)
    return dict(by_day=g, sum=round(sum(g), 2), pos=sum(x > 0 for x in g), worst=min(g), shares=round(sh),
                c_sh=round(100 * sum(g) / max(sh, 1e-9), 3), mo1_c_sh=round(100 * mo / max(sh, 1e-9), 3),
                inv_p95=max(S[d][arm]['abs_net_end_p95'] for d in days))
SIZES = ['5', '25', 'A1:5:25:0', 'A1:5:25:0.25', 'A3:5:25:0.1', 'A3:5:25:0.25', 'A4:5:25:0', 'A6:5:25:120']
def arm(base, sz):
    return f'{base}/mid' + ('' if sz == '5' else ('@25' if sz == '25' else '@' + sz))
out = dict(bases={}, restate={})
for lab, base in (('B1 combo G1+G5+M025', 'MICRO+G5+G1+M025'), ('B2 D1+M025', f'{ST}+M025')):
    T = {sz: dict(dev=row(O, arm(base, sz), DEV), c1=row(O, arm(base, sz), C1)) for sz in SIZES}
    ref = T['25']
    for sz, v in T.items():
        d = [x - y for x, y in zip(v['dev']['by_day'], ref['dev']['by_day'])]
        v['dev_vs_fixed25'] = dict(by_day=[round(x, 2) for x in d], sum=round(sum(d), 2), pos=sum(x > 0 for x in d))
        v['adopted'] = bool(sz.startswith('A') and sum(d) > 0 and v['dev_vs_fixed25']['pos'] >= 2 and v['dev']['mo1_c_sh'] >= ref['dev']['mo1_c_sh'])
    out['bases'][lab] = T
out['restate'] = {
    'C1 candidate D1+M05 (C1 preds) mid': dict(fixed_loop=dict(dev=row(F, f'{ST}+M05/mid', DEV), c1=row(F, f'{ST}+M05/mid', C1)), earlier=dict(dev=237.33, c1=190.67)),
    'C1 candidate early': dict(fixed_loop=dict(c1=row(F, f'{ST}+M05/early', C1)), earlier=dict(c1=147.7)),
    'C1 candidate late': dict(fixed_loop=dict(c1=row(F, f'{ST}+M05/late', C1)), earlier=dict(c1=226.17)),
    'D1 alone mid': dict(fixed_loop=dict(dev=row(F, f'{ST}/mid', DEV), c1=row(F, f'{ST}/mid', C1)), earlier=dict(dev=-5.45, c1=-84.17)),
    'D1+M05 (lodo6) mid': dict(fixed_loop=dict(dev=row(O, f'{ST}+M05/mid', DEV), c1=row(O, f'{ST}+M05/mid', C1)), earlier=dict(dev=212.52, c1=188.54)),
    'O1 combo fixed 5': dict(fixed_loop=out['bases']['B1 combo G1+G5+M025']['5'], earlier=dict(dev=496.72, c1=679.82)),
    'D1+M025 fixed 5': dict(fixed_loop=out['bases']['B2 D1+M025']['5'], earlier=dict(dev=346.48, c1=191.61)),
    'D1+M025 fixed 25': dict(fixed_loop=out['bases']['B2 D1+M025']['25'], earlier=dict(dev=1751.21, c1=863.95)),
}
io.write_json(f'{io.OUT}/step4/report_o2.json', out)
for lab, T in out['bases'].items():
    print(f'\n{lab}  (dev 10-23..25 | second look 10-26/27/31)')
    for sz, v in T.items():
        a, c = v['dev'], v['c1']
        print(f"  {sz:13s} dev {a['sum']:9.2f} {a['by_day']} c/sh {a['c_sh']:6.3f} mo1 {a['mo1_c_sh']:6.3f} vs25 {v['dev_vs_fixed25']['sum']:8.2f} "
              f"({v['dev_vs_fixed25']['pos']}/3) | c1 {c['sum']:9.2f} {c['by_day']} c/sh {c['c_sh']:6.3f} mo1 {c['mo1_c_sh']:6.3f} | sh {a['shares'] + c['shares']:8d} "
              f"inv95 {c['inv_p95']:6.1f} {'ADOPTED' if v['adopted'] else ''}")
print('\nRESTATED with the fixed loop (earlier numbers in brackets)')
for k, v in out['restate'].items():
    fl = v['fixed_loop']
    print(f"  {k:36s} " + ' | '.join(f"{p} {fl[p]['sum']:9.2f} {fl[p]['by_day']} [{v['earlier'][p]}]" for p in ('dev', 'c1') if p in fl and p in v['earlier']))
