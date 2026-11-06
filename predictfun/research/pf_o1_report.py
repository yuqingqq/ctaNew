"""Round O1 report (declared in STATUS.yml 2026-11-01 11:46Z): size test + strategy/model variants.

Dev days 10-23..25 select; 10-26/27/31 are reported as a second look (seen by C1). Predicates are computed here.
Parity: the size armset's size-5 arms must equal the model armset's same arms (C1 preds) on every day.
Usage: python3 -m predictfun.research.pf_o1_report
Writes step4/report_o1.json.
"""
import json, os
from . import pf_io as io

DEV = ['20261023', '20261024', '20261025']; C1 = ['20261026', '20261027', '20261031']; ALL = DEV + C1
ST = 'MICRO+G4+G2+G6+G5+G3+G1'; BASE = f'{ST}+M05/mid'
def load(tag):
    return {d: json.load(open(f'{io.OUT}/step4/{tag}/{d}/summary.json'))['arms'] for d in ALL
            if os.path.exists(f'{io.OUT}/step4/{tag}/{d}/summary.json')}
out = {}

# ---- parity ----
S = load('tokyo20_size_e2'); M = load('tokyo20_model_e2')
par = {}
for d in ALL:
    if d in S and d in M:
        par[d] = {a: (S[d][a]['gross_usd'], M[d][a]['gross_usd'], S[d][a]['fills'] == M[d][a]['fills'])
                  for a in (f'{ST}/mid', BASE, f'{ST}+M05/early', f'{ST}+M05/late')}
out['parity_size5_vs_model_armset'] = par
out['parity_ok'] = all(abs(x - y) < 1e-6 and f for v in par.values() for x, y, f in v.values())

def row(A, arm, days):
    g = [A[d][arm]['gross_usd'] for d in days if d in A and arm in A[d]]
    sh = sum(A[d][arm]['shares'] for d in days if d in A and arm in A[d])
    return dict(by_day=g, sum=round(sum(g), 2), pos=sum(x > 0 for x in g), shares=round(sh), c_sh=round(100 * sum(g) / max(sh, 1e-9), 3),
                inv_p95=[A[d][arm]['abs_net_end_p95'] for d in days if d in A and arm in A[d]])

# ---- (a) size ----
sizes = {}
for sz in (5, 10, 25, 50, 100):
    suf = '' if sz == 5 else f'@{sz}'
    sizes[sz] = {k: dict(dev=row(S, a + suf, DEV), c1=row(S, a + suf, C1)) for k, a in
                 (('D1', f'{ST}/mid'), ('D1+M05', BASE), ('D1+M05 early', f'{ST}+M05/early'), ('D1+M05 late', f'{ST}+M05/late'))}
elig = [sz for sz in sizes if sizes[sz]['D1+M05']['dev']['pos'] >= 2]
pick = max(elig, key=lambda sz: sizes[sz]['D1+M05']['dev']['sum']) if elig else None
out['size'] = dict(table=sizes, eligible_dev_2of3=elig, pick_by_rule=pick)

# ---- (b)+(c) variants on preds_lodo6 ----
O = load('tokyo20_opt_e2_preds_lodo6')
if O:
    arms = sorted(next(iter(O.values())).keys())
    var = {a: dict(dev=row(O, a, DEV), c1=row(O, a, C1)) for a in arms}
    b = var[BASE]['dev']['by_day']
    for a, v in var.items():
        d = [x - y for x, y in zip(v['dev']['by_day'], b)]
        v['dev_delta_vs_base'] = dict(by_day=[round(x, 2) for x in d], sum=round(sum(d), 2), pos=sum(x > 0 for x in d))
    fam = {'theta': [f'{ST}+M025/mid', f'{ST}+M075/mid', f'{ST}+M10/mid'],
           '250ms': [f'{ST}+M05+Q025/mid', f'{ST}+M05+Q05/mid', f'{ST}+Q05/mid'],
           'stop': [f'{ST}+M05+S270/mid', f'{ST}+M05+S300/mid']}
    for g in ('G1', 'G2', 'G3', 'G4', 'G6'):
        fam[f'drop_{g}'] = ['MICRO+' + '+'.join(x for x in ['G4', 'G2', 'G6', 'G5', 'G3', 'G1'] if x != g) + '+M05/mid']
    adopt = {}
    for f_, cand in fam.items():
        cand = [c for c in cand if c in var]
        best = max(cand, key=lambda c: var[c]['dev']['sum']) if cand else None
        ok = best is not None and var[best]['dev_delta_vs_base']['sum'] > 0 and var[best]['dev_delta_vs_base']['pos'] >= 2
        adopt[f_] = dict(best=best, adopted=bool(ok), dev_delta=var[best]['dev_delta_vs_base'] if best else None)
    out['variants'] = var; out['adoption'] = adopt
    out['more_data'] = dict(base_c1_preds=dict(dev=row(S, BASE, DEV), c1=row(S, BASE, C1)), base_lodo6=var[BASE])
io.write_json(f'{io.OUT}/step4/report_o1.json', out)

print('parity ok:', out['parity_ok'])
print('\nSIZE (D1 + model guard, mid; dev 10-23..25 | 10-26/27/31)')
for sz, v in sizes.items():
    a, b_ = v['D1+M05'], v['D1']
    print(f"  {sz:>3} sh: dev {a['dev']['sum']:9.2f} {a['dev']['by_day']} pos {a['dev']['pos']} c/sh {a['dev']['c_sh']:6.3f} | c1 {a['c1']['sum']:9.2f} "
          f"{a['c1']['by_day']} pos {a['c1']['pos']} c/sh {a['c1']['c_sh']:6.3f} | early/late c1 {v['D1+M05 early']['c1']['sum']}/{v['D1+M05 late']['c1']['sum']} "
          f"| D1 alone dev {b_['dev']['sum']} c1 {b_['c1']['sum']} | inv p95 {a['c1']['inv_p95']}")
print('  pick by rule:', pick)
if O:
    print('\nVARIANTS (preds_lodo6, size 5, mid)')
    for a in sorted(var, key=lambda x: -var[x]['dev']['sum']):
        v = var[a]
        print(f"  {a.replace('MICRO+', ''):34s} dev {v['dev']['sum']:9.2f} {v['dev']['by_day']} d_base {v['dev_delta_vs_base']['sum']:8.2f} "
              f"({v['dev_delta_vs_base']['pos']}/3) | c1 {v['c1']['sum']:9.2f} {v['c1']['by_day']} pos {v['c1']['pos']} c/sh {v['c1']['c_sh']}")
    print('  adoption:', {k: (v['best'].replace('MICRO+', '') if v['best'] else None, v['adopted']) for k, v in adopt.items()})
