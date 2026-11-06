"""Round O4 report (declared in STATUS.yml 2026-11-03 10:49Z): sweep protection on all 8 days (no hold-out).

Base P3 = full combined (D1 stack + M025 + XNA + XMIN1 + XS0 + XV + B2W100 + Q025 + BKA + S60, A1 5..25) at the Tokyo
clock, decision 2 ms. Adoption per family (computed): the best 8-day-sum variant only if its sum > P3's AND it beats P3
on >= 6 of 8 days AND its 8-day +1 s mark c/sh >= P3's. Prints the combined arm. Parity: P3 here equals O3's combined
sequential arm (read days) and C3's P3 (fresh days) to the cent.
Usage: python3 -m predictfun.research.pf_o4_report
"""
import json
from . import pf_io as io

READ = ['20261023', '20261024', '20261025', '20261026', '20261027', '20261031']; FRESH = ['20261101', '20261102']; ALL = READ + FRESH
S4 = f'{io.OUT}/step4'
S = {d: {} for d in ALL}
for d in ALL:
    pre = 'preds_c3' if d in FRESH else 'preds_o3'
    for x in ('o4a', 'o4b'):
        S[d].update(json.load(open(f'{S4}/live_tokyoreal_custom_e2_{pre}_{x}/{d}/summary.json'))['arms'])
ST = 'MICRO+G4+G2+G6+G5+G3+G1'; X = 'XNA+XMIN1+XS0+XV+B2W100'; P3R = f'{ST}+M025+{X}+Q025+BKA+S60'; A1 = '/mid@A1:5:25:0'
P3 = P3R + A1
FAM = {'F7 stand back 2 ticks': [('BKD2', f'{P3R}+BKD2{A1}')],
       'F8 touch cooldown': [(t, f'{P3R}+{t}{A1}') for t in ('CD250', 'CD500', 'CD1000')],
       'F9 fill cooldown': [(t, f'{P3R}+{t}{A1}') for t in ('FC500', 'FC2000')],
       'F10 stop time': [(t, f'{ST}+M025+{X}+Q025+BKA+{t}{A1}') for t in ('S30', 'S45', 'S90')],
       'F11 size': [('fixed 5', f'{P3R}/mid'), ('A1:5:10:0', f'{P3R}/mid@A1:5:10:0')],
       'F12 thresholds': [('M015', f'{ST}+M015+{X}+Q025+BKA+S60{A1}'), ('Q015', f'{ST}+M025+{X}+Q015+BKA+S60{A1}'),
                          ('Q05', f'{ST}+M025+{X}+Q05+BKA+S60{A1}')]}


def row(name):
    if not all(name in S[d] for d in ALL):
        return None
    v = [S[d][name] for d in ALL]; g = [x['gross_usd'] for x in v]; sh = sum(x['shares'] for x in v)
    return dict(by_day=g, sum=round(sum(g), 2), pos=sum(x > 0 for x in g), read=round(sum(g[:6]), 2), fresh=round(sum(g[6:]), 2),
                shares=round(sh), mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3))


base = row(P3)
o3c = [json.load(open(f'{S4}/live_tokyoreal_custom_e2_preds_o3_o3c/{d}/summary.json'))['arms'][P3]['gross_usd'] for d in READ]
c3 = [json.load(open(f'{S4}/live_tokyoreal_custom_e2_preds_c3_c3/{d}/summary.json'))['arms'][P3]['gross_usd'] for d in FRESH]
parity = base['by_day'] == o3c + c3
out = dict(base=base, parity=parity, families={}, adopted={})
for fam, vs in FAM.items():
    rows = {t: row(n) for t, n in vs}
    best = max((t for t in rows if rows[t]), key=lambda t: rows[t]['sum']); b = rows[best]
    better = sum(x > y for x, y in zip(b['by_day'], base['by_day']))
    adopt = bool(b['sum'] > base['sum'] and better >= 6 and b['mo1_c_sh'] >= base['mo1_c_sh'])
    out['families'][fam] = dict(variants=rows, best=best, days_better=better, adopted=adopt)
    if adopt:
        out['adopted'][fam] = best
rules = P3R.split('+'); size = 'A1:5:25:0'
for fam, t in out['adopted'].items():
    if fam.startswith('F10'):
        rules = [t if r == 'S60' else r for r in rules]
    elif fam.startswith('F11'):
        size = '5' if t == 'fixed 5' else t
    elif t.startswith('M0'):
        rules = [t if r == 'M025' else r for r in rules]
    elif t.startswith('Q0'):
        rules = [t if r == 'Q025' else r for r in rules]
    else:
        rules.append(t)
out['combined_arm'] = dict(sequential=f"MICRO|{'+'.join(rules[1:])}|mid|{size}", concurrent=f"MICRO|{'+'.join(r for r in rules[1:] if r != 'XS0')}|mid|{size}")
io.write_json(f'{S4}/live_tokyoreal_custom_e2_preds_o3_o4a/report_o4.json', out)
f = lambda r: f"8d {r['sum']:8.0f} ({r['pos']}/8) read {r['read']:7.0f} fresh {r['fresh']:7.0f} mo1 {r['mo1_c_sh']:6.3f} sh {r['shares']:6d}"
print('parity P3 == O3 combined (read) + C3 P3 (fresh):', parity)
print(f"{'P3 base':14s} {f(base)}  by day {[round(x) for x in base['by_day']]}")
for fam, v in out['families'].items():
    print(f"-- {fam}: best {v['best']} better on {v['days_better']}/8 days  ADOPTED={v['adopted']}")
    for t, r in v['variants'].items():
        print(f"   {t:12s} {f(r)}  by day {[round(x) for x in r['by_day']]}" if r else f'   {t:12s} n/a')
print('adopted:', out['adopted']); print('combined:', out['combined_arm'])
