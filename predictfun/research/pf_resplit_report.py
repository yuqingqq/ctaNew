"""Re-split check (user 2026-11-03: "use first 2 days for validation"): rerun the O3 + O4 selection with training days
10-25, 10-26, 10-27, 10-31, 11-01, 11-02 and validation days 10-23, 10-24, by the same rules scaled to six days:
  stage 1 (O3 families on the O3 base): best training-sum variant adopted if training sum > base AND better than base on
    >= 4 of 6 training days AND training +1 s mark c/sh >= base's; adopted changes combined (C1)
  stage 2 (O4 families on C1): same, with >= 5 of 6 training days (O4's 6/8 scaled); only computable here if C1 equals O3's
    P3 (the O4 runs are on P3); otherwise the arms to run are printed
Reports the selected arm on the validation days, beside the base and O3's original P3. Caveat: read-day predictions come
from leave-one-day-out models that include the validation days in training (model leakage, not selection leakage).
Usage: python3 -m predictfun.research.pf_resplit_report
"""
import json, os
from . import pf_io as io

TR = ['20261025', '20261026', '20261027', '20261031', '20261101', '20261102']; VA = ['20261023', '20261024']; ALL = VA + TR
S4 = f'{io.OUT}/step4'
S = {d: {} for d in ALL}
for d in ALL:
    pre = 'preds_c3' if d >= '20261101' else 'preds_o3'
    for x in ('o3a', 'o3b', 'o3c', 'o4a', 'o4b', 'c3'):
        p = f'{S4}/live_tokyoreal_custom_e2_{pre}_{x}/{d}/summary.json'
        if os.path.exists(p):
            S[d].update(json.load(open(p))['arms'])
ST = 'MICRO+G4+G2+G6+G5+G3+G1'; X = 'XNA+XMIN1+XS0+XV'; BR = f'{ST}+M025+{X}'; A1 = '/mid@A1:5:25:0'
F1 = {'F2': [(t, f'{BR}+{t}{A1}') for t in ('B1T03', 'B2T03', 'B2W100', 'B2W100T03')],
      'F3': [(t, f'{BR}+{t}{A1}') for t in ('Q025', 'Q05', 'Q10')] + [(t, f'{ST}+{t}+{X}{A1}') for t in ('M015', 'M05')],
      'F4': [(t, f'{BR}+{t}{A1}') for t in ('BKA', 'BK60', 'BKM')],
      'F5': [(t, f'{BR}+{t}{A1}') for t in ('S60', 'S120', 'S180')] + [('A7:5:25:60', f'{BR}/mid@A7:5:25:60')],
      'F6': [(t, f'{BR}+{t}{A1}') for t in ('HY50', 'HY100', 'HY250')]}
P3R = f'{BR}+B2W100+Q025+BKA+S60'; X4 = f'{X}+B2W100'
F2_ = {'F7': [('BKD2', f'{P3R}+BKD2{A1}')], 'F8': [(t, f'{P3R}+{t}{A1}') for t in ('CD250', 'CD500', 'CD1000')],
       'F9': [(t, f'{P3R}+{t}{A1}') for t in ('FC500', 'FC2000')],
       'F10': [(t, f'{ST}+M025+{X4}+Q025+BKA+{t}{A1}') for t in ('S30', 'S45', 'S90')],
       'F11': [('fixed 5', f'{P3R}/mid'), ('A1:5:10:0', f'{P3R}/mid@A1:5:10:0')],
       'F12': [('M015', f'{ST}+M015+{X4}+Q025+BKA+S60{A1}'), ('Q015', f'{ST}+M025+{X4}+Q015+BKA+S60{A1}'), ('Q05', f'{ST}+M025+{X4}+Q05+BKA+S60{A1}')]}


def stat(name, days):
    if not all(name in S[d] for d in days):
        return None
    v = [S[d][name] for d in days]; g = [x['gross_usd'] for x in v]; sh = sum(x['shares'] for x in v)
    return dict(by_day=[round(x, 2) for x in g], sum=round(sum(g), 2), pos=sum(x > 0 for x in g), shares=round(sh),
                mo1_c_sh=round(100 * sum(x['mo1_usd'] for x in v) / max(sh, 1e-9), 3))


def select(base_name, fams, need):
    b = stat(base_name, TR); res = {}; adopted = {}
    for fam, vs in fams.items():
        rows = {t: stat(n, TR) for t, n in vs}
        ok = [t for t in rows if rows[t]]
        if not ok:
            res[fam] = dict(missing=True); continue
        best = max(ok, key=lambda t: rows[t]['sum']); r = rows[best]
        better = sum(x > y for x, y in zip(r['by_day'], b['by_day']))
        adopt = bool(r['sum'] > b['sum'] and better >= need and r['mo1_c_sh'] >= b['mo1_c_sh'])
        res[fam] = dict(best=best, train=r, days_better=better, adopted=adopt)
        if adopt:
            adopted[fam] = best
    return b, res, adopted


b1, r1, a1 = select(f'{BR}{A1}', F1, 4)
rules = BR.split('+'); size = 'A1:5:25:0'
for fam, t in a1.items():
    if t.startswith('A7'):
        size = t
    elif t.startswith('M0'):
        rules = [t if r == 'M025' else r for r in rules]
    else:
        rules.append(t)
order = {'B2W100': 0, 'Q025': 1, 'BKA': 2, 'S60': 3}
c1_name = '+'.join(rules) + ('/mid@' + size)
p3_tokens = {'B2W100', 'Q025', 'BKA', 'S60'}
c1_is_p3 = set(rules) - set(BR.split('+')) == p3_tokens and size == 'A1:5:25:0'
out = dict(train=TR, validation=VA, stage1=dict(base_train=b1, families=r1, adopted=a1, combined=c1_name, equals_O3_P3=c1_is_p3))
final = c1_name if stat(c1_name, ALL) else None
if c1_is_p3:
    b2, r2, a2 = select(P3R + A1, F2_, 5)
    out['stage2'] = dict(base_train=b2, families=r2, adopted=a2)
    if not a2:
        final = P3R + A1
    elif len(a2) == 1:
        final = next(n for fam, vs in F2_.items() if fam in a2 for t, n in vs if t == a2[fam])
    else:
        out['stage2']['note'] = 'several stage-2 changes adopted: their combination needs a new run'; final = None
else:
    out['stage2'] = dict(note=f'stage 1 picked {c1_name}, not P3: the O4 families must be run on it before stage 2')
out['final_arm'] = final
out['validation'] = {lab: stat(n, VA) for lab, n in (('final', final), ('O3 base', f'{BR}{A1}'), ('O3 P3 (original pick)', P3R + A1)) if n}
out['training'] = {lab: stat(n, TR) for lab, n in (('final', final), ('O3 base', f'{BR}{A1}'), ('O3 P3 (original pick)', P3R + A1)) if n}
io.write_json(f'{S4}/report_resplit_first2.json', out)
print('stage 1 adopted (training days):', a1, '-> combined', c1_name, '| equals O3 P3:', c1_is_p3)
for fam, v in r1.items():
    if 'best' in v:
        print(f"   {fam}: best {v['best']:10s} train {v['train']['sum']:8.0f} better {v['days_better']}/6 mo1 {v['train']['mo1_c_sh']:6.3f} adopted {v['adopted']}")
print('   base train', b1['sum'], b1['by_day'], 'mo1', b1['mo1_c_sh'])
if 'families' in out['stage2']:
    print('stage 2 adopted:', out['stage2']['adopted'])
    for fam, v in out['stage2']['families'].items():
        print(f"   {fam}: best {v['best']:10s} train {v['train']['sum']:8.0f} better {v['days_better']}/6 mo1 {v['train']['mo1_c_sh']:6.3f} adopted {v['adopted']}")
else:
    print('stage 2:', out['stage2']['note'])
print('final arm:', final)
for part in ('training', 'validation'):
    for lab, r in out[part].items():
        if r:
            print(f"  {part:10s} {lab:22s} {r['sum']:8.2f} by day {r['by_day']} mo1 {r['mo1_c_sh']:6.3f} shares {r['shares']}")
