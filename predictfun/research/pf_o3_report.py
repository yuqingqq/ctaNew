"""Round O3 report (declared in STATUS.yml 2026-11-03 08:47Z): optimisation of live-like C2-P at the measured Tokyo clock.

Base = MICRO + D1 stack + M025 + XNA + XMIN1 + XS0 + XV, A1 5..25 (sequential, venue throttle), decision 2 ms, order live
40 / resting cancel 22 / pending cancel 95 after create / fills known 100 ms; preds_o3. Dev days 10-23..25; second look
10-26/27/31 (seen). Adoption per family: the best dev-sum variant is adopted only if dev sum > base AND it beats base on
>= 2/3 dev days AND dev +1 s mark c/sh >= base's (computed below). Prints the combined arm (base + adopted changes) for the
next step. Parity: base equals latency test L2c's decision-2 'sequential, venue' arm to the cent.
Usage: python3 -m predictfun.research.pf_o3_report
"""
import json, os
from . import pf_io as io

DEV = ['20261023', '20261024', '20261025']; SL = ['20261026', '20261027', '20261031']; ALL = DEV + SL
S4 = f'{io.OUT}/step4'
DIRS = ['live_tokyoreal_custom_e2_preds_o3_o3a', 'live_tokyoreal_custom_e2_preds_o3_o3b']
S = {d: {} for d in ALL}
for x in DIRS:
    for d in ALL:
        S[d].update(json.load(open(f'{S4}/{x}/{d}/summary.json'))['arms'])
ST = 'MICRO+G4+G2+G6+G5+G3+G1'; X = 'XNA+XMIN1+XS0+XV'; BASE_R = f'{ST}+M025+{X}'; A1 = '/mid@A1:5:25:0'
BASE = BASE_R + A1
FAM = {'F2 Binance guard': [(t, f'{BASE_R}+{t}{A1}') for t in ('B1T03', 'B2T03', 'B2W100', 'B2W100T03')],
       'F3 model guard': [(t, f'{BASE_R}+{t}{A1}') for t in ('Q025', 'Q05', 'Q10')] + [(t, f'{ST}+{t}+{X}{A1}') for t in ('M015', 'M05')],
       'F4 stand back': [(t, f'{BASE_R}+{t}{A1}') for t in ('BKA', 'BK60', 'BKM')],
       'F5 time': [(t, f'{BASE_R}+{t}{A1}') for t in ('S60', 'S120', 'S180')] + [('A7:5:25:60', f'{BASE_R}/mid@A7:5:25:60')],
       'F6 dead-band': [(t, f'{BASE_R}+{t}{A1}') for t in ('HY50', 'HY100', 'HY250')]}


def row(name):
    if not all(name in S[d] for d in ALL):
        return None
    v = {d: S[d][name] for d in ALL}

    def part(days):
        g = [v[d]['gross_usd'] for d in days]; sh = sum(v[d]['shares'] for d in days)
        return dict(by_day=g, sum=round(sum(g), 2), pos=sum(x > 0 for x in g), shares=round(sh),
                    mo1_c_sh=round(100 * sum(v[d]['mo1_usd'] for d in days) / max(sh, 1e-9), 3))
    k = {}
    for d in ALL:
        for kk, x in v[d]['by_kind'].items():
            k.setdefault(kk, dict(shares=0.0, settle=0.0, mo1=0.0))
            for f in ('shares', 'settle', 'mo1'):
                k[kk][f] += x[f]
    return dict(dev=part(DEV), sl=part(SL), all=part(ALL), by_kind={kk: {f: round(x[f], 1) for f in x} for kk, x in k.items()})


base = row(BASE)
l2c = f'{S4}/live_tokyoreal_custom_e2_preds_rl_a_d2_l2c_40_22_f100_d2'
par = None
if os.path.isdir(l2c):
    ref = [json.load(open(f'{l2c}/{d}/summary.json'))['arms'][BASE]['gross_usd'] for d in ALL]
    par = ref == base['dev']['by_day'] + base['sl']['by_day']
out = dict(base=base, parity_base_equals_l2c=par, families={}, adopted={})
for fam, vs in FAM.items():
    rows = {t: row(n) for t, n in vs}
    best = max((t for t in rows if rows[t]), key=lambda t: rows[t]['dev']['sum'])
    b = rows[best]; dd = [x - y for x, y in zip(b['dev']['by_day'], base['dev']['by_day'])]
    adopt = bool(b['dev']['sum'] > base['dev']['sum'] and sum(x > 0 for x in dd) >= 2 and b['dev']['mo1_c_sh'] >= base['dev']['mo1_c_sh'])
    out['families'][fam] = dict(variants=rows, best=best, best_vs_base_dev=dict(by_day=[round(x, 2) for x in dd], sum=round(sum(dd), 2)), adopted=adopt)
    if adopt:
        out['adopted'][fam] = best
rules = BASE_R.split('+'); size = 'A1:5:25:0'
for fam, t in out['adopted'].items():
    if t.startswith('A7'):
        size = t
    elif t.startswith('M'):
        rules = [t if r == 'M025' else r for r in rules]
    else:
        rules.append(t)
combo_seq = f"MICRO|{'+'.join(rules[1:])}|mid|{size}"
combo_conc = f"MICRO|{'+'.join(r for r in rules[1:] if r != 'XS0')}|mid|{size}"
out['combined_arm'] = dict(sequential=combo_seq, concurrent=combo_conc)
io.write_json(f'{S4}/{DIRS[0]}/report_o3.json', out)
f = lambda r: f"dev {r['dev']['sum']:8.0f} ({r['dev']['pos']}/3) mo1 {r['dev']['mo1_c_sh']:6.3f} | 2nd {r['sl']['sum']:8.0f} ({r['sl']['pos']}/3) mo1 {r['sl']['mo1_c_sh']:6.3f} | 6d {r['all']['sum']:8.0f}"
print('parity base == L2c decision-2 sequential venue:', par)
print(f"{'base':14s} {f(base)}")
for fam, v in out['families'].items():
    print(f'-- {fam}: best {v["best"]} vs base dev {v["best_vs_base_dev"]}  ADOPTED={v["adopted"]}')
    for t, r in v['variants'].items():
        print(f"   {t:12s} {f(r)}" if r else f'   {t:12s} n/a')
print('adopted:', out['adopted'])
print('combined arm (sequential):', combo_seq)
print('combined arm (concurrent):', combo_conc)
for nm in ('at', 'through'):
    x = base['by_kind'].get(nm)
    if x:
        print(f"base {nm:8s} shares {x['shares']:9.0f} settle {x['settle']:8.0f} mo1 {x['mo1']:7.0f} ({100 * x['mo1'] / max(x['shares'], 1):.2f} c/sh)")
