"""INV-RULES (POST HOC, 2026-11-12, session ce68ebe0). USER: "Can we recheck our quoting logic, I remember there are skew/inventory
related logics that will influence our quoting price, will that influence our profit test, if we optimize the timing now, we can disable
those first".

Reads EXISTING outputs only; no new engine run. The in-sample runs i2J / i2X of the frozen pm_ladder_v3 engine (pf_step4_ladder_v3.py,
never edited) carry four arms that differ only in the inventory / leftover rules (the v3 build's in-sample comparison, chosen on these
same 9 consumed days, so every difference here is in-sample and descriptive):
  v3        ...+LD2+CAP10+XNA+XMIN1+XV+RO60+S240   (the frozen v3: cap 10 sh net, reduce-only from 60 s, stop 240 s)
  noCAP     ...+LD2+XNA+XMIN1+XV+RO60+S240         (no CAP10; G6 still caps at 2 x 25 = 50 sh)
  S60       ...+LD2+CAP10+XNA+XMIN1+XV+S60         (cap 10; no quoting at all from 60 s: no reduce-only phase)
  noCAP_S60 ...+LD2+XNA+XMIN1+XV+S60               (no CAP10, no quoting from 60 s)
No arm drops G6 (its soft cap 50 sh binds only without CAP10) and none quotes freely after 60 s; those would need new runs.
v3 has no inventory price skew: SK / RX / RL / A4 are not in its arm; MICRO + G4 choose the side from the best-level sizes.
Columns follow FINDINGS section 1 (C1 age split, w/o young, B1a-weighted; C3 cancel-pending share; ties at 50 for settlement), first
60 s and the whole window, plus inventory at the stop from each day's summary.json. No readings were declared: descriptive only.
Usage: predictfun/research/run_light.sh inv_rules -- python3 predictfun/research/pf_inv_rules_posthoc.py
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_fv3_check as fv3                                       # noqa: E402  (load / summarise / ties: the FV3 definitions)

OUTF = f'{fv3.IO}/inv_rules_20261112'
G = 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_'
ARMS = {'v3': G + 'CAP10_XNA_XMIN1_XV_RO60_S240__mid_szAM-5-25-95.parquet',
        'noCAP': G + 'XNA_XMIN1_XV_RO60_S240__mid_szAM-5-25-95.parquet',
        'S60': G + 'CAP10_XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet',
        'noCAP_S60': G + 'XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'}
KEYS = {a: f.replace('__mid_sz', '/mid@').replace('.parquet', '').replace('_', '+').replace('+mid@', '/mid@').replace('AM-5-25-95', 'AM:5:25:95')
        for a, f in ARMS.items()}


def main():
    os.makedirs(OUTF, exist_ok=True); D = {}; S = {}
    for tag in ('i2J', 'i2X'):
        for day in fv3.DAYS9:
            sm = json.load(open(f'{fv3.TAG(tag)}/{day}/summary.json'))['arms']
            for a, af in ARMS.items():
                fv3.ARMF = af
                f, _, _ = fv3.load(tag, day)
                D[(tag, a, day)] = dict(all=fv3.summarise(f), m1=fv3.summarise(f, f.t < 60.0),
                                        late=fv3.summarise(f, f.t >= 60.0))
                k = KEYS[a]
                if k not in sm:
                    sys.exit(f'REFUSED: arm key {k} not in {tag} {day} summary.json')
                s = sm[k]; w = s['withheld_decisions']
                S[(tag, a, day)] = dict(abs_net_stop_mean=s['abs_net_stop_mean'], carry_stop_usd=s['carry_stop_usd'],
                                        mtm_stop_usd=s['mtm_stop_usd'], abs_net_end_p95=s['abs_net_end_p95'],
                                        capn=w.get('CAPN', 0) + w.get('LD_CAPN', 0), ro=w.get('RO', 0) + w.get('LD_RO', 0),
                                        g6=w.get('G6_soft', 0) + w.get('LD_G6_soft', 0), minusd=w.get('MIN_USD', 0) + w.get('LD_MIN_USD', 0))
    # control: the v3 arm here equals FV3's committed i2J / i2X table
    ref = json.load(open(f'{fv3.OUTF}/fv3.json'))
    for tag in ('i2J', 'i2X'):
        mo = sum(D[(tag, 'v3', d)]['all']['mo1'] for d in fv3.DAYS9) / 9.0
        if abs(mo - ref[f'{tag}|all']['mo1']) > 1e-6:
            sys.exit(f'INV-RULES CONTROL FAILED: v3 {tag} {mo} != FV3 {ref[tag + "|all"]["mo1"]}')
    L = [f'INV-RULES (POST HOC) frozen v3 engine, in-sample runs i2J / i2X, 9 consumed replay days; as of '
         f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}; control passed (v3 arm = FV3 table); descriptive, no readings declared']
    RES = {}
    for part, nm in (('m1', 'fills in the first 60 s'), ('late', 'fills from 60 s'), ('all', 'whole window')):
        L.append(''); L.append(f'{nm}: $/day at 1 s - all | < 100 ms | 100-500 ms | >= 500 ms | w/o young | B1a M0 / M50 | settle (ties 50) | '
                               f'shares/day | pending share | days w/o-young > v3 (of 9)')
        for tag in ('i2J', 'i2X'):
            for a in ARMS:
                p = {k_: sum(D[(tag, a, d)][part][k_] for d in fv3.DAYS9) / 9.0 for k_ in D[(tag, a, fv3.DAYS9[0])][part]}
                up = int(sum(D[(tag, a, d)][part]['wo'] > D[(tag, 'v3', d)][part]['wo'] + 1e-9 for d in fv3.DAYS9))
                L.append(f'  {tag[-1]} {a:<10} {p["mo1"]:>7.2f} | {p["mo1|< 100 ms"]:>6.2f} | {p["mo1|100-500 ms"]:>6.2f} | {p["mo1|>= 500 ms"]:>6.2f} | '
                         f'{p["wo"]:>6.2f} | {p["w_M0"]:>5.2f} / {p["w_M50"]:>5.2f} | {p["settle"]:>7.2f} | {p["q"]:>6.0f} | '
                         f'{p["zombie_q"] / max(p["q"], 1e-9):.1%} | {"-" if a == "v3" else up}')
                RES[f'{tag}|{a}|{part}'] = p
        for tag in ('i2J', 'i2X'):
            for a in [a_ for a_ in ARMS if a_ != 'v3']:
                cnt = {k_: (int(sum(D[(tag, a, d)][part][k_] > D[(tag, 'v3', d)][part][k_] + 1e-9 for d in fv3.DAYS9)),
                            int(sum(D[(tag, a, d)][part][k_] < D[(tag, 'v3', d)][part][k_] - 1e-9 for d in fv3.DAYS9))) for k_ in ('mo1', 'wo', 'settle')}
                L.append(f'    {tag[-1]} {a} vs v3, days above / below (of 9): 1 s {cnt["mo1"]}, w/o young {cnt["wo"]}, settle {cnt["settle"]}')
                RES[f'{tag}|{a}|{part}|days_vs_v3'] = cnt
    L.append(''); L.append('inventory (per day; summary.json): mean |net| at the stop per window | p95 |net| at the end | carry stop -> settlement $ | '
                           'mark at the stop $ | decisions withheld by CAP10 / RO / G6 soft / XMIN1')
    for tag in ('i2J', 'i2X'):
        for a in ARMS:
            m = {k_: float(np.mean([S[(tag, a, d)][k_] for d in fv3.DAYS9])) for k_ in S[(tag, a, fv3.DAYS9[0])]}
            L.append(f'  {tag[-1]} {a:<10} {m["abs_net_stop_mean"]:>5.2f} | {m["abs_net_end_p95"]:>5.1f} | {m["carry_stop_usd"]:>7.2f} | {m["mtm_stop_usd"]:>7.2f} | '
                     f'{m["capn"]:>7.0f} / {m["ro"]:>7.0f} / {m["g6"]:>6.0f} / {m["minusd"]:>6.0f}')
            RES[f'{tag}|{a}|inv'] = m
    json.dump(dict(days=fv3.DAYS9, arms=ARMS, res=RES), open(f'{OUTF}/inv_rules.json', 'w'), indent=1, default=float)
    open(f'{OUTF}/inv_rules.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
