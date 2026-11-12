"""TB, the timing base and 'create the level' on the frozen v3 engine (2026-11-12, session ce68ebe0). USER: "We can remove the quoting
related limits first" / "we need to get front position in the queue" / "We can test, but finding missing information or explain the move
will be more important".

Engine: pf_step4_ladder_v3.py (generated, frozen; never edited), the v3 forward check's settings minus PF_FORWARD (as FV3), 9 consumed
replay days (10-23..27, 10-31, 11-01..03), arms through PF_ARMS only:
  B0    MICRO|G4+G2+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95       v3 without its quoting limits: no CAP10, no G6, no RO60;
                                                                                quotes stop at 60 s (first minute only); venue rules kept
  IM95  B0 + IM95: one tick inside a known spread of >= 2 ticks when v >= the minute's p95 line (the engine's IM rule; it refuses IM
        while CAP / RO tokens are present, so it becomes testable only with the limits off). Our order is then alone at a new price:
        first in the queue
  CTRL  MICRO|G4+G2+G6+G5+G3+G1+PM90+GS+LD2+XNA+XMIN1+XV+S60|mid|AM:5:25:95 (the in-sample arm noCAP_S60), control days only
Timings (run_tb.sh): tbJ (J: PF_TXJIT=frame), tbX (X: PF_TXSHIFT=-50 PF_QACT=max), tb250 (PF_TXSHIFT=-250 PF_QACT=max); order path 40 ms.
Columns as FV3 (FINDINGS section 1): $/day at 1 s by order age (< 100 / 100-500 / >= 500 ms), without young fills, B1a-weighted, settlement
with ties at 50, shares/day, cancel-pending share; plus queue ahead at activation (ahead0) over filled shares.

Readings DECLARED BEFORE THE RUNS (evaluated in code; day rule = >= 7 of 9 days; 'w/o young' = $/day at 1 s on orders >= 100 ms old):
 R360 LIMITS-OFF (B0 vs the in-sample v3 arm, fills in the first 60 s, J and X): w/o young above by the day rule at both -> HIGHER;
      below at both -> LOWER; else MIXED. Shares/day printed beside.
 R361 IMPROVE (IM95 vs B0, J and X): w/o young above at both -> GAIN; below at both -> LOSS; else NOT-SHOWN.
 R362 IMPROVE-AT-250 (IM95 vs B0 at tb250): w/o young above -> GAIN-AT-250; below -> LOSS-AT-250; else NOT-SHOWN.
 R363 FRONT (descriptive): share of filled shares with ahead0 <= 5 sh and the median ahead0, B0 vs IM95.
Controls (any failure -> no table): CTRL reproduces i2J / i2X's noCAP_S60 fills row for row on 10-23 and 11-03; B0 and IM95 withhold
nothing by CAPN / G6 / RO (the limits are really off); IM95 wants an improve on every day and B0 never (the rule is engaged, FINDINGS:
an off-parity control must run the override); the age bins add up.
Usage: research/run_tb.sh (runs + report); python3 predictfun/research/pf_tb_check.py report
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pf_fv3_check as fv3                                       # noqa: E402  (load / summarise / ties / days)

OUTF = f'{fv3.IO}/tb_20261112'
TAIL = 'XNA_XMIN1_XV_S60__mid_szAM-5-25-95.parquet'
ARMF = {'B0': 'MICRO_G4_G2_G5_G3_G1_PM90_GS_LD2_' + TAIL, 'IM95': 'MICRO_G4_G2_G5_G3_G1_PM90_GS_LD2_IM95_' + TAIL,
        'CTRL': 'MICRO_G4_G2_G6_G5_G3_G1_PM90_GS_LD2_' + TAIL}
V3F = fv3.ARMF
TAGS = {'tbJ': 'J', 'tbX': 'X', 'tb250': 'X250'}
REF = {'tbJ': 'i2J', 'tbX': 'i2X'}


def key_of(f):
    return f.replace('__mid_sz', '/mid@').replace('.parquet', '').replace('_', '+').replace('AM-5-25-95', 'AM:5:25:95')


def arm_sum(tag, af, day):
    fv3.ARMF = af
    f, _, _ = fv3.load(tag, day)
    d = fv3.summarise(f, f.t < 60.0)
    q = f.q.values[f.t.values < 60.0]; a0 = f.ahead0.values[f.t.values < 60.0]
    d['q_front5'] = float(q[a0 <= 5.0].sum()); d['ahead0_med'] = float(np.median(np.repeat(a0, np.maximum(np.round(q).astype(int), 1)))) if len(q) else np.nan
    return d


def report():
    os.makedirs(OUTF, exist_ok=True); bad = {}; D = {}
    for tag in TAGS:
        for day in fv3.DAYS9:
            sm = json.load(open(f'{fv3.TAG(tag)}/{day}/summary.json'))['arms']
            for a in ('B0', 'IM95'):
                s = sm[key_of(ARMF[a])]; w = s['withheld_decisions']; ctr = s['live'].get('counters', {})
                lim = sum(v for k, v in w.items() if k in ('CAPN', 'LD_CAPN', 'G6_soft', 'G6_hard', 'LD_G6_soft', 'LD_G6_hard', 'G6_final', 'RO', 'LD_RO'))
                if lim:
                    bad[f'limits on {a} {tag} {day}'] = lim
                imp = ctr.get('improve_wanted', 0)
                if (a == 'IM95' and imp <= 0) or (a == 'B0' and imp != 0):
                    bad[f'improve engaged {a} {tag} {day}'] = imp
                D[(tag, a, day)] = arm_sum(tag, ARMF[a], day)
                D[(tag, a, day)]['improve_wanted'] = float(imp)
                b = D[(tag, a, day)]
                if abs(sum(b[f'mo1|{x[0]}'] for x in fv3.BINS) - b['mo1']) > 1e-6:
                    bad[f'bins {a} {tag} {day}'] = 1
            if tag in REF:
                D[(tag, 'v3', day)] = arm_sum(REF[tag], V3F, day)
    for tag, ref in REF.items():
        for day in fv3.CTRL_DAYS:
            import pandas as pd
            a_ = pd.read_parquet(f'{fv3.TAG(ref)}/{day}/fills/{ARMF["CTRL"]}'); b_ = pd.read_parquet(f'{fv3.TAG(tag)}/{day}/fills/{ARMF["CTRL"]}')
            if len(a_) != len(b_) or not a_.reset_index(drop=True).equals(b_.reset_index(drop=True)):
                bad[f'control {tag} {day}'] = (len(a_), len(b_))
    if bad:
        sys.exit(f'TB CONTROL FAILED -> no table: {bad}')
    asof = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()); RES = dict(as_of=asof, days=fv3.DAYS9, arms=ARMF)
    L = [f'TB frozen v3 engine, limits off and IM95, 9 consumed replay days, fills in the first 60 s; as of {asof}; controls passed '
         f'(CTRL = i2 noCAP_S60 on {fv3.CTRL_DAYS}, limits withhold 0, IM engaged only in IM95, bins)', '',
         '$/day at 1 s - all | < 100 ms | 100-500 ms | >= 500 ms | w/o young | B1a M0 / M50 | settle (ties 50) | shares/day | pending share | '
         'shares ahead0 <= 5 sh | median ahead0 | improve decisions/day']
    dv = lambda tag, a, k_: np.array([D[(tag, a, d)][k_] for d in fv3.DAYS9])
    for tag, lab in TAGS.items():
        for a in (['v3'] if tag in REF else []) + ['B0', 'IM95']:
            p = {k_: dv(tag, a, k_).mean() for k_ in D[(tag, a, fv3.DAYS9[0])]}
            L.append(f'  {lab:<4} {a:<5} {p["mo1"]:>7.2f} | {p["mo1|< 100 ms"]:>6.2f} | {p["mo1|100-500 ms"]:>6.2f} | {p["mo1|>= 500 ms"]:>6.2f} | {p["wo"]:>6.2f} | '
                     f'{p["w_M0"]:>5.2f} / {p["w_M50"]:>5.2f} | {p["settle"]:>7.2f} | {p["q"]:>6.0f} | {p["zombie_q"] / max(p["q"], 1e-9):.1%} | '
                     f'{p["q_front5"] / max(p["q"], 1e-9):.1%} | {p["ahead0_med"]:>5.0f} | {p.get("improve_wanted", 0):>6.0f}')
            RES[f'{tag}|{a}'] = p
    def rule(t1, a1, t0, a0):
        return int(np.sum(dv(t1, a1, 'wo') > dv(t0, a0, 'wo') + 1e-9)), int(np.sum(dv(t1, a1, 'wo') < dv(t0, a0, 'wo') - 1e-9))
    L.append('')
    up = [rule(t, 'B0', t, 'v3') for t in REF]
    lab = 'HIGHER' if min(u for u, _ in up) >= 7 else ('LOWER' if min(d for _, d in up) >= 7 else 'MIXED')
    diff = [float((dv(t, 'B0', 'wo') - dv(t, 'v3', 'wo')).mean()) for t in REF]
    RES['R360'] = dict(label=lab, days_up_down_J_X=up, diff_wo_J_X=diff)
    L.append(f'R360 LIMITS-OFF (B0 vs v3, first 60 s): {lab}; days above / below J, X {up}; w/o young diff J / X ' + ' / '.join(f'{x:+.2f}' for x in diff) + ' $/day')
    up = [rule(t, 'IM95', t, 'B0') for t in REF]
    lab = 'GAIN' if min(u for u, _ in up) >= 7 else ('LOSS' if min(d for _, d in up) >= 7 else 'NOT-SHOWN')
    diff = [float((dv(t, 'IM95', 'wo') - dv(t, 'B0', 'wo')).mean()) for t in REF]
    RES['R361'] = dict(label=lab, days_up_down_J_X=up, diff_wo_J_X=diff)
    L.append(f'R361 IMPROVE (IM95 vs B0): {lab}; days above / below J, X {up}; w/o young diff J / X ' + ' / '.join(f'{x:+.2f}' for x in diff) + ' $/day')
    u, d_ = rule('tb250', 'IM95', 'tb250', 'B0')
    lab = 'GAIN-AT-250' if u >= 7 else ('LOSS-AT-250' if d_ >= 7 else 'NOT-SHOWN')
    RES['R362'] = dict(label=lab, days_up=u, days_down=d_, diff_wo=float((dv('tb250', 'IM95', 'wo') - dv('tb250', 'B0', 'wo')).mean()))
    L.append(f'R362 IMPROVE-AT-250: {lab}; days above {u}, below {d_}; w/o young diff {RES["R362"]["diff_wo"]:+.2f} $/day')
    fr = {a: (RES[f'tbJ|{a}']['q_front5'] / max(RES[f'tbJ|{a}']['q'], 1e-9), RES[f'tbJ|{a}']['ahead0_med']) for a in ('B0', 'IM95')}
    RES['R363'] = fr
    L.append(f'R363 FRONT (J): shares with ahead0 <= 5 sh / median ahead0: B0 {fr["B0"][0]:.1%} / {fr["B0"][1]:.0f}, IM95 {fr["IM95"][0]:.1%} / {fr["IM95"][1]:.0f}')
    L.append(''); L.append('per day w/o young $ (J): B0 ' + ' '.join(f'{x:.1f}' for x in dv('tbJ', 'B0', 'wo')) + '; IM95 ' + ' '.join(f'{x:.1f}' for x in dv('tbJ', 'IM95', 'wo')))
    json.dump(RES, open(f'{OUTF}/tb.json', 'w'), indent=1, default=float)
    open(f'{OUTF}/tb.txt', 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'report':
        report()
    else:
        sys.exit(__doc__)
